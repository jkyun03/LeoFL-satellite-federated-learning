# scenarios_contact.py
#
# Contact-aware LH-FL scenario  (Design B: opportunistic per-round leader mixing)
#
# Motivation
# ----------
# The original OffloadScenario (LH-FL) mixes leaders over a STATIC top-k graph
# every aggregation cycle, ignoring whether the two leaders are actually in
# range. At max_dist_km=2000 most of that graph is physically impossible
# (fallback ring edges between leaders that never meet). This scenario fixes it
# for a fair comparison against FedISL (which models sink->PS contact waiting):
#
#   * a realistic ISL range (max_dist_km ~= 5000 km, within the ~5418 km LOS
#     horizon at 552 km altitude) is used, and
#   * inter-plane leader mixing follows the paper's own Eq.(9) neighbour set
#     N_p^(r) = "leaders connected to l_p IN ROUND r": each cycle, leader p mixes
#     only with the neighbours it is actually in contact with at that moment
#     (beta re-normalised over the available neighbours).
#
# Because the persistent (always-in-range) leader edges keep the leader graph
# connected at EVERY round (validated on the geometry), consensus still
# propagates with NO artificial waiting, so LH-FL stays fast while every mixing
# uses a genuine contact. Communication is counted only for edges actually used.
#
# use_real_contacts=False reproduces the idealized "mix all static neighbours"
# behaviour (for an ablation of the fix). Both modes share the clean per-cycle
# synchronous timing model (T_round ~ max_i T_i), matching Eq.(4).
#
# intra-plane aggregation stays instantaneous (intra-plane paths assumed
# available, consistent with the paper and with FedISL's intra-plane ISL ring).
# Everything else (capacity model, one-shot offloading, comm accounting, CSV
# outputs) is inherited unchanged from OffloadScenario.

import os
import copy
from typing import Optional
import numpy as np
import pandas as pd

from LeoFL.Simulator.fl_core import (
    build_fixed_constellation,
    compute_orbits_for_rounds,
    reindex_satellites_by_orbital_phase,
    precompute_meetings_and_select_leaders,
    offload_balance_datasets,
    save_class_distribution_csv,
    build_transfer_plan_within_plane,
    sample_train_rounds_with_exp_jitter,
    train_one_epoch,
    SimConfig,
    filter_by_link_failure,
    jitter_hop_count,
    LINK_FAIL_P_DEFAULT,
)

from LeoFL.Simulator.scenarios import (
    OffloadScenario,
    ring_hop_distance,
    _build_topk_leader_plane_neighbors,
    _mix_self_with_neighbor_average,
    _weighted_average_state_dict_from_models,
    _evaluate_weighted_model_accuracy,
)


class ContactAwareOffloadScenario(OffloadScenario):
    """
    LH-FL with opportunistic per-round inter-plane leader mixing (Design B).

    Extra args beyond OffloadScenario:
        use_real_contacts : if True (default), each cycle a leader mixes only
                            with neighbours actually in contact that round
                            (paper Eq.9 N_p^(r)). If False, mixes with all static
                            top-k neighbours (idealized, for ablation).
        contact_horizon_mult : how many times num_rounds to precompute leader
                            contact timelines for (lookahead buffer).
        use_relay : if True, a leader-to-leader edge (p, q) also counts as "in
                            contact" when leader_p (or leader_q) is within
                            max_dist_km of ANY ordinary satellite belonging to
                            the other plane, not just the other plane's leader.
                            Ordinary satellites never aggregate/mix themselves —
                            they only carry an already-current copy of their own
                            plane's model (from the existing intra-plane
                            broadcast), so this only relaxes the CONTACT check;
                            the static top-k leader_neighbors graph and the
                            mixing math (uniform average + beta) are unchanged.
                            Default False reproduces the original leader-only
                            contact behaviour exactly.
        relay_max_age_rounds : reserved for a future "drop relay contact if the
                            carried model is older than N rounds" rule. Not
                            enforced yet (current policy is A-1: no age limit,
                            use whatever contact is found) — kept as an explicit
                            hook so that rule can be added without touching the
                            call sites.
    """

    def __init__(
        self,
        client_datasets,
        test_dataset,
        config: SimConfig,
        max_dist_km: float = 5000.0,
        offload_ratio: float = 0.1,
        top_pairs_per_sat: int = 1,
        log_prefix: str = "sat_events_offload_contact",
        do_offload: bool = True,
        per_sat_minutes=None,
        train_minutes: int = 10,
        leader_top_k: int = 2,
        leader_mixing_beta: float = 0.5,
        use_real_contacts: bool = True,
        contact_horizon_mult: int = 2,
        inter_plane_mode: str = "mixing",
        use_relay: bool = False,
        relay_max_age_rounds: Optional[int] = None,
        link_fail_rng: Optional[np.random.Generator] = None,
        link_fail_p: float = LINK_FAIL_P_DEFAULT,
    ):
        super().__init__(
            client_datasets=client_datasets,
            test_dataset=test_dataset,
            config=config,
            max_dist_km=max_dist_km,
            offload_ratio=offload_ratio,
            top_pairs_per_sat=top_pairs_per_sat,
            log_prefix=log_prefix,
            do_offload=do_offload,
            per_sat_minutes=per_sat_minutes,
            train_minutes=train_minutes,
            leader_top_k=leader_top_k,
            leader_mixing_beta=leader_mixing_beta,
            link_fail_rng=link_fail_rng,
            link_fail_p=link_fail_p,
        )
        self.use_real_contacts = use_real_contacts
        self.contact_horizon_mult = int(max(1, contact_horizon_mult))
        assert inter_plane_mode in ("mixing", "central"), (
            "inter_plane_mode must be 'mixing' (LH-FL, B3) or 'central' (hierarchical, B2)"
        )
        self.inter_plane_mode = inter_plane_mode
        self.use_relay = use_relay
        self.relay_max_age_rounds = relay_max_age_rounds  # unused for now (A-1 policy)

    # ------------------------------------------------------------------
    def _leader_contact_timelines(
        self, sats, leaders_per_plane, leader_neighbors, horizon
    ):
        """
        edges      : set of undirected (p, q) leader-graph edges
        contact_tl : dict edge -> boolean np.array (len horizon), True where the
                     two plane leaders are within max_dist_km at that round.
        """
        spr = self.cfg.seconds_per_round
        times = np.arange(0, horizon * spr, spr)

        pos = {}
        for p, li in leaders_per_plane.items():
            s = copy.copy(sats[li])  # shallow copy; compute_leo_orbit reassigns arrays
            s.compute_leo_orbit(times)
            # read positions via the confirmed LEOSat API (do NOT assume .x/.y/.z arrays)
            pos[p] = np.array(
                [s.get_leo_orbit_position(f) for f in range(horizon)], dtype=float
            )  # (horizon, 3) meters

        edges = set()
        for p in leader_neighbors:
            for q in leader_neighbors[p]:
                if p < q:
                    edges.add((p, q))

        contact_tl = {}
        for p, q in edges:
            d_km = np.sqrt(((pos[p] - pos[q]) ** 2).sum(axis=1)) / 1000.0
            contact_tl[(p, q)] = d_km <= self.max_dist_km
        return edges, contact_tl

    def _relay_contact_timelines(
        self, sats, leaders_per_plane, leader_neighbors, plane_to_sats, horizon
    ):
        """
        For every static leader-graph edge (p, q), compute a boolean timeline of
        whether an ordinary-satellite relay path is available that round:
            leader_p is within max_dist_km of ANY satellite in plane q, OR
            leader_q is within max_dist_km of ANY satellite in plane p.

        Ordinary satellites already carry a synced copy of their own plane's
        model (via the existing per-cycle intra-plane broadcast), so "leader_p
        meets some member of plane q" is treated as equivalent to leader_p
        receiving plane q's current model — no extra relay hop bookkeeping is
        needed on the sending side, since that member got the model as part of
        the intra-plane broadcast cost that's already accounted for.

        Returns (edges, relay_tl) with the same shapes/conventions as
        _leader_contact_timelines. Note this timeline is a superset of the
        direct leader-leader timeline (members_of_plane includes the leader
        itself), so callers should OR it with the direct timeline (or just use
        it directly) rather than expecting it to be relay-only.
        """
        spr = self.cfg.seconds_per_round
        times = np.arange(0, horizon * spr, spr)

        edges = set()
        for p in leader_neighbors:
            for q in leader_neighbors[p]:
                if p < q:
                    edges.add((p, q))

        touched_planes = set()
        for p, q in edges:
            touched_planes.add(p)
            touched_planes.add(q)

        needed_idx = set()
        for p in touched_planes:
            needed_idx.update(int(i) for i in plane_to_sats.get(p, []))

        pos_cache = {}
        for idx in needed_idx:
            s = copy.copy(sats[idx])  # shallow copy; compute_leo_orbit reassigns arrays
            s.compute_leo_orbit(times)
            pos_cache[idx] = np.array(
                [s.get_leo_orbit_position(f) for f in range(horizon)], dtype=float
            )  # (horizon, 3) meters

        relay_tl = {}
        for p, q in edges:
            leader_p_pos = pos_cache[int(leaders_per_plane[p])]
            leader_q_pos = pos_cache[int(leaders_per_plane[q])]

            min_d_p_to_q = np.full(horizon, np.inf)
            for j in plane_to_sats.get(q, []):
                d_km = (
                    np.sqrt(((leader_p_pos - pos_cache[int(j)]) ** 2).sum(axis=1))
                    / 1000.0
                )
                min_d_p_to_q = np.minimum(min_d_p_to_q, d_km)

            min_d_q_to_p = np.full(horizon, np.inf)
            for i in plane_to_sats.get(p, []):
                d_km = (
                    np.sqrt(((leader_q_pos - pos_cache[int(i)]) ** 2).sum(axis=1))
                    / 1000.0
                )
                min_d_q_to_p = np.minimum(min_d_q_to_p, d_km)

            relay_tl[(p, q)] = (min_d_p_to_q <= self.max_dist_km) | (
                min_d_q_to_p <= self.max_dist_km
            )

        return edges, relay_tl

    def _available_neighbors(self, p, leader_neighbors, contact_tl, at_round):
        """Neighbours of leader p that are in contact at `at_round` (Eq.9 N_p^(r))."""
        avail = []
        fr = int(at_round)
        for q in sorted(leader_neighbors.get(p, set())):
            if not self.use_real_contacts:
                avail.append(q)
                continue
            e = (min(p, q), max(p, q))
            tl = contact_tl.get(e)
            in_contact = True if (tl is None or fr >= len(tl)) else bool(tl[fr])
            if in_contact:
                avail.append(q)
        return avail

    # ------------------------------------------------------------------
    def run(self):
        n_clients = self.n_clients
        print(
            f"[SIM][offload-contact] start: num_rounds={self.cfg.num_rounds}, "
            f"n_clients={n_clients}, use_real_contacts={self.use_real_contacts}, "
            f"max_dist_km={self.max_dist_km}"
        )

        # --- offloading 전 클래스 분포 ---
        before_path = os.path.join(
            self.cfg.log_dir or ".", f"{self.log_prefix}_classdist_before.csv"
        )
        save_class_distribution_csv(self.client_datasets, before_path)

        # 1) 위성/궤도
        configs = self.cfg.constellation_configs[:n_clients]
        sats = build_fixed_constellation(configs)
        assert len(sats) == n_clients
        compute_orbits_for_rounds(
            sats,
            num_rounds=self.cfg.num_rounds,
            seconds_per_round=self.cfg.seconds_per_round,
        )
        plane_ids = [c.get("plane_id", 0) for c in configs]
        sats, plane_ids = reindex_satellites_by_orbital_phase(
            sats, plane_ids, round_idx=0
        )

        # 2) meetings + leaders (max_dist_km 반영)
        meeting_counts, leaders_per_plane, plane_to_sats = (
            precompute_meetings_and_select_leaders(
                sats,
                plane_ids=plane_ids,
                num_rounds=self.cfg.num_rounds,
                seconds_per_round=self.cfg.seconds_per_round,
                max_dist_km=self.max_dist_km,
                log_dir=self.cfg.log_dir,
                filename=f"{self.log_prefix}_meeting_counts.csv",
            )
        )
        print("[LEADER][offload-contact] per plane:")
        for p, li in leaders_per_plane.items():
            print(f"  plane={p}, leader_sat=sat_{li + 1}, members={plane_to_sats[p]}")

        leader_neighbors = _build_topk_leader_plane_neighbors(
            leaders_per_plane=leaders_per_plane,
            meeting_counts=meeting_counts,
            top_k=self.leader_top_k,
        )
        print("[LEADER-GRAPH][offload-contact] inter-orbit leader neighbors:")
        for p in sorted(leader_neighbors.keys()):
            print(f"  plane={p} -> {sorted(leader_neighbors[p])}")

        # -------- overview logs --------
        overview_dir = self.cfg.log_dir or "."
        os.makedirs(overview_dir, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "plane_id": int(p),
                    "local_idx": int(k),
                    "sat_id": int(sid),
                    "sat_name": f"sat_{sid + 1}",
                }
                for p, si in plane_to_sats.items()
                for k, sid in enumerate(si)
            ]
        ).to_csv(
            os.path.join(overview_dir, f"{self.log_prefix}_plane_members.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(
            [
                {
                    "plane_id": int(p),
                    "leader_sat_id": int(li),
                    "leader_name": f"sat_{li + 1}",
                }
                for p, li in leaders_per_plane.items()
            ]
        ).to_csv(
            os.path.join(overview_dir, f"{self.log_prefix}_leaders.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        lex = []
        for p in sorted(leader_neighbors.keys()):
            for q in sorted(leader_neighbors[p]):
                if int(p) < int(q):
                    a, b = int(leaders_per_plane[p]), int(leaders_per_plane[q])
                    lex.append(
                        {
                            "src_plane": int(p),
                            "dst_plane": int(q),
                            "src_leader": a,
                            "dst_leader": b,
                            "src_name": f"sat_{a + 1}",
                            "dst_name": f"sat_{b + 1}",
                        }
                    )
        pd.DataFrame(lex).to_csv(
            os.path.join(overview_dir, f"{self.log_prefix}_leader_exchange.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        # -------- capacity (offload 전) --------
        cap_path = os.path.join(
            self.cfg.log_dir or ".", f"{self.log_prefix}_capacity.csv"
        )
        pd.DataFrame(
            [
                {
                    "sat_id": i,
                    "sat_name": f"sat_{i + 1}",
                    "plane_id": plane_ids[i],
                    "train_capacity": float(cap),
                }
                for i, cap in enumerate(self.train_capacities)
            ]
        ).to_csv(cap_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE][CAPACITY] {cap_path}")

        # 3) capacity-aware one-shot offloading
        if self.do_offload:
            offload_stats = offload_balance_datasets(
                client_datasets=self.client_datasets,
                meeting_counts=meeting_counts,
                plane_ids=plane_ids,
                offload_ratio=self.offload_ratio,
                top_pairs_per_sat=self.top_pairs_per_sat,
                seed=0,
                train_capacities=self.train_capacities,
            )
        else:
            offload_stats = None

        # -------- offload hop/byte + comm log(round 0) + offload_pairs --------
        offload_hops_total = offload_bytes_total = offload_samples_total = 0
        offload_rows = []
        if self.do_offload:
            for p, si in plane_to_sats.items():
                m = len(si)
                if m <= 1:
                    continue
                Nb = np.array([offload_stats["Ni_before"][i] for i in si], dtype=int)
                Nt = np.array([offload_stats["Ni_target"][i] for i in si], dtype=int)
                for s_local, r_local, delta in build_transfer_plan_within_plane(Nb, Nt):
                    hop = ring_hop_distance(s_local, r_local, m)
                    hop = jitter_hop_count(self.link_rng, hop)  # 궤도 섭동 근사 ±10%
                    moved = int(abs(delta))
                    if moved <= 0:
                        continue
                    src, dst = int(si[s_local]), int(si[r_local])
                    offload_hops_total += int(hop)
                    ob = int(moved * self.sample_bytes * hop)
                    offload_bytes_total += ob
                    offload_samples_total += moved
                    offload_rows.append(
                        {
                            "plane_id": int(p),
                            "src_sat_id": src,
                            "dst_sat_id": dst,
                            "src_name": f"sat_{src + 1}",
                            "dst_name": f"sat_{dst + 1}",
                            "moved_samples": moved,
                            "hop": int(hop),
                            "sample_bytes": int(self.sample_bytes),
                            "offload_bytes": ob,
                        }
                    )
        self.comm_logger.log(
            0,
            model_bytes=0,
            offload_bytes=int(offload_bytes_total),
            model_hops=0,
            offload_hops=int(offload_hops_total),
            note=(
                f"one_shot_offload,samples={offload_samples_total},"
                f"sample_bytes={self.sample_bytes},hops={offload_hops_total}"
            ),
        )
        pd.DataFrame(offload_rows).to_csv(
            os.path.join(
                self.cfg.log_dir or ".", f"{self.log_prefix}_offload_pairs.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )

        # -------- classdist after + capacity(with Ni) --------
        after_path = os.path.join(
            self.cfg.log_dir or ".", f"{self.log_prefix}_classdist_after.csv"
        )
        save_class_distribution_csv(self.client_datasets, after_path)
        if offload_stats is not None:
            pd.DataFrame(
                [
                    {
                        "sat_id": i,
                        "plane_id": plane_ids[i],
                        "train_capacity": float(self.train_capacities[i]),
                        "Ni_before": float(offload_stats["Ni_before"][i]),
                        "Ni_target": float(offload_stats["Ni_target"][i]),
                        "Ni_after": float(offload_stats["Ni_after"][i]),
                    }
                    for i in range(n_clients)
                ]
            ).to_csv(cap_path, index=False, encoding="utf-8-sig")
            print(f"[SAVE][CAPACITY] {cap_path}")

        # 4) plane별 글로벌 모델
        plane_global_models = {
            p: copy.deepcopy(self.global_model).to(self.cfg.device)
            for p in plane_to_sats.keys()
        }
        self.acc_log = [(0, 0.0)]
        self.best_acc = -1.0

        # ==============================================================
        # per-cycle synchronous loop
        #   inter_plane_mode="mixing"  -> Design B opportunistic leader mixing (B3, LH-FL)
        #   inter_plane_mode="central" -> idealized hierarchical central average (B2)
        # ==============================================================
        contact_tl = {}
        relay_gain_tl = {}  # edge -> bool timeline: True where relay newly enables contact
        if self.inter_plane_mode == "mixing":
            horizon = self.cfg.num_rounds * self.contact_horizon_mult
            edges, direct_tl = self._leader_contact_timelines(
                sats, leaders_per_plane, leader_neighbors, horizon
            )
            print("[CONTACT-B] inter-plane leader edge contact fractions:")
            for e in sorted(edges):
                frac = 100.0 * float(direct_tl[e].mean())
                kind = (
                    "persistent"
                    if frac > 99.0
                    else ("intermittent" if frac > 0 else "NEVER")
                )
                print(f"  plane {e[0]}<->{e[1]}: {frac:5.1f}%  ({kind})")

            contact_tl = direct_tl
            if self.use_relay:
                _, relay_reach_tl = self._relay_contact_timelines(
                    sats, leaders_per_plane, leader_neighbors, plane_to_sats, horizon
                )
                combined_tl = {}
                print(
                    "[RELAY] inter-plane contact fractions with ordinary-satellite relay:"
                )
                for e in sorted(edges):
                    combined = direct_tl[e] | relay_reach_tl[e]
                    relay_gain_tl[e] = relay_reach_tl[e] & (~direct_tl[e])
                    combined_tl[e] = combined
                    print(
                        f"  plane {e[0]}<->{e[1]}: direct={100 * float(direct_tl[e].mean()):5.1f}%  "
                        f"+relay={100 * float(relay_gain_tl[e].mean()):5.1f}%  "
                        f"combined={100 * float(combined.mean()):5.1f}%"
                    )
                contact_tl = combined_tl
        else:
            print(
                "[HIER-CENTRAL] inter_plane_mode=central: idealized central average "
                "(no leader-contact modeling)"
            )

        spr = self.cfg.seconds_per_round
        sim_round = 0
        cycle = 0
        while sim_round < self.cfg.num_rounds and cycle < self.cfg.num_rounds:
            cycle += 1

            # (1) 위성별 학습시간 샘플 (offload 후 dataset size 고정)
            sampled = []
            for i in range(n_clients):
                if self.client_datasets[i] is None or len(self.client_datasets[i]) == 0:
                    sampled.append(0)
                    continue
                sr = sample_train_rounds_with_exp_jitter(
                    mean_minutes=self._current_mean_minutes(i),
                    seconds_per_round=spr,
                    jitter_sec=0.5,
                    clamp=True,
                    rng=self.rng,
                )
                sampled.append(sr)
                self.train_time_log.append((sim_round, i, float(sr * spr)))
            train_rounds = max([s for s in sampled if s > 0], default=0)

            # (2) 로컬 학습 (동기 1 epoch)
            for i in range(n_clients):
                if self.client_datasets[i] is None or len(self.client_datasets[i]) == 0:
                    continue
                train_one_epoch(
                    self.client_models[i],
                    self.client_datasets[i],
                    device=self.cfg.device,
                    batch_size=32,
                    lr=self.cfg.lr,
                    class_weight=self.client_datasets[i].class_weight,
                )

            # (3) intra-plane aggregation (즉시)
            # 링크 실패: plane 리더 자신은 항상 참여(자기 자신에게 보내는 게 아니므로 실패 대상이
            # 아님), 그 외 위성 -> 리더 업링크만 독립적으로 실패시킨다. 전원 실패 시(사실상 거의
            # 발생하지 않음) 정상 집계로 폴백해 plane이 통째로 멈추지 않게 한다.
            uplink_ok_per_plane = {}
            for p, si in plane_to_sats.items():
                leader = leaders_per_plane[p]
                others = [i for i in si if i != leader]
                ok = filter_by_link_failure(self.link_rng, self.link_fail_p, others)
                uplink_ok_per_plane[p] = ok
                si_eff = [leader] + ok
                if len(si_eff) <= 1 and len(si) > 1:
                    si_eff = si  # 안전장치: 전원 실패 폴백
                self.aggregate_fn(plane_global_models[p], self.client_models, si_eff)

            # (4) inter-plane aggregation. No wait in either mode.
            finish_round = sim_round + train_rounds
            mixing_round = finish_round
            old_plane_models = {
                p: copy.deepcopy(plane_global_models[p]).to(self.cfg.device)
                for p in plane_to_sats.keys()
            }

            if self.inter_plane_mode == "central":
                # B2: idealized hierarchical central. Average ALL plane models into
                # one (assumed always-reachable central server), broadcast to all planes.
                planes_sorted = sorted(plane_to_sats.keys())
                central_models = [old_plane_models[p] for p in planes_sorted]
                central_weights = [
                    sum(len(self.client_datasets[i]) for i in plane_to_sats[p])
                    for p in planes_sorted
                ]
                central_state = _weighted_average_state_dict_from_models(
                    central_models, weights=central_weights
                )
                for p in plane_to_sats.keys():
                    plane_global_models[p].load_state_dict(central_state)
                # comm: each plane leader up to PS + down from PS
                inter_hops = 2 * len(plane_to_sats)
                relay_hops_this_round = 0
                neighbor_count_report = None
            else:
                # B3 (LH-FL Design B): opportunistic per-round leader mixing (Eq.9 N_p^(r))
                mixed_states = {}
                inter_hops = 0
                relay_hops_this_round = 0
                neighbor_count_report = {}
                for p in plane_to_sats.keys():
                    avail = self._available_neighbors(
                        p, leader_neighbors, contact_tl, finish_round
                    )
                    # 궤도상 접촉 가능(결정적) 여부와, 그 접촉에서 실제 수신 성공(랜덤) 여부를 분리
                    avail = filter_by_link_failure(
                        self.link_rng, self.link_fail_p, avail
                    )
                    inter_hops += len(
                        avail
                    )  # directed transmissions actually used (성공한 것만)
                    neighbor_count_report[p] = len(avail)
                    if self.use_relay:
                        for q in avail:
                            e = (min(p, q), max(p, q))
                            tl = relay_gain_tl.get(e)
                            if (
                                tl is not None
                                and finish_round < len(tl)
                                and bool(tl[finish_round])
                            ):
                                relay_hops_this_round += 1
                    neighbor_models = [old_plane_models[q] for q in avail]
                    mixed_states[p] = _mix_self_with_neighbor_average(
                        self_model=old_plane_models[p],
                        neighbor_models=neighbor_models,
                        beta=self.leader_mixing_beta,
                        neighbor_weights=None,
                    )
                for p in plane_to_sats.keys():
                    plane_global_models[p].load_state_dict(mixed_states[p])
                if self.use_relay:
                    print(
                        f"[RELAY][round={finish_round}] neighbor counts per plane: "
                        f"{neighbor_count_report}, relay_assisted_edges_this_round="
                        f"{relay_hops_this_round}"
                    )

            # (5) plane 내부 broadcast (업링크와는 독립적으로 다운링크 실패를 새로 굴림)
            downlink_ok_per_plane = {}
            for p, si in plane_to_sats.items():
                leader = leaders_per_plane[p]
                others = [i for i in si if i != leader]
                ok = filter_by_link_failure(self.link_rng, self.link_fail_p, others)
                downlink_ok_per_plane[p] = ok
                for i in ok:
                    self.client_models[i].load_state_dict(
                        copy.deepcopy(plane_global_models[p].state_dict())
                    )
                # ok에 없는(실패한) 위성은 load_state_dict을 건너뛰어 이전 라운드 로컬 모델을 유지

            # (6) representative global model
            rep_models = [plane_global_models[p] for p in sorted(plane_to_sats.keys())]
            rep_weights = [
                sum(len(self.client_datasets[i]) for i in plane_to_sats[p])
                for p in sorted(plane_to_sats.keys())
            ]
            self.global_model.load_state_dict(
                _weighted_average_state_dict_from_models(
                    rep_models, weights=rep_weights
                )
            )

            # (7) comm log (intra + inter), mixing_round.
            # link failure가 켜져 있으면 "성공한" 업링크+다운링크만 카운트되어 seed마다 변동이 생김.
            plane_hops = sum(
                len(uplink_ok_per_plane[p]) + len(downlink_ok_per_plane[p])
                for p in plane_to_sats.keys()
            )
            model_hops = plane_hops + inter_hops
            relay_note = (
                f",use_relay={self.use_relay},relay_hops={relay_hops_this_round}"
                if self.use_relay
                else ""
            )
            self.comm_logger.log(
                mixing_round,
                model_bytes=model_hops * self.model_bytes,
                model_hops=model_hops,
                offload_bytes=0,
                offload_hops=0,
                note=f"inter_plane_mode={self.inter_plane_mode},"
                f"realcontacts={self.use_real_contacts},inter_hops={inter_hops}"
                f"{relay_note}",
            )

            # (8) accuracy
            acc = _evaluate_weighted_model_accuracy(
                models=rep_models,
                weights=rep_weights,
                test_dataset=self.test_dataset,
                device=self.cfg.device,
            )
            self.acc_log.append((mixing_round, acc))
            self.best_acc = max(self.best_acc, acc)
            print(
                f"[ACC][lhfl-{self.inter_plane_mode}] cycle={cycle} round={sim_round}->{mixing_round} "
                f"(train={train_rounds}, inter_hops={inter_hops}) "
                f"acc={acc:.2f}% best={self.best_acc:.2f}%"
            )
            if acc >= self.cfg.target_acc:
                print(
                    f"[INFO] Target {self.cfg.target_acc}% reached at round {mixing_round}. (continue)"
                )
            sim_round = mixing_round

        print("[END][offload-contact] finished.")
        self.finalize_logs(prefix=self.log_prefix)
        return self.global_model, self.best_acc


def run_contact_aware_lhfl(
    client_datasets,
    test_dataset,
    per_sat_minutes,
    num_rounds,
    constellation_configs,
    device,
    log_dir,
    log_prefix,
    use_real_contacts: bool = True,
    max_dist_km: float = 5000.0,
    target_acc: float = 85.0,
    seconds_per_round: int = 60,
    hidden_size: int = 100,
    lr: float = 1e-3,
    agg_name: str = "fedavg",
    offload_ratio: float = 0.1,
    leader_top_k: int = 2,
    leader_mixing_beta: float = 0.5,
    inter_plane_mode: str = "mixing",
    do_offload: bool = True,
    use_relay: bool = False,
    relay_max_age_rounds: Optional[int] = None,
    link_fail_seed: Optional[int] = None,
    link_fail_p: float = LINK_FAIL_P_DEFAULT,
):
    """Convenience runner (run_simulation hardcodes max_dist_km=2000; this lets us set 5000).

    inter_plane_mode="mixing"  -> LH-FL Design B (ladder rung B3)
    inter_plane_mode="central" -> idealized hierarchical central average (ladder rung B2)
    do_offload=False           -> skip capacity-aware one-shot offload (ablation-ladder
                                   rung "Decentral+Leader": leader hierarchy without offload).
                                   Default True reproduces the original LH-FL behaviour.
    use_relay=True             -> also count leader<->ordinary-satellite-of-other-plane
                                   contact as an available mixing edge (see
                                   ContactAwareOffloadScenario docstring). Default
                                   False reproduces the original behaviour exactly.
    """
    cfg = SimConfig(
        num_rounds=num_rounds,
        device=device,
        hidden_size=hidden_size,
        lr=lr,
        seconds_per_round=seconds_per_round,
        log_dir=log_dir,
        target_acc=target_acc,
        aggregate_name=agg_name,
        constellation_configs=constellation_configs,
        log_prefix=log_prefix,
        model_name="dnn",
    )
    if len(constellation_configs) != len(client_datasets):
        raise ValueError("constellation_configs length must match client_datasets.")
    if len(per_sat_minutes) != len(client_datasets):
        raise ValueError("per_sat_minutes length must match client_datasets.")
    link_fail_rng = (
        np.random.default_rng(link_fail_seed) if link_fail_seed is not None else None
    )
    scenario = ContactAwareOffloadScenario(
        client_datasets=client_datasets,
        test_dataset=test_dataset,
        config=cfg,
        max_dist_km=max_dist_km,
        offload_ratio=offload_ratio,
        top_pairs_per_sat=1,
        log_prefix=log_prefix,
        do_offload=do_offload,
        per_sat_minutes=per_sat_minutes,
        leader_top_k=leader_top_k,
        leader_mixing_beta=leader_mixing_beta,
        use_real_contacts=use_real_contacts,
        inter_plane_mode=inter_plane_mode,
        use_relay=use_relay,
        relay_max_age_rounds=relay_max_age_rounds,
        link_fail_rng=link_fail_rng,
        link_fail_p=link_fail_p,
    )
    return scenario.run()
