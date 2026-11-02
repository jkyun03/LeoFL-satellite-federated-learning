# scenarios_fedisl_mixing.py
#
# Ladder rungs 2 and 4: "FedISL + hierarchical decentralized aggregation".
#
#   rung 2 = do_offload=False  (FedISL structure + sink decentralized mixing, no offload)
#   rung 4 = do_offload=True   (rung 2 + capacity-aware one-shot offloading)
#
# Idea (per advisor):
#   Take FedISL EXACTLY as-is -- source selection, intra-plane ISL ring
#   distribution, local training, sink selection, AND the sink->PS ground
#   contact wait -- and replace ONLY the final "sink -> PS central averaging"
#   step with decentralized sink-to-sink model mixing. The sink of each plane
#   plays the role of the plane leader.
#
# Key point: the ground-contact wait is KEPT in every rung here (only LH-FL
# removes it). So rungs 2 and 4 are expected to stay ground-bottlenecked
# (~as slow as FedISL / FedISL+offload); LH-FL's speedup comes from removing
# the ground dependency, which this ladder isolates at the (4 -> LH-FL) step.
#
# The sink-to-sink mixing uses the same opportunistic per-round contact rule as
# LH-FL Design B (mix only with sinks within max_dist_km at that round) and the
# same uniform-average mixing (Eq. 9). There is no central server, so each plane
# maintains its own model across rounds (decentralized); accuracy is the
# sample-weighted average over participating plane models. Offloading (rung 4)
# uses the same capacity-aware one-shot scheme as the other offloading rungs.

import os
import copy
import numpy as np

from LeoFL.Simulator.fl_core import (
    connected_clients_at_round,
    sample_train_rounds_with_exp_jitter,
    offload_balance_datasets,
    save_class_distribution_csv,
    build_transfer_plan_within_plane,
    SimConfig,
)
from LeoFL.Simulator.scenarios import (
    FedISLScenario,
    ring_hop_distance,
    _ring_distance_in_plane,
    _choose_fedisl_sink,
    _mix_self_with_neighbor_average,
    _weighted_average_state_dict_from_models,
    _evaluate_weighted_model_accuracy,
)


class FedISLLeaderMixingScenario(FedISLScenario):
    """FedISL structure + sink-to-sink decentralized mixing, optional offloading."""

    def __init__(
        self,
        client_datasets,
        test_dataset,
        config: SimConfig,
        log_prefix: str = "sat_events_fedisl_mix",
        per_sat_minutes=None,
        train_minutes: int = 10,
        max_dist_km: float = 5000.0,
        leader_mixing_beta: float = 0.5,
        do_offload: bool = False,
        offload_ratio: float = 0.1,
    ):
        super().__init__(
            client_datasets,
            test_dataset,
            config,
            log_prefix,
            per_sat_minutes,
            train_minutes,
        )
        self.max_dist_km = max_dist_km
        self.leader_mixing_beta = leader_mixing_beta
        self.do_offload = do_offload
        self.offload_ratio = offload_ratio

        # capacity + n_i/C_i time model (only needed when offloading; kept
        # consistent with the other offloading rungs so timing is comparable)
        if per_sat_minutes is not None:
            mins = np.asarray(per_sat_minutes, dtype=float)
        else:
            mins = np.full(self.n_clients, float(train_minutes))
        cap = 1.0 / np.clip(mins, 1e-6, None)
        cap = cap / cap.mean()
        self.train_capacities = cap
        Ni0 = np.array([len(ds) for ds in client_datasets], dtype=float)
        ratio0 = Ni0 / np.clip(cap, 1e-6, None)
        self.time_scale_const = float(np.mean(mins / np.clip(ratio0, 1e-6, None)))

    def _nc_mean_minutes(self, i: int) -> float:
        Ni = float(len(self.client_datasets[i]))
        Si = float(self.train_capacities[i])
        return max(self.time_scale_const * (Ni / max(Si, 1e-6)), 0.5)

    def _sinks_in_contact(self, sink_a: int, sink_b: int, round_idx: int) -> bool:
        xa, ya, za = self.sats[sink_a].get_leo_orbit_position(round_idx)
        xb, yb, zb = self.sats[sink_b].get_leo_orbit_position(round_idx)
        d_km = np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2) / 1000.0
        return d_km <= self.max_dist_km

    def _do_one_shot_offload(self):
        """Capacity-aware one-shot offload (same scheme as other offloading rungs).
        Logs offload comm at round 0, then rebases base_train_minutes to n_i/C_i."""
        save_class_distribution_csv(
            self.client_datasets,
            os.path.join(
                self.cfg.log_dir or ".", f"{self.log_prefix}_classdist_before.csv"
            ),
        )

        dummy_mc = np.zeros(
            (self.n_clients, self.n_clients), dtype=int
        )  # capacity mode ignores it
        offload_stats = offload_balance_datasets(
            client_datasets=self.client_datasets,
            meeting_counts=dummy_mc,
            plane_ids=self.plane_ids,
            offload_ratio=self.offload_ratio,
            top_pairs_per_sat=1,
            seed=0,
            train_capacities=self.train_capacities,
        )

        offload_hops_total = offload_bytes_total = offload_samples_total = 0
        for p, sat_indices in self.plane_to_sats.items():
            m = len(sat_indices)
            if m <= 1:
                continue
            Nb = np.array(
                [offload_stats["Ni_before"][i] for i in sat_indices], dtype=int
            )
            Nt = np.array(
                [offload_stats["Ni_target"][i] for i in sat_indices], dtype=int
            )
            for s_local, r_local, delta in build_transfer_plan_within_plane(Nb, Nt):
                hop = ring_hop_distance(s_local, r_local, m)
                moved = int(abs(delta))
                if moved <= 0:
                    continue
                offload_hops_total += int(hop)
                offload_bytes_total += int(moved * self.sample_bytes * hop)
                offload_samples_total += moved
        self.comm_logger.log(
            0,
            model_bytes=0,
            offload_bytes=int(offload_bytes_total),
            model_hops=0,
            offload_hops=int(offload_hops_total),
            note=(
                f"fedisl_mix_one_shot_offload,samples={offload_samples_total},"
                f"hops={offload_hops_total}"
            ),
        )

        save_class_distribution_csv(
            self.client_datasets,
            os.path.join(
                self.cfg.log_dir or ".", f"{self.log_prefix}_classdist_after.csv"
            ),
        )
        print("[FedISL-MIX] offload done.")

    def run(self):
        print(
            f"[INFO][FedISL-MIX] start (FedISL timing + sink decentralized mixing, "
            f"do_offload={self.do_offload})"
        )

        if self.do_offload:
            self._do_one_shot_offload()

        # Unified n_i/C_i time model across ALL ladder rungs (so offloading's
        # timing effect is isolated). Applied whether or not offloading ran;
        # if offloaded, it reflects the post-offload dataset sizes.
        self.base_train_minutes = np.array(
            [self._nc_mean_minutes(i) for i in range(self.n_clients)], dtype=float
        )
        print(
            "[FedISL-MIX] base_train_minutes set to n_i/C_i "
            f"(offload={self.do_offload})."
        )

        # decentralized: each plane keeps its own model across rounds
        plane_models = {
            p: copy.deepcopy(self.global_model).to(self.cfg.device)
            for p in self.plane_to_sats.keys()
        }
        self.acc_log = [(0, 0.0)]
        self.best_acc = -1.0

        current_round = 0
        while current_round < self.cfg.num_rounds:
            plane_finish_rounds = []
            participating = []  # dicts: plane, sink, model, weight
            plane_hops = 0

            for p, sat_indices in self.plane_to_sats.items():
                sat_indices = list(sat_indices)
                m = len(sat_indices)

                # --- FedISL source selection (ground contact kept) ---
                source, source_round = self._find_source_for_plane(
                    sat_indices, start_round=current_round
                )
                if source is None:
                    print(f"[WARN][FedISL-MIX] no source for plane={p}; skip.")
                    continue

                # distribute THIS plane's model through the plane (source-rooted ring)
                for sid in sat_indices:
                    self.client_models[sid].load_state_dict(
                        copy.deepcopy(plane_models[p].state_dict())
                    )

                dist_hops = max(
                    _ring_distance_in_plane(sid, source, sat_indices)
                    for sid in sat_indices
                )
                distribution_done_round = source_round + dist_hops
                plane_hops += max(m - 1, 0)  # ring distribution edges

                # --- sample training times (FedISL model; n_i/C_i if offloaded) ---
                max_train_rounds = 0
                for sid in sat_indices:
                    sampled = sample_train_rounds_with_exp_jitter(
                        mean_minutes=float(self.base_train_minutes[sid]),
                        seconds_per_round=self.cfg.seconds_per_round,
                        jitter_sec=0.5,
                        clamp=True,
                        rng=self.rng,
                    )
                    max_train_rounds = max(max_train_rounds, sampled)
                    self.train_time_log.append(
                        (
                            distribution_done_round,
                            sid,
                            float(sampled * self.cfg.seconds_per_round),
                        )
                    )

                pre_sink_finish = distribution_done_round + max_train_rounds + dist_hops
                sink = _choose_fedisl_sink(
                    self.sats,
                    sat_indices=sat_indices,
                    future_round=pre_sink_finish,
                    max_rounds=self.cfg.num_rounds,
                )

                # local training
                for sid in sat_indices:
                    self._train_satellite_once(sid)

                # intra-plane aggregation into this plane's model (sink holds it)
                tmp_plane_model = copy.deepcopy(plane_models[p]).to(self.cfg.device)
                self.aggregate_fn(tmp_plane_model, self.client_models, sat_indices)
                plane_hops += max(m - 1, 0)  # aggregation edges

                # --- sink -> PS ground-contact WAIT (kept identical to FedISL) ---
                sink_upload_round = None
                for t in range(pre_sink_finish, self.cfg.num_rounds):
                    if sink in set(connected_clients_at_round(self.sats, t)):
                        sink_upload_round = t
                        break
                if sink_upload_round is None:
                    print(
                        f"[WARN][FedISL-MIX] sink cannot reach PS for plane={p}; skip."
                    )
                    continue

                participating.append(
                    {
                        "plane": p,
                        "sink": sink,
                        "model": tmp_plane_model,
                        "weight": sum(
                            len(self.client_datasets[sid]) for sid in sat_indices
                        ),
                    }
                )
                plane_finish_rounds.append(sink_upload_round)
                print(
                    f"[FedISL-MIX][plane={p}] source=sat_{source + 1}, sink=sat_{sink + 1}, "
                    f"finish={sink_upload_round}"
                )

            if len(participating) == 0:
                break

            # aggregation happens when the slowest sink is ready (same as FedISL)
            aggregate_round = max(plane_finish_rounds)

            # -------- sink-to-sink decentralized mixing (Design B, 5000km) --------
            planes_here = [d["plane"] for d in participating]
            sink_of = {d["plane"]: d["sink"] for d in participating}
            weight_of = {d["plane"]: d["weight"] for d in participating}
            old_models = {
                d["plane"]: copy.deepcopy(d["model"]).to(self.cfg.device)
                for d in participating
            }

            inter_hops = 0
            new_states = {}
            for p in planes_here:
                neigh = []
                for q in planes_here:
                    if p == q:
                        continue
                    if self._sinks_in_contact(sink_of[p], sink_of[q], aggregate_round):
                        neigh.append(q)
                inter_hops += len(neigh)
                neighbor_models = [old_models[q] for q in neigh]
                new_states[p] = _mix_self_with_neighbor_average(
                    self_model=old_models[p],
                    neighbor_models=neighbor_models,
                    beta=self.leader_mixing_beta,
                    neighbor_weights=None,
                )

            for p in planes_here:
                plane_models[p].load_state_dict(new_states[p])

            # -------- comm log (intra ring + realized sink mixing; no PS) --------
            model_hops = plane_hops + inter_hops
            self.comm_logger.log(
                aggregate_round,
                model_bytes=model_hops * self.model_bytes,
                model_hops=model_hops,
                offload_bytes=0,
                offload_hops=0,
                note=f"fedisl_leader_mixing,do_offload={self.do_offload},inter_hops={inter_hops}",
            )

            # -------- accuracy: weighted avg over participating plane models --------
            rep_models = [plane_models[p] for p in planes_here]
            rep_weights = [weight_of[p] for p in planes_here]
            self.global_model.load_state_dict(
                _weighted_average_state_dict_from_models(
                    rep_models, weights=rep_weights
                )
            )
            acc = _evaluate_weighted_model_accuracy(
                models=rep_models,
                weights=rep_weights,
                test_dataset=self.test_dataset,
                device=self.cfg.device,
            )
            self.acc_log.append((aggregate_round, acc))
            self.best_acc = max(self.best_acc, acc)
            print(
                f"[ACC][FedISL-MIX] round={aggregate_round}, acc={acc:.2f}%, "
                f"best={self.best_acc:.2f}%, inter_hops={inter_hops}"
            )

            if acc >= self.cfg.target_acc:
                print(
                    f"[INFO][FedISL-MIX] Target {self.cfg.target_acc}% reached "
                    f"at round {aggregate_round}. (continue)"
                )

            current_round = aggregate_round + 1

        self.finalize_logs(prefix=self.log_prefix)
        return self.global_model, self.best_acc


def run_fedisl_leader_mixing(
    client_datasets,
    test_dataset,
    per_sat_minutes,
    num_rounds,
    constellation_configs,
    device,
    log_dir,
    log_prefix,
    do_offload: bool = False,
    offload_ratio: float = 0.1,
    max_dist_km: float = 5000.0,
    leader_mixing_beta: float = 0.5,
    target_acc: float = 85.0,
    seconds_per_round: int = 60,
    hidden_size: int = 100,
    lr: float = 1e-3,
    agg_name: str = "fedavg",
):
    """Runner for ladder rung 2 (do_offload=False) and rung 4 (do_offload=True)."""
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
    scenario = FedISLLeaderMixingScenario(
        client_datasets=client_datasets,
        test_dataset=test_dataset,
        config=cfg,
        log_prefix=log_prefix,
        per_sat_minutes=per_sat_minutes,
        max_dist_km=max_dist_km,
        leader_mixing_beta=leader_mixing_beta,
        do_offload=do_offload,
        offload_ratio=offload_ratio,
    )
    return scenario.run()
