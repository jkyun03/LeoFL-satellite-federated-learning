# LeoFL/followup/engine.py
"""
Synchronous, whole-constellation-barrier engine reproducing the existing
LH-FL's cycle structure (``LeoFL.Simulator.scenarios.OffloadScenario``), with
local-step-count throttling in place of pre-training data offloading.

One cycle (matches ``OffloadScenario.run()`` exactly, on a continuous clock
instead of a tick countdown):

  0. (Method "offload" only) Once, before cycle 1: redistribute each plane's
     samples proportional to capacity share via the existing, unmodified
     ``fl_core.offload_balance_datasets``/``offload_within_plane``
     (``n_i' = (C_i/ΣC_j) × Σn_j``), then run the exact same fixed-step
     policy as Method "fixed" on the rebalanced data. This is the
     fair-comparison reference for "move data" vs. "throttle steps": every
     other mechanism below (optimizer, time model, aggregation weighting,
     barrier) is identical across all four methods, so only the data
     redistribution differs. (``legacy_reference.py`` additionally differs
     in optimizer (Adam) and aggregation weighting (plain, unweighted mean)
     from every method here, so it is a *supplementary*, not fair-controlled,
     reference -- see README.)
  1. Decide ``step_i`` for every satellite, per plane (policy: "fixed" /
     "capacity" / "capacity_minwork" / "offload" -- see ``policies.py``).
  2. Train every satellite for ``step_i`` real vanilla-SGD steps
     (``train_n_steps``; vanilla SGD, momentum=0, because FedNova's
     normalization here is only valid for that case -- the legacy
     ``fl_core.train_one_epoch`` uses Adam, so it is not reused).
  3. ``cycle_duration`` = max over **every satellite in every plane** of
     (train time + intra-orbit relay to its leader), plus one shared
     intra-orbit aggregation cost, plus one shared inter-orbit transfer cost.
     This whole-constellation barrier is carried over unchanged from
     ``OffloadScenario`` (``len(completed_this_cycle) == n_clients``), so
     Method "fixed" reproduces the exact straggler problem the existing
     offloading mechanism exists to solve -- making "move data" vs.
     "throttle steps" an apples-to-apples comparison.
  4. Intra-orbit aggregate every plane (FedNova; reduces to weighted-avg
     when every member did the same number of steps, i.e. Method "fixed").
  5. Snapshot the post-aggregation plane states, then inter-orbit convex-mix
     every plane with its static top-k neighbors
     (``fednova.convex_mix_states``), unconditionally, every cycle -- same
     as ``OffloadScenario``'s ``old_plane_models`` -> mix pattern. There is
     no contact-timing gate: this is a deliberate simplification per the
     redesigned research direction (see README), not an oversight.
  6. Advance ``sim_time`` by the cycle's total duration; checkpoint accuracy.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from LeoFL.Simulator.fl_core import build_model, offload_balance_datasets
from LeoFL.Simulator.comm_Logger import calc_model_bytes, calc_sample_bytes

from LeoFL.followup import fednova
from LeoFL.followup.config import ExperimentConfig
from LeoFL.followup.data_iterators import PersistentShuffledIterator, fetch_batch
from LeoFL.followup.logging_schema import RunLogs
from LeoFL.followup.metrics import (
    overrepresented_classes,
    slow_quartile_satellites,
    subgroup_f1,
    subgroup_recall,
    weighted_representative_eval,
)
from LeoFL.followup.policies import (
    AGGREGATION_FEDNOVA,
    METHOD_CAPACITY,
    METHOD_CAPACITY_MINWORK,
    METHOD_FIXED,
    METHOD_OFFLOAD,
    apply_minimum_work_floor,
    capacity_proportional_steps,
    fixed_steps,
)
from LeoFL.followup.time_model import (
    SatelliteTimeProfile,
    build_time_profiles,
    draw_epoch_minutes,
)
from LeoFL.followup.topology import Topology


def train_n_steps(
    model: nn.Module,
    dataset,
    iterator: PersistentShuffledIterator,
    n_steps: int,
    batch_size: int,
    lr: float,
    device,
) -> None:
    """Vanilla SGD, momentum=0 (required for the FedNova scope in
    fednova.py)."""
    if n_steps <= 0:
        return
    model.to(device)
    model.train()
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    criterion = nn.CrossEntropyLoss()
    for _ in range(n_steps):
        idx = iterator.next_indices(batch_size)
        if len(idx) == 0:
            break
        X, Y = fetch_batch(dataset, idx)
        X, Y = X.to(device), Y.to(device)
        optim.zero_grad()
        loss = criterion(model(X), Y)
        loss.backward()
        optim.step()


def transfer_time_s(model_bytes: int, cfg: ExperimentConfig) -> float:
    return model_bytes / cfg.isl_bandwidth_bps + cfg.per_hop_latency_s


def relay_time_s(hops: int, model_bytes: int, cfg: ExperimentConfig) -> float:
    if hops <= 0:
        return 0.0
    return hops * (model_bytes / cfg.isl_bandwidth_bps + cfg.per_hop_latency_s)


def agg_time_s(n_members: int, cfg: ExperimentConfig) -> float:
    return n_members * cfg.leader_agg_time_per_member_s


def decide_steps(
    cfg: ExperimentConfig,
    members: List[int],
    time_profiles: Dict[int, SatelliteTimeProfile],
) -> Dict[int, int]:
    if cfg.method in (METHOD_FIXED, METHOD_OFFLOAD):
        # "offload" rebalances the *data* once before cycle 1 (see
        # run_followup_experiment) and otherwise trains exactly like
        # "fixed" -- same fixed step count for everyone, every cycle.
        return fixed_steps(members, cfg.fixed_steps, cfg.max_local_steps)
    elif cfg.method == METHOD_CAPACITY:
        return capacity_proportional_steps(
            members, time_profiles, cfg.fixed_steps, cfg.max_local_steps
        )
    elif cfg.method == METHOD_CAPACITY_MINWORK:
        base = capacity_proportional_steps(
            members, time_profiles, cfg.fixed_steps, cfg.max_local_steps
        )
        return apply_minimum_work_floor(
            base,
            members,
            time_profiles,
            cfg.minimum_epoch_fraction,
            cfg.max_local_steps,
        )
    else:
        raise ValueError(f"Unknown method: {cfg.method!r}")


def run_followup_experiment(
    cfg: ExperimentConfig,
    topology: Topology,
    client_datasets: list,
    test_dataset,
    device,
) -> Tuple[RunLogs, List[Tuple[float, float]]]:
    # Seeds every torch random draw from here on (crucially, the initial
    # model's weight init); combined with the explicitly-seeded numpy RNGs
    # used elsewhere (data split, compute-heterogeneity draw, per-satellite
    # shuffled iterators), the same cfg.seed reproduces an identical run.
    torch.manual_seed(cfg.seed)

    n_sats = len(client_datasets)
    n_samples = [int(ds.X.shape[0]) for ds in client_datasets]
    num_classes = test_dataset.n_classes
    input_size = getattr(test_dataset, "n_features", None)

    run_logs = RunLogs(method=cfg.method, seed=cfg.seed, log_dir=cfg.log_dir)

    if cfg.method == METHOD_OFFLOAD:
        # Capacity is a pure function of (n_sats, seed, low, high) -- independent
        # of n_samples -- so it can be computed before build_time_profiles, using
        # the exact same capacity_i = 1/epoch_minutes_i definition OffloadScenario
        # itself uses (normalized to mean 1, matching its own convention).
        epoch_minutes_arr = draw_epoch_minutes(
            n_sats, cfg.seed, cfg.epoch_minutes_low, cfg.epoch_minutes_high
        )
        train_capacities = 1.0 / epoch_minutes_arr
        train_capacities = train_capacities / train_capacities.mean()

        offload_stats = offload_balance_datasets(
            client_datasets=client_datasets,
            meeting_counts=topology.meeting_counts,
            plane_ids=topology.plane_ids,
            offload_ratio=cfg.offload_ratio,
            top_pairs_per_sat=cfg.offload_top_pairs_per_sat,
            seed=cfg.seed,
            train_capacities=train_capacities,
        )
        n_samples = [
            int(ds.X.shape[0]) for ds in client_datasets
        ]  # offload mutated client_datasets in place

        sample_bytes = (
            calc_sample_bytes(n_features=input_size)
            if input_size is not None
            else calc_sample_bytes(sample_shape=client_datasets[0].X[0].shape)
        )
        offload_bytes_total = int(offload_stats["offload_hops"] * sample_bytes)
        run_logs.comm_log.add(
            sim_time_s=0.0,
            kind="data_offload",
            plane_or_edge="all",
            bytes=offload_bytes_total,
            hops=int(offload_stats["offload_hops"]),
        )

    # Built *after* the optional offload step above, so steps_per_epoch/step_time
    # reflect the (possibly rebalanced) n_samples; epoch_minutes_i is unchanged
    # (same seed -> same draw_epoch_minutes result) -- a satellite's intrinsic
    # speed doesn't change, only how much data it now has to get through.
    time_profiles_list = build_time_profiles(
        n_samples,
        cfg.batch_size,
        seed=cfg.seed,
        low=cfg.epoch_minutes_low,
        high=cfg.epoch_minutes_high,
    )
    time_profiles = {p.sat_id: p for p in time_profiles_list}
    iterators = {
        i: PersistentShuffledIterator(n_samples[i], seed=cfg.seed * 100_003 + i)
        for i in range(n_sats)
    }

    initial_model = build_model(
        model_name="dnn",
        num_classes=num_classes,
        input_size=input_size,
        hidden_size=100,
    ).to(device)
    model_bytes = calc_model_bytes(initial_model)
    scratch_model = build_model(
        model_name="dnn",
        num_classes=num_classes,
        input_size=input_size,
        hidden_size=100,
    ).to(device)

    leader_ids = set(topology.leaders_per_plane.values())
    for p in time_profiles_list:
        run_logs.sat_profile_log.add(
            sat_id=p.sat_id,
            plane_id=topology.plane_ids[p.sat_id],
            is_leader=(p.sat_id in leader_ids),
            n_samples=p.n_samples,
            epoch_minutes=p.epoch_minutes,
            steps_per_epoch=p.steps_per_epoch,
            step_time_s=p.step_time_s,
        )

    initial_state = {
        k: v.detach().clone() for k, v in initial_model.state_dict().items()
    }
    plane_state: Dict[int, Dict[str, torch.Tensor]] = {
        p: {k: v.clone() for k, v in initial_state.items()}
        for p in topology.plane_to_sats
    }
    model_version: Dict[int, int] = {p: 0 for p in topology.plane_to_sats}

    epoch_minutes_per_sat = {p.sat_id: p.epoch_minutes for p in time_profiles_list}
    slow_ids = slow_quartile_satellites(epoch_minutes_per_sat, frac=0.25)
    over_classes = overrepresented_classes(
        client_datasets, slow_ids, list(range(n_sats)), n_classes=num_classes, top_n=3
    )

    acc_log: List[Tuple[float, float]] = []
    checkpoint_interval = cfg.effective_accuracy_checkpoint_interval_s()
    state = {"next_checkpoint": 0.0}

    def do_checkpoint(t: float) -> None:
        if t < state["next_checkpoint"]:
            return
        orbit_models, orbit_weights = {}, {}
        for p, members in topology.plane_to_sats.items():
            m = build_model(
                model_name="dnn",
                num_classes=num_classes,
                input_size=input_size,
                hidden_size=100,
            ).to(device)
            m.load_state_dict(plane_state[p])
            orbit_models[p] = m
            orbit_weights[p] = float(sum(n_samples[i] for i in members))
        weighted, _per_orbit, spread = weighted_representative_eval(
            orbit_models, orbit_weights, test_dataset, device
        )
        run_logs.accuracy_log.add(
            sim_time_s=t,
            weighted_accuracy=weighted.accuracy,
            weighted_loss=weighted.loss,
            weighted_macro_f1=weighted.macro_f1,
            orbit_acc_min=spread["min"],
            orbit_acc_mean=spread["mean"],
            orbit_acc_max=spread["max"],
            slow_subgroup_recall=subgroup_recall(weighted, over_classes),
            slow_subgroup_macro_f1=subgroup_f1(weighted, over_classes),
        )
        acc_log.append((t, weighted.accuracy))
        while state["next_checkpoint"] <= t:
            state["next_checkpoint"] += checkpoint_interval

    do_checkpoint(0.0)

    horizon = cfg.total_simulated_time_s
    sim_time = 0.0
    cycle_id = 0

    while sim_time < horizon:
        cycle_id += 1
        finish_times: List[float] = []
        plane_end_states: Dict[int, List[dict]] = {}
        plane_weights: Dict[int, List[float]] = {}
        plane_steps: Dict[int, List[int]] = {}

        # 1-2: decide steps + train, per plane; accumulate the global
        # (whole-constellation) list of per-satellite finish times.
        for p, members in topology.plane_to_sats.items():
            steps_per_member = decide_steps(cfg, members, time_profiles)
            start_state = plane_state[p]
            end_states, weights, steps_list = [], [], []

            for m in members:
                steps_i = steps_per_member[m]
                scratch_model.load_state_dict(start_state)
                train_n_steps(
                    scratch_model,
                    client_datasets[m],
                    iterators[m],
                    steps_i,
                    cfg.batch_size,
                    cfg.lr,
                    device,
                )
                end_state_m = {
                    k: v.detach().clone() for k, v in scratch_model.state_dict().items()
                }

                train_finish = sim_time + steps_i * time_profiles[m].step_time_s
                hops = topology.ring_hops_to_leader(p, m)
                arrival = train_finish + relay_time_s(hops, model_bytes, cfg)
                finish_times.append(arrival)

                run_logs.step_log.add(
                    cycle_id=cycle_id,
                    plane_id=p,
                    sat_id=m,
                    is_leader=(m == topology.leaders_per_plane[p]),
                    model_version=model_version[p],
                    t_start_s=sim_time,
                    t_end_s=arrival,
                    steps=steps_i,
                    step_time_s=time_profiles[m].step_time_s,
                    samples_processed_cumulative=iterators[m].total_samples_processed,
                    unique_samples_seen_cumulative=iterators[m].unique_samples_seen,
                )

                end_states.append(end_state_m)
                weights.append(float(client_datasets[m].X.shape[0]))
                steps_list.append(steps_i)

            plane_end_states[p] = end_states
            plane_weights[p] = weights
            plane_steps[p] = steps_list

        # 3: whole-constellation barrier -- wait for the single slowest
        # satellite across every plane, then one shared aggregation cost.
        max_plane_size = max(len(m) for m in topology.plane_to_sats.values())
        agg_done = max(finish_times) + agg_time_s(max_plane_size, cfg)

        # 4: intra-orbit aggregation, every plane.
        new_plane_state: Dict[int, Dict[str, torch.Tensor]] = {}
        for p, members in topology.plane_to_sats.items():
            start_state, end_states = plane_state[p], plane_end_states[p]
            weights, steps_list = plane_weights[p], plane_steps[p]

            if cfg.aggregation == AGGREGATION_FEDNOVA:
                new_state = fednova.fednova_aggregate(
                    start_state, end_states, weights, steps_list
                )
            else:
                new_state = fednova.weighted_avg_aggregate(end_states, weights)
            new_plane_state[p] = new_state
            model_version[p] += 1

            w_arr = np.asarray(weights, dtype=float)
            w_arr = w_arr / w_arr.sum()
            tau_eff_val = float(np.sum(w_arr * np.asarray(steps_list, dtype=float)))
            run_logs.aggregation_log.add(
                plane_id=p,
                cycle_id=cycle_id,
                model_version=model_version[p],
                agg_time_s=agg_done,
                aggregation_type=cfg.aggregation,
                n_members=len(members),
                tau_eff=tau_eff_val,
                min_steps=min(steps_list),
                max_steps=max(steps_list),
            )

        # 5: inter-orbit mixing -- unconditional every cycle, using the
        # pre-mix snapshot (matches OffloadScenario's old_plane_models ->
        # mixed_plane_states pattern). No contact-timing gate (see module
        # docstring).
        mix_time = agg_done + transfer_time_s(model_bytes, cfg)
        mixed_state: Dict[int, Dict[str, torch.Tensor]] = {}
        logged_edges = set()

        for p in topology.plane_to_sats:
            neighbors = sorted(topology.leader_neighbors.get(p, set()))
            if not neighbors:
                mixed_state[p] = new_plane_state[p]
                continue

            neighbor_states = [new_plane_state[q] for q in neighbors]
            if len(neighbor_states) == 1:
                avg_neighbor = neighbor_states[0]
            else:
                avg_neighbor = fednova.weighted_avg_aggregate(
                    neighbor_states, [1.0] * len(neighbor_states)
                )
            mixed_state[p] = fednova.convex_mix_states(
                new_plane_state[p], avg_neighbor, cfg.leader_mixing_beta
            )

            for q in neighbors:
                edge = (min(p, q), max(p, q))
                if edge in logged_edges:
                    continue
                logged_edges.add(edge)
                run_logs.mixing_log.add(
                    cycle_id=cycle_id,
                    edge_p=edge[0],
                    edge_q=edge[1],
                    mix_time_s=mix_time,
                    bytes_transferred=2 * model_bytes,
                )
                run_logs.comm_log.add(
                    sim_time_s=mix_time,
                    kind="inter_leader_mix",
                    plane_or_edge=f"{edge[0]}-{edge[1]}",
                    bytes=2 * model_bytes,
                    hops=2,
                )

        plane_state = mixed_state
        sim_time = mix_time
        do_checkpoint(min(sim_time, horizon))

    do_checkpoint(horizon)
    return run_logs, acc_log
