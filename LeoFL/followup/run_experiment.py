# LeoFL/followup/run_experiment.py
"""
CLI entry point for one (method, seed) run of the followup study.

Usage (from the repository root, e.g. in a VS Code terminal):

    python -m LeoFL.followup.run_experiment --config LeoFL/followup/configs/smoke.json --method fixed --seed 0

or, equivalently, running this file directly (a sys.path shim below makes
the ``LeoFL.*`` absolute imports work either way):

    python LeoFL/followup/run_experiment.py --config LeoFL/followup/configs/smoke.json --method fixed --seed 0

For the existing LH-FL offloading reference (unmodified code path), use
``legacy_reference.py`` instead -- see LeoFL/followup/README.md.

Reuses, unchanged: EuroSAT ResNet18 embedding + cache (LeoFL.EuroSAT), the
Dirichlet non-IID client split and Starlink-like constellation builders
(LeoFL.Simulator.fl_core). Training/scheduling/aggregation/evaluation is the
new code in this package (engine.py, policies.py, fednova.py, metrics.py).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

from LeoFL.EuroSAT import build_eurosat_embeddings, EmbedTabularDataset
from LeoFL.EuroSAT.eurosat_embed_dataset import _compute_balanced_class_weight
from LeoFL.Simulator.fl_core import (
    make_clients_from_dataset,
    make_starlink_shell_active10_constellation,
    make_walker_delta_constellation,
)

from LeoFL.followup.config import ExperimentConfig
from LeoFL.followup.engine import run_followup_experiment
from LeoFL.followup.metrics import accuracy_at_budget, time_to_target
from LeoFL.followup.topology import build_topology_cached

TOPOLOGY_CACHE_DIR = "./logs_followup/.topology_cache"


def split_train_test(full_ds: EmbedTabularDataset, train_ratio: float, seed: int):
    """80/20 split of the cached embedding dataset, matching
    simulation_eurosat.py's split_train_test exactly (same recipe, kept
    independent so this module has no import-time dependency on the EuroSAT
    driver script)."""
    n = len(full_ds)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_ratio)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    train_X, train_Y = full_ds.X[train_idx], full_ds.Y[train_idx]
    test_X, test_Y = full_ds.X[test_idx], full_ds.Y[test_idx]

    train_cw = _compute_balanced_class_weight(train_Y, full_ds.n_classes)
    train_ds = EmbedTabularDataset(
        train_X, train_Y, n_classes=full_ds.n_classes, class_weight=train_cw
    )
    test_ds = EmbedTabularDataset(
        test_X, test_Y, n_classes=full_ds.n_classes, class_weight=None
    )
    return train_ds, test_ds


def build_constellation_configs(cfg: ExperimentConfig):
    kind = cfg.constellation.get("kind", "starlink_active10")
    if kind == "starlink_active10":
        active_configs, _full, _active_ids = make_starlink_shell_active10_constellation(
            full_num_planes=cfg.constellation.get("full_num_planes", 72),
            sats_per_plane=cfg.constellation.get("sats_per_plane", 22),
            active_num_planes=cfg.constellation.get("active_num_planes", 10),
            h_km=cfg.constellation.get("h_km", 552.0),
            incl_deg=cfg.constellation.get("incl_deg", 53.0),
        )
        return active_configs
    elif kind == "toy_ring":
        # Small-scale (but still real Walker-delta orbital mechanics, not a
        # fabricated contact schedule) constellation for smoke tests only
        # (spec section 10: "합성 접촉 궤적은 개발 및 검증용으로 구분").
        return make_walker_delta_constellation(
            num_planes=cfg.constellation.get("num_planes", 3),
            sats_per_plane=cfg.constellation.get("sats_per_plane", 4),
            h_km=cfg.constellation.get("h_km", 552.0),
            incl_deg=cfg.constellation.get("incl_deg", 53.0),
        )
    else:
        raise ValueError(f"Unknown constellation kind: {kind!r}")


def run(cfg: ExperimentConfig) -> None:
    wall_start = time.time()
    os.makedirs(cfg.log_dir, exist_ok=True)

    device = torch.device(
        cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )

    full_ds = build_eurosat_embeddings()
    train_ds, test_ds = split_train_test(full_ds, cfg.train_ratio, seed=cfg.seed)

    constellation_configs = build_constellation_configs(cfg)
    num_clients = len(constellation_configs)

    client_datasets = make_clients_from_dataset(
        dataset=train_ds,
        num_clients=num_clients,
        split_mode="non",
        alpha=cfg.non_iid_alpha,
        seed=cfg.seed,
    )

    wall_data_ready = time.time()

    # Cached across every method/config sharing the same constellation/seed-
    # independent topology inputs -- see topology.build_topology_cached's
    # docstring; this is what turns "N methods x ~9min topology_build" into
    # "1 x ~9min" for a multi-method sweep at full (220-satellite) scale.
    topology = build_topology_cached(
        constellation_configs,
        horizon_s=cfg.total_simulated_time_s,
        cache_dir=TOPOLOGY_CACHE_DIR,
        leader_top_k=cfg.leader_top_k,
        leader_select_max_dist_km=cfg.leader_select_max_dist_km,
        log_dir=cfg.log_dir,
    )

    wall_topology_ready = time.time()

    run_logs, acc_log = run_followup_experiment(
        cfg, topology, client_datasets, test_ds, device
    )

    wall_engine_done = time.time()

    paths = run_logs.save_all()
    cfg_path = os.path.join(
        cfg.log_dir, f"followup_{cfg.method}_{cfg.seed}_config.json"
    )
    cfg.save(cfg_path)

    t_target = time_to_target(acc_log, cfg.target_accuracy)
    acc_at_horizon = accuracy_at_budget(acc_log, cfg.total_simulated_time_s)

    print(
        f"[followup] method={cfg.method} seed={cfg.seed} num_clients={num_clients} "
        f"num_orbits={len(topology.plane_to_sats)}"
    )
    print(
        f"[followup] time to reach target_accuracy={cfg.target_accuracy}%: "
        f"{'NOT REACHED' if t_target is None else f'{t_target:.1f}s'}"
    )
    print(
        f"[followup] weighted accuracy at horizon ({cfg.total_simulated_time_s:.0f}s): "
        f"{acc_at_horizon if acc_at_horizon is not None else float('nan'):.2f}%"
    )
    print(f"[followup] logs written to: {cfg.log_dir}")
    for artifact, path in paths.items():
        print(f"  - {artifact}: {path}")
    print(f"  - config: {cfg_path}")

    wall_end = time.time()
    print(
        f"[followup][timing] data_prep={wall_data_ready - wall_start:.1f}s "
        f"topology_build={wall_topology_ready - wall_data_ready:.1f}s "
        f"engine_run={wall_engine_done - wall_topology_ready:.1f}s "
        f"logging={wall_end - wall_engine_done:.1f}s "
        f"TOTAL={wall_end - wall_start:.1f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one followup-study experiment (fixed / capacity / capacity_minwork / offload)."
    )
    parser.add_argument(
        "--config", required=True, help="Path to an ExperimentConfig JSON file."
    )
    parser.add_argument(
        "--method",
        default=None,
        choices=["fixed", "capacity", "capacity_minwork", "offload"],
        help="Override config's method.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override config's seed."
    )
    parser.add_argument("--log-dir", default=None, help="Override config's log_dir.")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override config's lr (e.g. for an LR sweep).",
    )
    parser.add_argument(
        "--total-simulated-time-s",
        type=float,
        default=None,
        help="Override config's total_simulated_time_s (e.g. for a short calibration run).",
    )
    parser.add_argument(
        "--fixed-steps",
        type=int,
        default=None,
        help="Override config's fixed_steps (Method 'fixed' step count; also the per-satellite "
        "budget basis for 'capacity'/'capacity_minwork').",
    )
    parser.add_argument(
        "--minimum-epoch-fraction",
        type=float,
        default=None,
        help="Override config's minimum_epoch_fraction (Method 'capacity_minwork').",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    if args.method is not None:
        cfg.method = args.method
    if args.seed is not None:
        cfg.seed = args.seed
    if args.log_dir is not None:
        cfg.log_dir = args.log_dir
    if args.lr is not None:
        cfg.lr = args.lr
    if args.total_simulated_time_s is not None:
        cfg.total_simulated_time_s = args.total_simulated_time_s
    if args.fixed_steps is not None:
        cfg.fixed_steps = args.fixed_steps
    if args.minimum_epoch_fraction is not None:
        cfg.minimum_epoch_fraction = args.minimum_epoch_fraction

    run(cfg)


if __name__ == "__main__":
    main()
