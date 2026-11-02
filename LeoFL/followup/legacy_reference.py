# LeoFL/followup/legacy_reference.py
"""
Runs the *existing, unmodified* LH-FL offloading scenario
(``LeoFL.Simulator.run_simulator.run_simulation(mode="offload")`` ->
``scenarios.OffloadScenario``) as the "move data" reference to compare
against this package's "throttle steps" methods (``engine.py``).

Zero changes to ``scenarios.py``/``run_simulator.py`` -- this module only
builds the **same** client_datasets/constellation/per_sat_minutes the new
engine uses (same seed, same recipes) and feeds them into the existing
function. It writes the legacy CSV schema (``{prefix}_accuracy.csv`` with
columns ``round, accuracy``; ``{prefix}_comm.csv`` with columns
``time, model_bytes, offload_bytes, model_hops, offload_hops, note``) to its
own log dir, via the existing, unmodified code path.

Important: the legacy scenario's local training is ``fl_core.train_one_epoch``,
which always uses **Adam** (hardcoded) -- unrelated to the vanilla-SGD
``lr=0.1`` tuned for this package's own FedNova-based training. This module
therefore has its own ``lr`` (default ``1e-3``, the legacy project's own
default), not ``ExperimentConfig.lr``.

``run_simulator.run_simulation(mode="offload")`` also hardcodes
``max_dist_km=2000.0`` for ``OffloadScenario``'s leader selection (not
configurable through that function) -- this is why
``ExperimentConfig.leader_select_max_dist_km`` defaults to 2000.0 too, so the
new methods and this reference pick the *same* leaders/neighbor graph.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from LeoFL.EuroSAT import build_eurosat_embeddings
from LeoFL.Simulator.fl_core import make_clients_from_dataset
from LeoFL.Simulator.run_simulator import run_simulation

from LeoFL.followup.run_experiment import build_constellation_configs, split_train_test
from LeoFL.followup.time_model import draw_epoch_minutes


@dataclass
class LegacyReferenceConfig:
    seed: int = 0
    constellation: Dict[str, Any] = field(
        default_factory=lambda: {
            "kind": "starlink_active10",
            "full_num_planes": 72,
            "sats_per_plane": 22,
            "active_num_planes": 10,
            "h_km": 552.0,
            "incl_deg": 53.0,
        }
    )
    leader_top_k: int = 2
    leader_mixing_beta: float = 0.5
    non_iid_alpha: float = 0.5
    train_ratio: float = 0.8
    epoch_minutes_low: int = 5
    epoch_minutes_high: int = 31
    total_simulated_time_s: float = 60000.0
    seconds_per_round: int = 60
    target_accuracy: float = 85.0
    lr: float = 1e-3  # legacy Adam default, NOT the new engine's SGD lr=0.1
    hidden_size: int = 100
    log_dir: str = "./logs_followup_legacy_reference"
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "LegacyReferenceConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data.keys()) - known
        if unknown:
            raise ValueError(f"Unknown config fields in {path}: {sorted(unknown)}")
        return cls(**data)


def run(cfg: LegacyReferenceConfig) -> None:
    os.makedirs(cfg.log_dir, exist_ok=True)
    device = torch.device(
        cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )

    full_ds = build_eurosat_embeddings()
    train_ds, test_ds = split_train_test(full_ds, cfg.train_ratio, seed=cfg.seed)

    constellation_configs = build_constellation_configs(
        _ConstellationShim(cfg.constellation)  # see helper below
    )
    num_clients = len(constellation_configs)

    client_datasets = make_clients_from_dataset(
        dataset=train_ds,
        num_clients=num_clients,
        split_mode="non",
        alpha=cfg.non_iid_alpha,
        seed=cfg.seed,
    )

    per_sat_minutes = draw_epoch_minutes(
        num_clients,
        seed=cfg.seed,
        low=cfg.epoch_minutes_low,
        high=cfg.epoch_minutes_high,
    ).tolist()

    num_rounds = max(1, int(cfg.total_simulated_time_s // cfg.seconds_per_round))
    log_prefix = f"legacy_offload_{cfg.seed}"

    global_model, best_acc = run_simulation(
        mode="offload",
        client_datasets=client_datasets,
        test_dataset=test_ds,
        num_rounds=num_rounds,
        per_sat_minutes=per_sat_minutes,
        agg_name="fedavg",
        target_acc=cfg.target_accuracy,
        device=device,
        hidden_size=cfg.hidden_size,
        lr=cfg.lr,
        seconds_per_round=cfg.seconds_per_round,
        log_dir=cfg.log_dir,
        model_name="dnn",
        leader_top_k=cfg.leader_top_k,
        leader_mixing_beta=cfg.leader_mixing_beta,
        log_prefix=log_prefix,
        constellation_configs=constellation_configs,
    )

    cfg_path = os.path.join(cfg.log_dir, f"{log_prefix}_followup_wrapper_config.json")
    cfg.save(cfg_path)

    print(
        f"[legacy_reference] num_clients={num_clients} num_rounds={num_rounds} best_acc={best_acc:.2f}%"
    )
    print(f"[legacy_reference] logs written to: {cfg.log_dir} (prefix={log_prefix})")
    print(f"  - config: {cfg_path}")


class _ConstellationShim:
    """build_constellation_configs (run_experiment.py) only reads
    ``cfg.constellation``; this avoids depending on ExperimentConfig's full
    field set just to reuse that one function."""

    def __init__(self, constellation: Dict[str, Any]):
        self.constellation = constellation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the existing LH-FL offloading scenario as a reference."
    )
    parser.add_argument(
        "--config", required=True, help="Path to a LegacyReferenceConfig JSON file."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override config's seed."
    )
    parser.add_argument("--log-dir", default=None, help="Override config's log_dir.")
    parser.add_argument(
        "--total-simulated-time-s",
        type=float,
        default=None,
        help="Override config's total_simulated_time_s.",
    )
    args = parser.parse_args()

    cfg = LegacyReferenceConfig.load(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.log_dir is not None:
        cfg.log_dir = args.log_dir
    if args.total_simulated_time_s is not None:
        cfg.total_simulated_time_s = args.total_simulated_time_s

    run(cfg)


if __name__ == "__main__":
    main()
