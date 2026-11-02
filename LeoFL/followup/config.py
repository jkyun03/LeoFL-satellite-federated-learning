# LeoFL/followup/config.py
"""
Experiment configuration: a plain dataclass + JSON I/O (stdlib ``json``, no
PyYAML -- not installed in this project's venv).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from LeoFL.followup.policies import AGGREGATION_FEDNOVA, METHOD_FIXED


@dataclass
class ExperimentConfig:
    # --- identity ---
    method: str = METHOD_FIXED  # "fixed" | "capacity" | "capacity_minwork" | "offload"
    seed: int = 0
    aggregation: str = AGGREGATION_FEDNOVA  # FedNova everywhere; reduces to weighted-avg when steps are equal (Method "fixed")

    # --- constellation (reuses LeoFL.Simulator.fl_core builders) ---
    # kind: "starlink_active10" (real Walker-delta orbit mechanics, same as
    #        existing LH-FL EuroSAT experiments) | "toy_ring" (small-scale,
    #        still real orbital mechanics, for smoke tests only)
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
    # Used only for the one-time leader-selection meeting-count computation
    # (fl_core.precompute_meetings_and_select_leaders) -- "기존 LH-FL 리더
    # 구조" setup, unrelated to any runtime contact-timing logic (there is
    # none in this engine). Matches the value run_simulator.run_simulation's
    # mode="offload" branch hardcodes for OffloadScenario (max_dist_km=2000.0,
    # not configurable there) so the legacy reference run picks the *same*
    # leaders/neighbor graph as these methods, per the "동일 리더 구성" requirement.
    leader_select_max_dist_km: float = 2000.0

    # --- data ---
    batch_size: int = 32
    lr: float = (
        0.1  # vanilla SGD (FedNova scope); empirically tuned via LR sweep, see README
    )
    non_iid_alpha: float = 0.5
    train_ratio: float = 0.8

    # --- compute heterogeneity (time_model.py) ---
    epoch_minutes_low: int = 5
    epoch_minutes_high: int = 31  # exclusive
    time_varying_compute: bool = (
        False  # interface support only; unused by the required methods
    )

    # --- communication timing model (documented assumptions, README) ---
    isl_bandwidth_bps: float = (
        50e6  # 50 Mbps ISL link, dedicated per edge (no cross-link contention modeled)
    )
    per_hop_latency_s: float = 0.5  # fixed propagation+processing latency per ISL hop
    leader_agg_time_per_member_s: float = (
        0.05  # trivial compute cost of averaging a small DNN
    )

    # --- method-specific knobs ---
    fixed_steps: int = 5  # Method "fixed"'s step count, and the
    # per-satellite step budget basis
    # (S_total = fixed_steps * plane_size) for
    # "capacity"/"capacity_minwork"
    minimum_epoch_fraction: Optional[float] = 0.25  # Method "capacity_minwork"
    max_local_steps: int = 50
    leader_mixing_beta: float = 0.5

    # --- Method "offload" only: one-shot pre-training data redistribution,
    # via the existing, unmodified fl_core.offload_balance_datasets.
    # Defaults match OffloadScenario's own (offload_ratio=0.1, top_pairs_per_sat=1).
    # NOTE: offload_balance_datasets only uses these two in its "swap" mode
    # (train_capacities=None); in "capacity" mode (what engine.py calls),
    # it always fully equalizes N_i/C_i within each plane via
    # offload_within_plane, so these two are currently inert there too --
    # the same way OffloadScenario itself passes (and silently doesn't use)
    # them in capacity mode. Kept for call-signature parity with the legacy
    # scenario, not because they currently do anything in this path.
    offload_ratio: float = 0.1
    offload_top_pairs_per_sat: int = 1

    # --- simulation horizon / stopping ---
    total_simulated_time_s: float = 60000.0
    target_accuracy: float = 85.0
    accuracy_checkpoint_interval_s: Optional[float] = (
        None  # defaults to total_simulated_time_s/100
    )

    # --- misc ---
    log_dir: str = "./logs_followup"
    device: str = "cpu"

    def effective_accuracy_checkpoint_interval_s(self) -> float:
        if self.accuracy_checkpoint_interval_s is not None:
            return float(self.accuracy_checkpoint_interval_s)
        return max(self.total_simulated_time_s / 100.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data.keys()) - known
        if unknown:
            raise ValueError(f"Unknown config fields in {path}: {sorted(unknown)}")
        return cls(**data)
