# LeoFL/followup/logging_schema.py
"""
CSV logging for the followup study. Independent, own-namespace schema
(``followup_{method}_{seed}_{artifact}.csv``) rather than the legacy
``{prefix}_accuracy.csv``/``{prefix}_comm.csv`` schema ``analysis_utils.py``
consumes -- the columns needed here (per-satellite step counts, per-orbit
accuracy spread, per-class recall) have no equivalent there. The legacy
offloading reference (``legacy_reference.py``) deliberately writes the
*existing* schema instead (it calls the unmodified ``scenarios.py`` code),
so comparison plots read both schemas side by side (see ``plotting.py``).
"""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd


class CsvLog:
    """A small typed-columns row accumulator -> CSV writer, used for every
    log type below so column contracts are explicit and enforced at
    ``add()`` time rather than discovered later at analysis time."""

    def __init__(self, columns: List[str]):
        self.columns = list(columns)
        self.rows: List[dict] = []

    def add(self, **kwargs) -> None:
        missing = [c for c in self.columns if c not in kwargs]
        if missing:
            raise KeyError(
                f"CsvLog missing required columns {missing}; got {list(kwargs.keys())}"
            )
        extra = [k for k in kwargs if k not in self.columns]
        if extra:
            raise KeyError(
                f"CsvLog got unexpected columns {extra}; declared columns are {self.columns}"
            )
        self.rows.append({c: kwargs[c] for c in self.columns})

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.columns)

    def save(self, path: str) -> pd.DataFrame:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return df


def followup_prefix(method: str, seed: int) -> str:
    return f"followup_{method}_{seed}"


# --- per-satellite local-training record ---
STEP_LOG_COLUMNS = [
    "cycle_id",
    "plane_id",
    "sat_id",
    "is_leader",
    "model_version",
    "t_start_s",
    "t_end_s",
    "steps",
    "step_time_s",
    "samples_processed_cumulative",
    "unique_samples_seen_cumulative",
]

# --- intra-orbit aggregation record ---
AGGREGATION_LOG_COLUMNS = [
    "plane_id",
    "cycle_id",
    "model_version",
    "agg_time_s",
    "aggregation_type",
    "n_members",
    "tau_eff",
    "min_steps",
    "max_steps",
]

# --- inter-orbit mixing record. Unconditional every cycle (matches legacy
#     OffloadScenario: every plane mixes with its static top-k neighbors
#     every cycle, no feasibility gating), so there is no "status" -- just
#     when it happened and how many bytes it cost. ---
MIXING_LOG_COLUMNS = ["cycle_id", "edge_p", "edge_q", "mix_time_s", "bytes_transferred"]

# --- accuracy / loss over simulated time (weighted representative value,
#     plus per-orbit spread and the slow-satellite-subgroup metric so the
#     weighted number is never the only figure reported) ---
ACCURACY_LOG_COLUMNS = [
    "sim_time_s",
    "weighted_accuracy",
    "weighted_loss",
    "weighted_macro_f1",
    "orbit_acc_min",
    "orbit_acc_mean",
    "orbit_acc_max",
    "slow_subgroup_recall",
    "slow_subgroup_macro_f1",
]

# --- communication bytes accounting (reuses comm_Logger.calc_model_bytes
#     for the byte counts themselves) ---
COMM_LOG_COLUMNS = ["sim_time_s", "kind", "plane_or_edge", "bytes", "hops"]

# --- per-satellite static metadata (compute speed draw, data count, steps
#     per epoch) -- written once per run for reproducibility/inspection ---
SAT_PROFILE_COLUMNS = [
    "sat_id",
    "plane_id",
    "is_leader",
    "n_samples",
    "epoch_minutes",
    "steps_per_epoch",
    "step_time_s",
]


def make_step_log() -> CsvLog:
    return CsvLog(STEP_LOG_COLUMNS)


def make_aggregation_log() -> CsvLog:
    return CsvLog(AGGREGATION_LOG_COLUMNS)


def make_mixing_log() -> CsvLog:
    return CsvLog(MIXING_LOG_COLUMNS)


def make_accuracy_log() -> CsvLog:
    return CsvLog(ACCURACY_LOG_COLUMNS)


def make_comm_log() -> CsvLog:
    return CsvLog(COMM_LOG_COLUMNS)


def make_sat_profile_log() -> CsvLog:
    return CsvLog(SAT_PROFILE_COLUMNS)


class RunLogs:
    """Bundles every log for one (method, seed) run and saves them all under
    a consistent ``followup_{method}_{seed}_{artifact}.csv`` naming
    convention."""

    def __init__(self, method: str, seed: int, log_dir: str):
        self.method = method
        self.seed = seed
        self.log_dir = log_dir
        self.step_log = make_step_log()
        self.aggregation_log = make_aggregation_log()
        self.mixing_log = make_mixing_log()
        self.accuracy_log = make_accuracy_log()
        self.comm_log = make_comm_log()
        self.sat_profile_log = make_sat_profile_log()

    def _path(self, artifact: str) -> str:
        return os.path.join(
            self.log_dir, f"{followup_prefix(self.method, self.seed)}_{artifact}.csv"
        )

    def save_all(self) -> Dict[str, str]:
        paths = {}
        for artifact, log in (
            ("steps", self.step_log),
            ("aggregations", self.aggregation_log),
            ("mixing", self.mixing_log),
            ("accuracy", self.accuracy_log),
            ("comm", self.comm_log),
            ("sat_profile", self.sat_profile_log),
        ):
            path = self._path(artifact)
            log.save(path)
            paths[artifact] = path
        return paths
