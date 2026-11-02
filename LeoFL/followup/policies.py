# LeoFL/followup/policies.py
"""
Step-count decision rules for the three compared methods, and the
aggregation-policy dispatch (FedNova vs. plain weighted average).

Research framing: the existing LH-FL mitigates stragglers by **moving data**
before training (``fl_core.offload_within_plane``: ``n_i' = (C_i/ΣC_j) × Σn_j``,
redistributing a plane's total sample count proportional to capacity share).
This module's capacity-aware policy mitigates the same straggler problem
**without moving any data**, by redistributing a plane's total *step budget*
the same way:

- Method "fixed": every satellite does the same fixed step count, regardless
  of speed (baseline -- reproduces the existing straggler problem, no
  mitigation).
- Method "capacity": steps are redistributed within each plane proportional
  to capacity share, holding the plane's total step budget constant
  (``S_total(p) = fixed_steps * |members(p)|``) -- the direct analogue of the
  offloading formula above, with "samples" replaced by "steps". Reduces
  exactly to "fixed" when every satellite in a plane has equal capacity.
- Method "capacity_minwork": "capacity" plus a floor
  (``minimum_steps_for_fraction``) so a satellite's local step count never
  drops so low that its (possibly rare-class) data is effectively never
  seen. The floor's cost (extra total compute when it binds) is not clawed
  back from faster satellites -- that cost *is* the straggler-mitigation vs.
  data-quality trade-off this study measures.
- Method "offload": the *fair-comparison* reference for "move data" instead
  of "throttle steps". Reuses the existing, unmodified
  ``fl_core.offload_balance_datasets``/``offload_within_plane`` to
  redistribute each plane's samples proportional to capacity share
  (``n_i' = (C_i/ΣC_j) × Σn_j``) **once**, before training starts, then runs
  the exact same fixed-step policy as "fixed" on the rebalanced data. Every
  other mechanism (optimizer, time model, aggregation weighting, barrier) is
  identical to "fixed"/"capacity"/"capacity_minwork" -- see ``engine.py`` --
  so this isolates "data moved vs. not" as the only variable, unlike
  ``legacy_reference.py`` (which also differs in optimizer and aggregation
  weighting from every method here).

These are plain, pure functions (no engine dependency) so they are unit
testable in isolation; ``engine.py`` calls them once per cycle per plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from LeoFL.followup.time_model import (
    SatelliteTimeProfile,
    capacity_weights,
    minimum_steps_for_fraction,
)
from LeoFL.followup import fednova

METHOD_FIXED = "fixed"
METHOD_CAPACITY = "capacity"
METHOD_CAPACITY_MINWORK = "capacity_minwork"
METHOD_OFFLOAD = "offload"
VALID_METHODS = (METHOD_FIXED, METHOD_CAPACITY, METHOD_CAPACITY_MINWORK, METHOD_OFFLOAD)


def fixed_steps(
    members: Sequence[int], steps: int, max_local_steps: int
) -> Dict[int, int]:
    """Method "fixed": every satellite performs the same fixed step count,
    independent of its own compute speed. The baseline straggler problem."""
    s = max(1, min(int(steps), int(max_local_steps)))
    return {m: s for m in members}


def capacity_proportional_steps(
    members: Sequence[int],
    time_profiles: Dict[int, SatelliteTimeProfile],
    fixed_steps_baseline: int,
    max_local_steps: int,
) -> Dict[int, int]:
    """Method "capacity": redistribute a plane's total step budget
    (``fixed_steps_baseline * len(members)``) proportional to each member's
    capacity share, mirroring ``n_i' = (C_i/ΣC_j) × Σn_j``.

    ``step_i = round(S_total * C_i / sum(C_j))``, clamped to ``[1,
    max_local_steps]``. When every member has equal capacity this reduces
    exactly to ``fixed_steps_baseline`` (see
    tests/test_capacity_step_allocation.py).
    """
    epoch_minutes = {m: time_profiles[m].epoch_minutes for m in members}
    cap = capacity_weights(epoch_minutes)
    total_cap = sum(cap.values())
    s_total = fixed_steps_baseline * len(members)

    out: Dict[int, int] = {}
    for m in members:
        share = cap[m] / total_cap if total_cap > 0 else 1.0 / len(members)
        steps_i = int(round(s_total * share))
        out[m] = max(1, min(int(max_local_steps), steps_i))
    return out


def apply_minimum_work_floor(
    steps_per_member: Dict[int, int],
    members: Sequence[int],
    time_profiles: Dict[int, SatelliteTimeProfile],
    minimum_epoch_fraction: float,
    max_local_steps: int,
) -> Dict[int, int]:
    """Method "capacity_minwork": raise any member below its fraction-based
    floor up to that floor; members already above it are left untouched.
    ``minimum_steps_i = max(1, ceil(minimum_epoch_fraction * steps_per_epoch_i))``
    (unchanged formula from ``time_model.minimum_steps_for_fraction``)."""
    out: Dict[int, int] = {}
    for m in members:
        floor = minimum_steps_for_fraction(
            minimum_epoch_fraction, time_profiles[m].steps_per_epoch
        )
        out[m] = max(1, min(int(max_local_steps), max(steps_per_member[m], floor)))
    return out


AGGREGATION_FEDNOVA = "fednova"
AGGREGATION_WEIGHTED_AVG = "weighted_avg"


@dataclass
class AggregationPolicy:
    kind: str  # AGGREGATION_FEDNOVA | AGGREGATION_WEIGHTED_AVG

    def aggregate(
        self,
        start_state,
        end_states: List[dict],
        weights: List[float],
        steps: List[int],
    ) -> dict:
        if self.kind == AGGREGATION_FEDNOVA:
            return fednova.fednova_aggregate(start_state, end_states, weights, steps)
        elif self.kind == AGGREGATION_WEIGHTED_AVG:
            return fednova.weighted_avg_aggregate(end_states, weights)
        else:
            raise ValueError(f"Unknown aggregation policy kind: {self.kind!r}")
