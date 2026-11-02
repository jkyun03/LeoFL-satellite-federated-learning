# LeoFL/followup/time_model.py
"""
Step-level time model for the followup study.

The existing simulator (LeoFL/Simulator/fl_core.py, scenarios.py) only ever
trains satellites for exactly one full epoch per cycle, and abstracts
training time as a per-satellite "minutes per epoch" value
(``per_sat_minutes`` in LeoFL/EuroSAT/simulation_eurosat.py, drawn
``Uniform{5..30}`` minutes). There is no notion of a local *step* anywhere in
that code: batch_size=32 is hardcoded in ``fl_core.train_one_epoch`` and
``drop_last`` is never passed (PyTorch default ``False``), so one epoch is
always ``ceil(n_i / 32)`` full SGD steps.

This module introduces the missing step-level granularity while keeping the
same heterogeneity *magnitude* as the existing experiments, so that per-
satellite compute heterogeneity remains comparable across the old and new
studies:

  1. Draw ``epoch_minutes_i`` with the exact same recipe as
     ``simulation_eurosat.py``: ``rng.integers(low=5, high=31, size=N)``.
  2. ``steps_per_epoch_i = ceil(n_i / batch_size)`` (``drop_last=False``,
     matching ``fl_core.train_one_epoch``'s DataLoader).
  3. ``step_time_i = epoch_minutes_i * 60 / steps_per_epoch_i`` seconds.

This is a genuinely new modeling *assumption* (the legacy code never had
sub-epoch granularity to validate against) and is documented as such in
LeoFL/followup/README.md, not presented as a measured quantity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

# Matches LeoFL/EuroSAT/simulation_eurosat.py's per_sat_minutes draw exactly
# (rng.integers(low=5, high=31, size=NUM_CLIENTS)), so compute heterogeneity
# magnitude is comparable to the existing LH-FL experiments.
DEFAULT_EPOCH_MINUTES_LOW = 5
DEFAULT_EPOCH_MINUTES_HIGH = 31  # exclusive, i.e. draws in [5, 30]


def steps_per_epoch(n_samples: int, batch_size: int) -> int:
    """ceil(n_samples / batch_size), drop_last=False (same convention as
    fl_core.train_one_epoch's DataLoader)."""
    if n_samples <= 0:
        return 0
    return int(math.ceil(n_samples / float(batch_size)))


def draw_epoch_minutes(
    num_satellites: int,
    seed: int,
    low: int = DEFAULT_EPOCH_MINUTES_LOW,
    high: int = DEFAULT_EPOCH_MINUTES_HIGH,
) -> np.ndarray:
    """Per-satellite 'time to train one full epoch' heterogeneity draw,
    identical recipe to simulation_eurosat.py's per_sat_minutes."""
    rng = np.random.default_rng(seed)
    return rng.integers(low=low, high=high, size=num_satellites).astype(float)


def capacity_weights(epoch_minutes_per_sat: Dict[int, float]) -> Dict[int, float]:
    """C_i = 1 / epoch_minutes_i -- the same capacity definition the existing
    LH-FL offloading code uses (``capacity_i = 1/per_sat_minutes_i`` in
    ``OffloadScenario.__init__``), reused here (unnormalized; callers
    normalize within whatever group they're allocating a budget over, e.g.
    per orbit)."""
    return {sat_id: 1.0 / minutes for sat_id, minutes in epoch_minutes_per_sat.items()}


class ComputeProfile:
    """Interface for per-satellite, time-varying compute speed.

    The required method comparison only needs constant-in-time per-satellite
    speed (``ConstantComputeProfile``), but the interface supports
    time-varying performance per spec section 5 ("인터페이스는 시간에 따른
    계산 성능 변화도 지원하게 해라") without requiring a time-varying
    experiment to be part of the required comparison.
    """

    def step_time(self, sat_id: int, sim_time_s: float) -> float:
        raise NotImplementedError


class ConstantComputeProfile(ComputeProfile):
    """Per-satellite step time, constant over the whole simulation."""

    def __init__(self, step_time_per_sat: Dict[int, float]):
        self._step_time = dict(step_time_per_sat)

    def step_time(self, sat_id: int, sim_time_s: float) -> float:
        return self._step_time[sat_id]

    def as_dict(self) -> Dict[int, float]:
        return dict(self._step_time)


class PiecewiseDegradingComputeProfile(ComputeProfile):
    """Example time-varying profile: step time multiplies by
    ``degradation_factor`` once ``sim_time_s >= degrade_after_s``.

    Not used by the required A-D comparison; provided only to demonstrate
    that the ``ComputeProfile`` interface genuinely supports time variation.
    """

    def __init__(
        self,
        step_time_per_sat: Dict[int, float],
        degrade_after_s: float,
        degradation_factor: float = 1.5,
    ):
        self._base = dict(step_time_per_sat)
        self._degrade_after_s = float(degrade_after_s)
        self._degradation_factor = float(degradation_factor)

    def step_time(self, sat_id: int, sim_time_s: float) -> float:
        base = self._base[sat_id]
        if sim_time_s >= self._degrade_after_s:
            return base * self._degradation_factor
        return base


@dataclass
class SatelliteTimeProfile:
    sat_id: int
    n_samples: int
    batch_size: int
    epoch_minutes: float
    steps_per_epoch: int
    step_time_s: float


def build_time_profiles(
    n_samples_per_sat: List[int],
    batch_size: int,
    seed: int,
    low: int = DEFAULT_EPOCH_MINUTES_LOW,
    high: int = DEFAULT_EPOCH_MINUTES_HIGH,
) -> List[SatelliteTimeProfile]:
    """Build the per-satellite step-time profile used by every method.

    Satellites with zero local samples get ``steps_per_epoch=0`` and a
    ``step_time_s`` of +inf (they can never usefully train); callers should
    treat such satellites as always contributing zero steps.
    """
    epoch_minutes = draw_epoch_minutes(len(n_samples_per_sat), seed, low, high)
    profiles = []
    for i, n_i in enumerate(n_samples_per_sat):
        spe = steps_per_epoch(n_i, batch_size)
        if spe <= 0:
            step_time_s = float("inf")
        else:
            step_time_s = float(epoch_minutes[i]) * 60.0 / spe
        profiles.append(
            SatelliteTimeProfile(
                sat_id=i,
                n_samples=n_i,
                batch_size=batch_size,
                epoch_minutes=float(epoch_minutes[i]),
                steps_per_epoch=spe,
                step_time_s=step_time_s,
            )
        )
    return profiles


def make_constant_compute_profile(
    profiles: List[SatelliteTimeProfile],
) -> ConstantComputeProfile:
    return ConstantComputeProfile({p.sat_id: p.step_time_s for p in profiles})


def minimum_steps_for_fraction(
    minimum_epoch_fraction: float, steps_per_epoch_i: int
) -> int:
    """minimum_steps_i = max(1, ceil(minimum_epoch_fraction * steps_per_epoch_i))

    Matches spec section 4-D exactly.
    """
    if steps_per_epoch_i <= 0:
        return 1
    return max(1, int(math.ceil(minimum_epoch_fraction * steps_per_epoch_i)))
