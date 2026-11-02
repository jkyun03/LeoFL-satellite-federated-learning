# Validates the minimum-work floor (policies.apply_minimum_work_floor):
# a satellite whose capacity-proportional allocation falls below its
# fraction-based floor is raised to the floor; one already above it is left
# untouched; max_local_steps is still enforced after flooring.
import unittest

from LeoFL.followup.policies import apply_minimum_work_floor
from LeoFL.followup.time_model import SatelliteTimeProfile


def _profile(
    sat_id: int, epoch_minutes: float = 10.0, steps_per_epoch: int = 40
) -> SatelliteTimeProfile:
    return SatelliteTimeProfile(
        sat_id=sat_id,
        n_samples=1280,
        batch_size=32,
        epoch_minutes=epoch_minutes,
        steps_per_epoch=steps_per_epoch,
        step_time_s=epoch_minutes * 60.0 / steps_per_epoch,
    )


class TestMinWorkRaisesFloor(unittest.TestCase):
    def test_satellite_below_floor_is_raised_above_floor_untouched(self):
        members = [0, 1]
        profiles = {0: _profile(0), 1: _profile(1)}
        base_steps = {0: 2, 1: 20}  # floor = ceil(0.25*40) = 10
        result = apply_minimum_work_floor(
            base_steps,
            members,
            profiles,
            minimum_epoch_fraction=0.25,
            max_local_steps=1000,
        )
        self.assertEqual(result[0], 10, "below the floor must be raised to it")
        self.assertEqual(
            result[1], 20, "already above the floor must be left untouched"
        )

    def test_fraction_1_0_demands_a_full_epoch(self):
        base_steps = {0: 5}
        result = apply_minimum_work_floor(
            base_steps,
            [0],
            {0: _profile(0, steps_per_epoch=40)},
            minimum_epoch_fraction=1.0,
            max_local_steps=1000,
        )
        self.assertEqual(result[0], 40)

    def test_max_local_steps_cap_applies_after_flooring(self):
        base_steps = {0: 2}
        result = apply_minimum_work_floor(
            base_steps,
            [0],
            {0: _profile(0, steps_per_epoch=40)},
            minimum_epoch_fraction=1.0,
            max_local_steps=15,
        )
        self.assertEqual(
            result[0], 15, "the floor (40) must still be capped by max_local_steps"
        )


if __name__ == "__main__":
    unittest.main()
