# Validates the capacity-proportional step formula (policies.py), the direct
# analogue of the existing offloading formula n_i' = (C_i/sum(C_j)) * sum(n_j)
# with "samples" replaced by "steps":
#   - equal capacity across a plane reduces exactly to the fixed-step baseline
#     (same sanity property FedNova has: reduces to FedAvg under equal steps)
#   - a 2x-faster satellite gets ~2x the steps of a baseline-speed one
#   - the plane's total step budget is conserved before any max_local_steps capping
import unittest

from LeoFL.followup.policies import capacity_proportional_steps
from LeoFL.followup.time_model import SatelliteTimeProfile


def _profile(
    sat_id: int, epoch_minutes: float, steps_per_epoch: int = 10
) -> SatelliteTimeProfile:
    return SatelliteTimeProfile(
        sat_id=sat_id,
        n_samples=100,
        batch_size=32,
        epoch_minutes=epoch_minutes,
        steps_per_epoch=steps_per_epoch,
        step_time_s=epoch_minutes * 60.0 / steps_per_epoch,
    )


class TestCapacityStepAllocation(unittest.TestCase):
    def test_equal_capacity_reduces_to_fixed_steps(self):
        members = [0, 1, 2]
        profiles = {m: _profile(m, epoch_minutes=10.0) for m in members}
        steps = capacity_proportional_steps(
            members, profiles, fixed_steps_baseline=7, max_local_steps=1000
        )
        self.assertEqual(steps, {0: 7, 1: 7, 2: 7})

    def test_faster_satellite_gets_proportionally_more_steps(self):
        members = [0, 1]
        # sat 1 has half the epoch_minutes of sat 0 -> twice the capacity (C=1/epoch_minutes)
        profiles = {
            0: _profile(0, epoch_minutes=10.0),
            1: _profile(1, epoch_minutes=5.0),
        }
        steps = capacity_proportional_steps(
            members, profiles, fixed_steps_baseline=6, max_local_steps=1000
        )
        # C0=0.1, C1=0.2, S_total=12 -> step0=round(12*0.1/0.3)=4, step1=round(12*0.2/0.3)=8
        self.assertEqual(steps, {0: 4, 1: 8})
        self.assertEqual(steps[1], 2 * steps[0])

    def test_total_step_budget_is_conserved_before_capping(self):
        members = [0, 1, 2, 3]
        profiles = {
            0: _profile(0, 8.0),
            1: _profile(1, 12.0),
            2: _profile(2, 20.0),
            3: _profile(3, 6.0),
        }
        fixed_steps_baseline = 10
        steps = capacity_proportional_steps(
            members, profiles, fixed_steps_baseline, max_local_steps=1000
        )
        total = sum(steps.values())
        expected_total = fixed_steps_baseline * len(members)
        # per-member rounding can shift the sum by at most ~0.5 per member
        self.assertLessEqual(abs(total - expected_total), len(members))

    def test_max_local_steps_cap_is_respected(self):
        members = [0, 1]
        # sat 1 is drastically faster -> would get a huge share without the cap
        profiles = {
            0: _profile(0, epoch_minutes=30.0),
            1: _profile(1, epoch_minutes=0.5),
        }
        steps = capacity_proportional_steps(
            members, profiles, fixed_steps_baseline=50, max_local_steps=20
        )
        self.assertLessEqual(steps[1], 20)
        self.assertGreaterEqual(steps[0], 1)


if __name__ == "__main__":
    unittest.main()
