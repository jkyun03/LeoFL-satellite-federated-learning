# Validation item (spec section 11): "모든 위성의 step 수가 같을 때, 단순 SGD
# 조건의 FedNova가 FedAvg와 일치하는지"
import unittest

import torch

from LeoFL.followup import fednova


class TestFedNovaEqualsFedAvgWhenStepsEqual(unittest.TestCase):
    def test_equal_steps_matches_weighted_fedavg(self):
        torch.manual_seed(0)
        start = {"w": torch.zeros(5), "b": torch.zeros(2)}
        ends = [
            {"w": torch.randn(5), "b": torch.randn(2)},
            {"w": torch.randn(5), "b": torch.randn(2)},
            {"w": torch.randn(5), "b": torch.randn(2)},
        ]
        weights = [10.0, 20.0, 30.0]
        tau = 7  # same step count for every client

        fednova_result = fednova.fednova_aggregate(
            start, ends, weights, [tau, tau, tau]
        )
        fedavg_result = fednova.weighted_avg_aggregate(ends, weights)

        for key in start:
            torch.testing.assert_close(fednova_result[key], fedavg_result[key])

    def test_unequal_steps_differs_from_plain_fedavg(self):
        """Sanity check that the two aggregators are NOT trivially identical
        in general (otherwise the equal-steps test above would be vacuous)."""
        start = {"w": torch.zeros(3)}
        ends = [
            {"w": torch.tensor([2.0, 2.0, 2.0])},
            {"w": torch.tensor([4.0, 4.0, 4.0])},
        ]
        weights = [1.0, 1.0]
        fednova_result = fednova.fednova_aggregate(start, ends, weights, [1, 4])
        fedavg_result = fednova.weighted_avg_aggregate(ends, weights)
        self.assertFalse(torch.allclose(fednova_result["w"], fedavg_result["w"]))

    def test_rejects_non_vanilla_sgd_scope(self):
        start = {"w": torch.zeros(3)}
        end = {"w": torch.ones(3)}
        with self.assertRaises(NotImplementedError):
            fednova.fednova_aggregate(start, [end], [1.0], [1], optimizer_name="adam")
        with self.assertRaises(NotImplementedError):
            fednova.fednova_aggregate(start, [end], [1.0], [1], momentum=0.9)


if __name__ == "__main__":
    unittest.main()
