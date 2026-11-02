# Validation item (spec section 11): "step 수가 다른 작은 수치 예제에서
# FedNova 집계가 맞는지" -- a hand-computed example.
#
# p_i = n_i / sum(n_j); here n = [1, 1] -> p = [0.5, 0.5]
# tau = [1, 4] -> tau_eff = 0.5*1 + 0.5*4 = 2.5
# d_1 = (2 - 0) / 1 = 2.0 ; d_2 = (4 - 0) / 4 = 1.0
# w_new = w_start + tau_eff * (p_1*d_1 + p_2*d_2)
#       = 0 + 2.5 * (0.5*2.0 + 0.5*1.0) = 2.5 * 1.5 = 3.75
import unittest

import torch

from LeoFL.followup import fednova


class TestFedNovaNumericExample(unittest.TestCase):
    def test_hand_computed_scalar_example(self):
        start = {"w": torch.tensor([0.0])}
        end_fast = {"w": torch.tensor([2.0])}  # tau=1
        end_slow = {"w": torch.tensor([4.0])}  # tau=4

        result = fednova.fednova_aggregate(
            start, [end_fast, end_slow], [1.0, 1.0], [1, 4]
        )
        self.assertAlmostEqual(result["w"].item(), 3.75, places=5)

    def test_hand_computed_weighted_example(self):
        # n = [3, 1] -> p = [0.75, 0.25]; tau = [2, 10]
        # tau_eff = 0.75*2 + 0.25*10 = 1.5 + 2.5 = 4.0
        # d_1 = (6-0)/2 = 3.0 ; d_2 = (20-0)/10 = 2.0
        # w_new = 4.0 * (0.75*3.0 + 0.25*2.0) = 4.0 * (2.25+0.5) = 4.0*2.75 = 11.0
        start = {"w": torch.tensor([0.0])}
        end_a = {"w": torch.tensor([6.0])}
        end_b = {"w": torch.tensor([20.0])}
        result = fednova.fednova_aggregate(start, [end_a, end_b], [3.0, 1.0], [2, 10])
        self.assertAlmostEqual(result["w"].item(), 11.0, places=5)

    def test_non_floating_buffers_pass_through_from_start(self):
        start = {"w": torch.tensor([0.0]), "n": torch.tensor([7], dtype=torch.long)}
        end = {"w": torch.tensor([2.0]), "n": torch.tensor([99], dtype=torch.long)}
        result = fednova.fednova_aggregate(start, [end], [1.0], [1])
        self.assertEqual(
            result["n"].item(), 7
        )  # carried from start_state, not averaged


if __name__ == "__main__":
    unittest.main()
