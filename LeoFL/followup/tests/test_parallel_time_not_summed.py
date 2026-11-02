# Validation: "위성의 병렬 학습 시간을 잘못 합산하지 않는지" -- and, since the
# redesigned engine reproduces the existing LH-FL's *whole-constellation*
# barrier (not just a per-orbit one), this also checks that the recorded
# aggregation time is the same single global value across every plane in a
# cycle, not computed independently per plane.
import unittest

import torch

from LeoFL.followup.config import ExperimentConfig
from LeoFL.followup.engine import agg_time_s, run_followup_experiment
from LeoFL.followup.policies import METHOD_FIXED
from LeoFL.followup.tests._helpers import (
    make_synthetic_client_datasets,
    make_synthetic_test_dataset,
    make_tiny_topology,
)


class TestGlobalBarrierIsMaxNotSum(unittest.TestCase):
    def test_aggregation_time_equals_global_max_and_is_shared_across_planes(self):
        num_planes, sats_per_plane = 3, 2
        topology = make_tiny_topology(
            num_planes=num_planes,
            sats_per_plane=sats_per_plane,
            horizon_s=20000.0,
            leader_top_k=1,
        )
        client_datasets = make_synthetic_client_datasets(
            num_clients=num_planes * sats_per_plane, n_per_client=40, seed=11
        )
        test_dataset = make_synthetic_test_dataset(seed=12)

        cfg = ExperimentConfig(
            method=METHOD_FIXED,
            aggregation="weighted_avg",
            seed=11,
            batch_size=8,
            lr=0.05,
            fixed_steps=5,
            max_local_steps=20,
            isl_bandwidth_bps=1e9,
            per_hop_latency_s=0.2,
            leader_agg_time_per_member_s=0.01,
            total_simulated_time_s=20000.0,
            accuracy_checkpoint_interval_s=5000.0,
            log_dir=".",
        )

        run_logs, _acc_log = run_followup_experiment(
            cfg, topology, client_datasets, test_dataset, torch.device("cpu")
        )

        step_df = run_logs.step_log.to_dataframe()
        agg_df = run_logs.aggregation_log.to_dataframe()

        cycle1_steps = step_df[step_df["cycle_id"] == 1]
        self.assertEqual(
            len(cycle1_steps),
            num_planes * sats_per_plane,
            "cycle 1 must include every satellite in every plane",
        )

        expected_agg_time = cycle1_steps["t_end_s"].max() + agg_time_s(
            sats_per_plane, cfg
        )
        cycle1_agg = agg_df[agg_df["cycle_id"] == 1]

        # One shared global-barrier value, identical across every plane this cycle
        # (not a separately-computed per-plane max).
        self.assertEqual(cycle1_agg["agg_time_s"].nunique(), 1)
        self.assertAlmostEqual(
            float(cycle1_agg["agg_time_s"].iloc[0]), expected_agg_time, places=6
        )

        # And it is the MAX across every satellite in every plane, not a sum:
        # whenever finish times actually differ, the barrier must land well
        # below their naive sum.
        finish_times = cycle1_steps["t_end_s"].values
        if finish_times.max() != finish_times.min():
            self.assertLess(
                float(cycle1_agg["agg_time_s"].iloc[0]), float(finish_times.sum())
            )


if __name__ == "__main__":
    unittest.main()
