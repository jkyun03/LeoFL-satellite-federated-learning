# Validates Method "offload": it must actually reduce the within-plane
# spread of N_i/C_i (the quantity fl_core.offload_within_plane equalizes),
# log the moved bytes as a one-time "data_offload" comm entry, and then
# train with the exact same fixed-step policy as Method "fixed" (offload
# only changes the data, never the step-decision logic).
import unittest

import numpy as np
import torch

from LeoFL.followup.config import ExperimentConfig
from LeoFL.followup.engine import run_followup_experiment
from LeoFL.followup.policies import METHOD_OFFLOAD
from LeoFL.followup.time_model import draw_epoch_minutes
from LeoFL.followup.tests._helpers import (
    make_synthetic_test_dataset,
    make_tiny_topology,
    make_variable_size_client_datasets,
)


class TestOffloadBalancesLoad(unittest.TestCase):
    def _make_scenario(self):
        num_planes, sats_per_plane = 2, 2
        topology = make_tiny_topology(
            num_planes=num_planes,
            sats_per_plane=sats_per_plane,
            horizon_s=5000.0,
            leader_top_k=1,
        )
        # deliberately skewed within each plane: member 0 has far more data than member 1
        sizes = [400, 20, 400, 20]
        seed = 5
        client_datasets = make_variable_size_client_datasets(sizes, seed=seed)
        test_dataset = make_synthetic_test_dataset(seed=seed + 1000)
        cfg = ExperimentConfig(
            method=METHOD_OFFLOAD,
            aggregation="weighted_avg",
            seed=seed,
            batch_size=8,
            lr=0.05,
            fixed_steps=5,
            max_local_steps=20,
            offload_ratio=0.5,
            offload_top_pairs_per_sat=1,
            isl_bandwidth_bps=1e9,
            per_hop_latency_s=0.2,
            leader_agg_time_per_member_s=0.01,
            total_simulated_time_s=5000.0,
            accuracy_checkpoint_interval_s=2500.0,
            log_dir=".",
        )
        return topology, client_datasets, test_dataset, cfg, sizes, seed

    def test_offload_reduces_within_plane_ni_over_ci_spread_and_logs_bytes(self):
        topology, client_datasets, test_dataset, cfg, sizes, seed = (
            self._make_scenario()
        )

        epoch_minutes = draw_epoch_minutes(
            len(sizes),
            seed=seed,
            low=cfg.epoch_minutes_low,
            high=cfg.epoch_minutes_high,
        )
        capacities = 1.0 / epoch_minutes
        before_ratio = np.array(sizes, dtype=float) / capacities
        before_spread = abs(before_ratio[0] - before_ratio[1]) + abs(
            before_ratio[2] - before_ratio[3]
        )

        run_logs, _acc_log = run_followup_experiment(
            cfg, topology, client_datasets, test_dataset, torch.device("cpu")
        )

        after_sizes = np.array([len(ds) for ds in client_datasets], dtype=float)
        after_ratio = after_sizes / capacities
        after_spread = abs(after_ratio[0] - after_ratio[1]) + abs(
            after_ratio[2] - after_ratio[3]
        )

        self.assertLess(
            after_spread,
            before_spread,
            "offload should reduce within-plane N_i/C_i spread",
        )
        self.assertFalse(
            np.array_equal(after_sizes, np.array(sizes, dtype=float)),
            "offload should actually move samples",
        )
        self.assertEqual(
            int(after_sizes.sum()),
            sum(sizes),
            "offload redistributes samples, never creates/destroys them",
        )

        comm_df = run_logs.comm_log.to_dataframe()
        offload_rows = comm_df[comm_df["kind"] == "data_offload"]
        self.assertEqual(
            len(offload_rows), 1, "exactly one one-shot data_offload comm entry"
        )
        self.assertGreater(offload_rows.iloc[0]["bytes"], 0)
        self.assertEqual(
            offload_rows.iloc[0]["sim_time_s"],
            0.0,
            "offload happens once, before cycle 1",
        )

    def test_offload_trains_with_the_same_fixed_step_policy_as_method_fixed(self):
        topology, client_datasets, test_dataset, cfg, _sizes, _seed = (
            self._make_scenario()
        )
        run_logs, _acc_log = run_followup_experiment(
            cfg, topology, client_datasets, test_dataset, torch.device("cpu")
        )

        step_df = run_logs.step_log.to_dataframe()
        self.assertTrue(
            (step_df["steps"] == cfg.fixed_steps).all(),
            "offload must use the exact fixed-step policy post-rebalance",
        )


if __name__ == "__main__":
    unittest.main()
