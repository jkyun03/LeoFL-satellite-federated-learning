# Validation item (spec section 11): "모든 위성의 업데이트가 궤도 내 집계에
# 포함되는지"
#
# Under Method "fixed" (uniform step count > 0 for everyone), every orbit
# member must appear in every cycle's step log, and every intra-orbit
# aggregation must include every member of that orbit.
import unittest

import torch

from LeoFL.followup.config import ExperimentConfig
from LeoFL.followup.engine import run_followup_experiment
from LeoFL.followup.policies import METHOD_FIXED
from LeoFL.followup.tests._helpers import (
    make_synthetic_client_datasets,
    make_synthetic_test_dataset,
    make_tiny_topology,
)


class TestAllUpdatesIncludedInAggregation(unittest.TestCase):
    def test_every_cycle_includes_every_orbit_member(self):
        num_planes, sats_per_plane = 2, 3
        topology = make_tiny_topology(
            num_planes=num_planes,
            sats_per_plane=sats_per_plane,
            horizon_s=3000.0,
            leader_top_k=1,
        )
        client_datasets = make_synthetic_client_datasets(
            num_clients=num_planes * sats_per_plane, n_per_client=40, seed=3
        )
        test_dataset = make_synthetic_test_dataset(seed=4)

        cfg = ExperimentConfig(
            method=METHOD_FIXED,
            aggregation="weighted_avg",
            seed=3,
            batch_size=8,
            lr=0.05,
            fixed_steps=5,
            max_local_steps=20,
            isl_bandwidth_bps=1e9,
            per_hop_latency_s=0.2,
            leader_agg_time_per_member_s=0.01,
            total_simulated_time_s=3000.0,
            accuracy_checkpoint_interval_s=500.0,
            log_dir=".",
        )

        run_logs, _acc_log = run_followup_experiment(
            cfg, topology, client_datasets, test_dataset, torch.device("cpu")
        )

        agg_df = run_logs.aggregation_log.to_dataframe()
        self.assertGreater(
            len(agg_df), 0, "at least one intra-orbit aggregation should have happened"
        )
        self.assertTrue(
            (agg_df["n_members"] == sats_per_plane).all(),
            agg_df[["plane_id", "cycle_id", "n_members"]],
        )

        step_df = run_logs.step_log.to_dataframe()
        group_sizes = step_df.groupby(["plane_id", "cycle_id"]).size()
        self.assertTrue(
            (group_sizes == sats_per_plane).all(),
            group_sizes[group_sizes != sats_per_plane],
        )


if __name__ == "__main__":
    unittest.main()
