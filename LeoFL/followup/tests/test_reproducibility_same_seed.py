# Validation item (spec section 11): "동일 seed와 설정으로 결과가 재현되는지"
import unittest

import torch

from LeoFL.followup.config import ExperimentConfig
from LeoFL.followup.engine import run_followup_experiment
from LeoFL.followup.policies import METHOD_CAPACITY_MINWORK
from LeoFL.followup.tests._helpers import (
    make_synthetic_client_datasets,
    make_synthetic_test_dataset,
    make_tiny_topology,
)


class TestReproducibility(unittest.TestCase):
    def test_same_seed_and_config_gives_identical_results(self):
        seed = 7
        cfg = ExperimentConfig(
            method=METHOD_CAPACITY_MINWORK,
            aggregation="fednova",
            seed=seed,
            batch_size=8,
            lr=0.05,
            fixed_steps=5,
            minimum_epoch_fraction=0.25,
            max_local_steps=15,
            isl_bandwidth_bps=1e9,
            per_hop_latency_s=0.2,
            leader_agg_time_per_member_s=0.01,
            total_simulated_time_s=3000.0,
            accuracy_checkpoint_interval_s=500.0,
            log_dir=".",
        )

        def run_once():
            topology = make_tiny_topology(
                num_planes=2, sats_per_plane=2, horizon_s=3000.0, leader_top_k=1
            )
            client_datasets = make_synthetic_client_datasets(
                num_clients=4, n_per_client=50, seed=seed
            )
            test_dataset = make_synthetic_test_dataset(seed=seed + 1000)
            return run_followup_experiment(
                cfg, topology, client_datasets, test_dataset, torch.device("cpu")
            )

        run_logs_1, acc_log_1 = run_once()
        run_logs_2, acc_log_2 = run_once()

        self.assertEqual(acc_log_1, acc_log_2)

        steps_1 = run_logs_1.step_log.to_dataframe()
        steps_2 = run_logs_2.step_log.to_dataframe()
        self.assertTrue(
            steps_1.equals(steps_2),
            "step logs must be byte-identical across repeated runs with the same seed",
        )

        mix_1 = run_logs_1.mixing_log.to_dataframe()
        mix_2 = run_logs_2.mixing_log.to_dataframe()
        self.assertTrue(
            mix_1.equals(mix_2),
            "mixing logs must be identical across repeated runs with the same seed",
        )


if __name__ == "__main__":
    unittest.main()
