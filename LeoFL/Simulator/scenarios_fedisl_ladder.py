# scenarios_fedisl_ladder.py
#
# FedISL -> LH-FL ablation ladder.
#
# Provides ONE new scenario used for two rungs of the ladder:
#     B0 = FedISL, unified n_i/C_i time model, NO offload   (do_offload=False)
#     B1 = FedISL, unified n_i/C_i time model, + offload     (do_offload=True)
# The third rung (B2 = LH-FL Design B) reuses run_contact_aware_lhfl.
#
# Why the time-model change:
#   The original FedISLScenario samples training time from a FIXED per-satellite
#   mean (base_train_minutes = per_sat_minutes), independent of data size, so
#   one-shot data offloading would NOT change its round-completion time at all.
#   To let offloading have a timing effect (and to match the paper's Eq.3,
#   T_i ∝ n_i/C_i, used by LH-FL), this scenario:
#       (1) optionally performs the same capacity-aware one-shot offload, then
#       (2) recomputes base_train_minutes = time_scale_const * (n_i / C_i) from
#           the CURRENT (post-offload) data sizes, and
#       (3) runs the ORIGINAL FedISL logic (source/sink, intra-plane ring
#           distribution, sink->PS contact-window wait, PS central aggregation)
#           unchanged, now driven by the n_i/C_i time model.
#
# Everything about FedISL's aggregation / connectivity structure is inherited
# from FedISLScenario and left intact.

import os
import numpy as np

from LeoFL.Simulator.fl_core import (
    offload_balance_datasets,
    build_transfer_plan_within_plane,
    save_class_distribution_csv,
    SimConfig,
)
from LeoFL.Simulator.scenarios import (
    FedISLScenario,
    ring_hop_distance,
)


class FedISLCapacityScenario(FedISLScenario):
    """
    FedISL with a capacity-aware n_i/C_i time model and optional one-shot
    capacity-aware data offloading (ladder rungs B0 / B1).
    """

    def __init__(
        self,
        client_datasets,
        test_dataset,
        config: SimConfig,
        log_prefix: str = "sat_events_fedisl_cap",
        per_sat_minutes=None,
        train_minutes: int = 10,
        do_offload: bool = False,
        offload_ratio: float = 0.1,
    ):
        super().__init__(
            client_datasets,
            test_dataset,
            config,
            log_prefix,
            per_sat_minutes,
            train_minutes,
        )

        # capacity C_i ∝ 1/minutes_i (normalized), and time_scale_const
        # tuned so that initial n_i/C_i times match per_sat_minutes on average.
        if per_sat_minutes is not None:
            mins = np.asarray(per_sat_minutes, dtype=float)
        else:
            mins = np.full(self.n_clients, float(train_minutes))

        cap = 1.0 / np.clip(mins, 1e-6, None)
        cap = cap / cap.mean()
        self.train_capacities = cap

        Ni0 = np.array([len(ds) for ds in client_datasets], dtype=float)
        ratio0 = Ni0 / np.clip(cap, 1e-6, None)
        self.time_scale_const = float(np.mean(mins / np.clip(ratio0, 1e-6, None)))

        self.do_offload = do_offload
        self.offload_ratio = offload_ratio

        print("[FedISL-CAP] train_capacities(norm) =", self.train_capacities.tolist())
        print("[FedISL-CAP] time_scale_const =", self.time_scale_const)
        print("[FedISL-CAP] do_offload =", self.do_offload)

    def _nc_mean_minutes(self, i: int) -> float:
        Ni = float(len(self.client_datasets[i]))
        Si = float(self.train_capacities[i])
        return max(self.time_scale_const * (Ni / max(Si, 1e-6)), 0.5)

    def run(self):
        # class distribution before offload
        save_class_distribution_csv(
            self.client_datasets,
            os.path.join(
                self.cfg.log_dir or ".", f"{self.log_prefix}_classdist_before.csv"
            ),
        )

        # (1) optional one-shot capacity-aware offload
        if self.do_offload:
            # capacity mode of offload_balance_datasets ignores meeting_counts,
            # so a zero matrix is sufficient (no leader/meeting logic needed here).
            dummy_mc = np.zeros((self.n_clients, self.n_clients), dtype=int)
            offload_stats = offload_balance_datasets(
                client_datasets=self.client_datasets,
                meeting_counts=dummy_mc,
                plane_ids=self.plane_ids,
                offload_ratio=self.offload_ratio,
                top_pairs_per_sat=1,
                seed=0,
                train_capacities=self.train_capacities,
            )

            # one-shot offload comm cost (hops/bytes), logged at round 0,
            # same accounting as OffloadScenario for a fair comm comparison.
            offload_hops_total = 0
            offload_bytes_total = 0
            offload_samples_total = 0
            for p, sat_indices in self.plane_to_sats.items():
                m = len(sat_indices)
                if m <= 1:
                    continue
                Nb = np.array(
                    [offload_stats["Ni_before"][i] for i in sat_indices], dtype=int
                )
                Nt = np.array(
                    [offload_stats["Ni_target"][i] for i in sat_indices], dtype=int
                )
                for s_local, r_local, delta in build_transfer_plan_within_plane(Nb, Nt):
                    hop = ring_hop_distance(s_local, r_local, m)
                    moved = int(abs(delta))
                    if moved <= 0:
                        continue
                    offload_hops_total += int(hop)
                    offload_bytes_total += int(moved * self.sample_bytes * hop)
                    offload_samples_total += moved

            self.comm_logger.log(
                0,
                model_bytes=0,
                offload_bytes=int(offload_bytes_total),
                model_hops=0,
                offload_hops=int(offload_hops_total),
                note=(
                    f"fedisl_one_shot_offload,samples={offload_samples_total},"
                    f"sample_bytes={self.sample_bytes},hops={offload_hops_total}"
                ),
            )

        # class distribution after offload
        save_class_distribution_csv(
            self.client_datasets,
            os.path.join(
                self.cfg.log_dir or ".", f"{self.log_prefix}_classdist_after.csv"
            ),
        )

        # (2) recompute base_train_minutes from CURRENT (post-offload) sizes: n_i/C_i
        self.base_train_minutes = np.array(
            [self._nc_mean_minutes(i) for i in range(self.n_clients)], dtype=float
        )
        print(
            "[FedISL-CAP] base_train_minutes (n_i/C_i) =",
            [round(x, 2) for x in self.base_train_minutes.tolist()],
        )

        # (3) run the ORIGINAL FedISL logic, now driven by the n_i/C_i time model
        return super().run()


def run_fedisl_capacity(
    client_datasets,
    test_dataset,
    per_sat_minutes,
    num_rounds,
    constellation_configs,
    device,
    log_dir,
    log_prefix,
    do_offload: bool = False,
    offload_ratio: float = 0.1,
    target_acc: float = 85.0,
    seconds_per_round: int = 60,
    hidden_size: int = 100,
    lr: float = 1e-3,
    agg_name: str = "fedavg",
):
    """Convenience runner for ladder rungs B0 (do_offload=False) and B1 (True)."""
    cfg = SimConfig(
        num_rounds=num_rounds,
        device=device,
        hidden_size=hidden_size,
        lr=lr,
        seconds_per_round=seconds_per_round,
        log_dir=log_dir,
        target_acc=target_acc,
        aggregate_name=agg_name,
        constellation_configs=constellation_configs,
        log_prefix=log_prefix,
        model_name="dnn",
    )
    if len(constellation_configs) != len(client_datasets):
        raise ValueError("constellation_configs length must match client_datasets.")
    if len(per_sat_minutes) != len(client_datasets):
        raise ValueError("per_sat_minutes length must match client_datasets.")

    scenario = FedISLCapacityScenario(
        client_datasets=client_datasets,
        test_dataset=test_dataset,
        config=cfg,
        log_prefix=log_prefix,
        per_sat_minutes=per_sat_minutes,
        train_minutes=10,
        do_offload=do_offload,
        offload_ratio=offload_ratio,
    )
    return scenario.run()
