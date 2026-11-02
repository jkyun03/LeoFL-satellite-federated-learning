# CIFAR-10 실험 진입점: Centralized(always) / Hybrid-Offload(LH-FL) / Decentralized FL을 같은 조건으로 실행하고 비교 그래프를 logs_cifar10/에 저장한다.
import os
import random
import numpy as np
import torch

from LeoFL.CIFAR10.cifar10_raw_dataset import build_cifar10_raw_datasets
from LeoFL.Simulator.fl_core import make_clients_from_dataset, FIXED_CONSTELLATION
from LeoFL.Simulator.run_simulator import run_simulation

from LeoFL.Simulator.analysis_utils import (
    plot_multiple_accuracy_round,
    plot_multiple_accuracy_time,
    plot_cycle_completion_time,
    plot_class_distribution_from_csv,
    plot_meeting_counts_heatmap,
    plot_capacity_from_csv,
    plot_normalized_load,
    plot_multiple_communication,
    plot_global_update_intervals,
)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_fresh_clients(central_train, num_clients, alpha, seed):
    return make_clients_from_dataset(
        dataset=central_train,
        num_clients=num_clients,
        split_mode="non",
        alpha=alpha,
        seed=seed,
    )


def main():
    GLOBAL_SEED = 20240301
    set_global_seed(GLOBAL_SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOG_DIR = "./logs_cifar10"
    os.makedirs(LOG_DIR, exist_ok=True)

    central_train, test_dataset = build_cifar10_raw_datasets(
        root="LeoFL/CIFAR10/data",
    )

    # 3) constellation / clients
    CONSTELLATION_CONFIGS = FIXED_CONSTELLATION
    NUM_CLIENTS = len(CONSTELLATION_CONFIGS)

    rng = np.random.default_rng(GLOBAL_SEED)
    per_sat_minutes = rng.integers(low=5, high=31, size=NUM_CLIENTS).tolist()

    ALPHA = 0.5
    N_ROUNDS = 1000

    # (1) Centralized FL
    client_datasets = make_fresh_clients(central_train, NUM_CLIENTS, ALPHA, GLOBAL_SEED)
    run_simulation(
        mode="always",
        client_datasets=client_datasets,
        test_dataset=test_dataset,
        per_sat_minutes=per_sat_minutes,
        num_rounds=N_ROUNDS,
        agg_name="fedavg",
        target_acc=70.0,
        device=device,
        seconds_per_round=60,
        log_dir=LOG_DIR,
        log_prefix="cifar10_ideal",
        constellation_configs=CONSTELLATION_CONFIGS,
        model_name="cnn",
    )

    # (2) Hybrid-Offload
    client_datasets = make_fresh_clients(central_train, NUM_CLIENTS, ALPHA, GLOBAL_SEED)
    run_simulation(
        mode="offload",
        client_datasets=client_datasets,
        test_dataset=test_dataset,
        per_sat_minutes=per_sat_minutes,
        num_rounds=N_ROUNDS,
        agg_name="fedavg",
        target_acc=70.0,
        device=device,
        seconds_per_round=60,
        log_dir=LOG_DIR,
        log_prefix="cifar10_offload",
        constellation_configs=CONSTELLATION_CONFIGS,
        model_name="cnn",
    )

    # (3) Decentralized FL
    client_datasets = make_fresh_clients(central_train, NUM_CLIENTS, ALPHA, GLOBAL_SEED)
    run_simulation(
        mode="decentral",
        client_datasets=client_datasets,
        test_dataset=test_dataset,
        per_sat_minutes=per_sat_minutes,
        num_rounds=N_ROUNDS,
        agg_name="decentral",
        target_acc=70.0,
        device=device,
        seconds_per_round=60,
        log_dir=LOG_DIR,
        log_prefix="cifar10_decentral",
        constellation_configs=CONSTELLATION_CONFIGS,
        model_name="cnn",
    )

    # ===============================
    # Accuracy Comparison (Round-based)
    # ===============================
    acc_csvs = [
        os.path.join(LOG_DIR, "cifar10_ideal_accuracy.csv"),
        os.path.join(LOG_DIR, "cifar10_offload_accuracy.csv"),
        os.path.join(LOG_DIR, "cifar10_decentral_accuracy.csv"),
    ]

    labels = [
        "CENTRALIZED FL",
        "HYBRID-OFFLOAD",
        "DECENTRALIZED FL",
    ]

    plot_multiple_accuracy_round(
        csv_paths=acc_csvs,
        labels=labels,
        title="CIFAR-10 Accuracy vs Global Update Round",
        save_path=os.path.join(LOG_DIR, "cifar10_accuracy_comparison_round.png"),
        use_step=True,
    )

    # ===============================
    # Accuracy Comparison (Time-based)
    # ===============================
    train_time_csvs = [
        os.path.join(LOG_DIR, "cifar10_ideal_train_time.csv"),
        os.path.join(LOG_DIR, "cifar10_offload_train_time.csv"),
        os.path.join(LOG_DIR, "cifar10_decentral_train_time.csv"),
    ]

    plot_multiple_accuracy_time(
        acc_csv_paths=acc_csvs,
        train_time_csv_paths=train_time_csvs,
        labels=labels,
        time_unit="min",
        title="CIFAR-10 Accuracy vs Time",
        save_path=os.path.join(LOG_DIR, "cifar10_accuracy_comparison_time.png"),
        use_step=True,
    )

    # ===============================
    # Offload-related plots
    # ===============================
    before_csv = os.path.join(LOG_DIR, "cifar10_offload_classdist_before.csv")
    after_csv = os.path.join(LOG_DIR, "cifar10_offload_classdist_after.csv")
    cap_csv = os.path.join(LOG_DIR, "cifar10_offload_capacity.csv")
    meeting_csv = os.path.join(LOG_DIR, "cifar10_offload_meeting_counts.csv")

    if os.path.exists(before_csv):
        plot_class_distribution_from_csv(
            before_csv,
            title="CIFAR-10 Class Distribution Before Offload",
            save_path=before_csv.replace(".csv", ".png"),
        )

    if os.path.exists(after_csv):
        plot_class_distribution_from_csv(
            after_csv,
            title="CIFAR-10 Class Distribution After Offload",
            save_path=after_csv.replace(".csv", ".png"),
        )

    if os.path.exists(meeting_csv):
        plot_meeting_counts_heatmap(
            meeting_csv,
            save_path=meeting_csv.replace(".csv", ".png"),
        )

    if os.path.exists(cap_csv):
        plot_capacity_from_csv(
            cap_csv,
            save_path=cap_csv.replace(".csv", ".png"),
        )

    if (
        os.path.exists(before_csv)
        and os.path.exists(after_csv)
        and os.path.exists(cap_csv)
    ):
        plot_normalized_load(
            before_csv,
            after_csv,
            cap_csv,
            save_path=os.path.join(LOG_DIR, "cifar10_offload_normalized_load.png"),
        )

    # ===============================
    # Communication plots
    # ===============================
    comm_csvs = [
        os.path.join(LOG_DIR, "cifar10_ideal_comm.csv"),
        os.path.join(LOG_DIR, "cifar10_offload_comm.csv"),
        os.path.join(LOG_DIR, "cifar10_decentral_comm.csv"),
    ]

    plot_multiple_communication(
        csv_paths=comm_csvs,
        labels=labels,
        which="model_hops",
        mode="per_round",
        max_x=N_ROUNDS,
        seconds_per_round=60,
        time_unit="min",
        title="CIFAR-10 Model Communication Over Time",
        save_path=os.path.join(LOG_DIR, "cifar10_comm_model_hops_comparison.png"),
    )

    plot_multiple_communication(
        csv_paths=comm_csvs,
        labels=labels,
        which="model_hops",
        mode="cumulative",
        max_x=N_ROUNDS,
        seconds_per_round=60,
        time_unit="min",
        title="CIFAR-10 Cumulative Communication Over Time",
        save_path=os.path.join(LOG_DIR, "cifar10_comm_model_hops_cumulative.png"),
    )

    # ===============================
    # Global update interval / cycle time
    # ===============================
    plot_global_update_intervals(
        csv_paths=acc_csvs,
        labels=labels,
        seconds_per_round=60,
        time_unit="min",
        title="CIFAR-10 Cycle Completion Time per Global Update",
        save_path=os.path.join(LOG_DIR, "cifar10_global_update_intervals.png"),
    )

    plot_cycle_completion_time(
        train_time_csv_paths=train_time_csvs,
        labels=labels,
        time_unit="min",
        title="CIFAR-10 Cycle Completion Time",
        save_path=os.path.join(LOG_DIR, "cifar10_cycle_completion_time.png"),
    )


if __name__ == "__main__":
    main()
