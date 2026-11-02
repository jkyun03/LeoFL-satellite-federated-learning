# LeoFL/EuroSAT/simulation_eurosat.py
# EuroSAT 메인 실험 진입점: Starlink-like 위성군에서 FedAvg·LH-FL·FedLEO·FedISL 등 시나리오를 multi-seed로 실행하고 정확도·통신량·라운드 완료시간 그래프를 생성한다.
import os
import random
import math
import numpy as np
import torch
import pandas as pd
import matplotlib as mpl
from torch.utils.data import DataLoader

from LeoFL.EuroSAT import build_eurosat_embeddings, EmbedTabularDataset
from LeoFL.Simulator.fl_core import (
    make_clients_from_dataset,
    FIXED_CONSTELLATION,
    make_starlink_shell_active10_constellation,
)
from LeoFL.Simulator.run_simulator import run_simulation
from LeoFL.Simulator.analysis_utils import (
    plot_multiple_accuracy_time,
    plot_cycle_completion_time,
    plot_class_distribution_from_csv,
    plot_meeting_counts_heatmap,
    plot_capacity_from_csv,
    plot_normalized_load,
    plot_multiple_communication,
    plot_global_update_intervals,
    plot_multiple_accuracy_sim_time,
    plot_multiple_accuracy_update_round,
    plot_multiple_accuracy_vs_cumulative_comm,
    plot_comm_cost_to_target_bar,
    plot_update_completion_time_from_accuracy,
    plot_update_completion_summary_from_accuracy,
    plot_update_completion_reduction_bar,
    plot_normalized_update_completion_time,
    plot_update_completion_dumbbell,
    plot_update_completion_saved_segment_bar,
    make_comm_cost_to_target_multiseed,
    plot_comm_cost_to_target_multiseed,
    make_round_completion_multiseed,
    plot_round_completion_multiseed,
    make_comm_cost_components_to_target_multiseed,
    plot_comm_cost_components_to_target_multiseed,
    make_accuracy_round_multiseed,
    plot_accuracy_round_multiseed,
)

from LeoFL.Simulator.scenarios_contact import run_contact_aware_lhfl

from LeoFL.Simulator.scenarios_fedisl_ladder import run_fedisl_capacity

from LeoFL.Simulator.scenarios_fedisl_mixing import run_fedisl_leader_mixing


def set_paper_plot_style():
    mpl.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 19,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 16,
            "figure.titlesize": 20,
            "axes.linewidth": 1.4,
            "lines.linewidth": 2.8,
            "lines.markersize": 6,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_train_test(full_ds: EmbedTabularDataset, train_ratio=0.8, seed=20240301):
    rng = np.random.default_rng(seed)
    N = len(full_ds)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_train = int(train_ratio * N)
    tr = idx[:n_train]
    te = idx[n_train:]

    train_ds = EmbedTabularDataset(
        full_ds.X[tr].clone(),
        full_ds.Y[tr].clone(),
        n_classes=full_ds.n_classes,
        class_weight=None,  # 아래에서 train 기준으로 다시 계산
    )
    test_ds = EmbedTabularDataset(
        full_ds.X[te].clone(),
        full_ds.Y[te].clone(),
        n_classes=full_ds.n_classes,
        class_weight=None,
    )

    from LeoFL.EuroSAT.eurosat_embed_dataset import _compute_balanced_class_weight

    train_ds.class_weight = _compute_balanced_class_weight(
        train_ds.Y, train_ds.n_classes
    )
    test_ds.class_weight = train_ds.class_weight.clone()

    return train_ds, test_ds


class PartialFedAvgMLP(torch.nn.Module):
    """
    EuroSAT ResNet18 embedding [512-d]를 입력으로 받는 3-layer DNN classifier.
    fl_core.DNN과 동일한 구조(hidden1=hidden2=hidden_size)로 맞춘 lightweight MLP.
    """

    def __init__(self, n_features: int, n_classes: int, hidden_size: int = 100):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_features, hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


@torch.no_grad()
def evaluate_partial_model(model, test_dataset, device, batch_size: int = 512):
    model.eval()
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0

    for X, Y in loader:
        X = X.to(device)
        Y = Y.to(device)
        logits = model(X)
        pred = logits.argmax(dim=1)
        correct += (pred == Y).sum().item()
        total += Y.numel()

    return 100.0 * correct / max(total, 1)


def train_partial_client(
    global_state,
    client_dataset,
    n_features: int,
    n_classes: int,
    hidden_size: int,
    device,
    lr: float = 1e-3,
    batch_size: int = 32,
    local_epochs: int = 1,
):
    """
    global model state에서 시작해서 client 하나를 local training.
    반환: trained local model state_dict
    """
    model = PartialFedAvgMLP(
        n_features=n_features,
        n_classes=n_classes,
        hidden_size=hidden_size,
    ).to(device)

    model.load_state_dict(global_state)
    model.train()

    loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)

    class_weight = getattr(client_dataset, "class_weight", None)
    if class_weight is not None:
        class_weight = class_weight.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(local_epochs):
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(X), Y)
            loss.backward()
            optimizer.step()

    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fedavg_state_dicts(local_states, weights):
    """
    local state_dict들을 sample 수 기준 weighted average.
    """
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Total FedAvg weight must be positive.")

    avg_state = {}

    for key in local_states[0].keys():
        avg_state[key] = sum(
            state[key].float() * (w / total) for state, w in zip(local_states, weights)
        )

    return avg_state


def run_partial_fedavg(
    client_datasets,
    test_dataset,
    per_sat_minutes,
    rho: float,
    num_rounds: int,
    log_dir: str,
    log_prefix: str,
    device,
    hidden_size: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    local_epochs: int = 1,
    seed: int = 20240301,
):
    """
    Straggler sensitivity analysis용 partial FedAvg baseline.

    rho=1.0  -> 모든 client update를 기다림
    rho=0.75 -> 가장 빨리 끝난 75% client update만 aggregation
    rho=0.50 -> 가장 빨리 끝난 50% client update만 aggregation

    주의:
    - train_time csv에는 aggregation에 실제 참여한 client만 기록함.
    - 따라서 기존 build_accuracy_time_dataframe()에서 round별 max(train_seconds)를 계산하면
      partial aggregation의 effective cycle completion time이 반영됨.
    """
    os.makedirs(log_dir, exist_ok=True)

    rng = np.random.default_rng(seed + int(rho * 1000))

    num_clients = len(client_datasets)
    k_required = int(math.ceil(rho * num_clients))
    k_required = max(1, min(k_required, num_clients))

    n_features = int(test_dataset.n_features)
    n_classes = int(test_dataset.n_classes)

    global_model = PartialFedAvgMLP(
        n_features=n_features,
        n_classes=n_classes,
        hidden_size=hidden_size,
    ).to(device)

    # 초기 accuracy도 기록
    acc_logs = []
    train_time_logs = []

    init_acc = evaluate_partial_model(global_model, test_dataset, device)
    acc_logs.append(
        {
            "round": 0,
            "accuracy": init_acc,
            "rho": rho,
            "num_selected": 0,
        }
    )

    base_train_seconds = np.array(per_sat_minutes, dtype=np.float32) * 60.0

    for r in range(1, num_rounds + 1):
        # round별 training time에 약간의 jitter를 줘서 매번 완전히 같은 client만 선택되지 않도록 함
        jitter = rng.uniform(0.90, 1.10, size=num_clients)
        train_seconds = base_train_seconds * jitter

        # 가장 빨리 끝나는 k_required개 client만 aggregation에 참여
        selected = np.argsort(train_seconds)[:k_required]

        global_state = {
            k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()
        }

        local_states = []
        weights = []

        for sat_id in selected:
            ds = client_datasets[int(sat_id)]
            if len(ds) == 0:
                continue

            local_state = train_partial_client(
                global_state=global_state,
                client_dataset=ds,
                n_features=n_features,
                n_classes=n_classes,
                hidden_size=hidden_size,
                device=device,
                lr=lr,
                batch_size=batch_size,
                local_epochs=local_epochs,
            )

            local_states.append(local_state)
            weights.append(len(ds))

            train_time_logs.append(
                {
                    "round": r,
                    "sat_id": int(sat_id),
                    "train_seconds": float(train_seconds[int(sat_id)]),
                    "rho": rho,
                    "selected": 1,
                }
            )

        if len(local_states) == 0:
            print(f"[WARN][PARTIAL] round={r}: no selected clients with data.")
            continue

        new_state = fedavg_state_dicts(local_states, weights)
        global_model.load_state_dict(new_state)

        acc = evaluate_partial_model(global_model, test_dataset, device)
        acc_logs.append(
            {
                "round": r,
                "accuracy": acc,
                "rho": rho,
                "num_selected": len(local_states),
            }
        )

        if r % 10 == 0 or r == 1:
            print(
                f"[PARTIAL-FEDAVG] rho={rho:.2f} "
                f"round={r}/{num_rounds} "
                f"selected={len(local_states)}/{num_clients} "
                f"acc={acc:.2f}"
            )

    acc_path = os.path.join(log_dir, f"{log_prefix}_accuracy.csv")
    tt_path = os.path.join(log_dir, f"{log_prefix}_train_time.csv")

    pd.DataFrame(acc_logs).to_csv(acc_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(train_time_logs).to_csv(tt_path, index=False, encoding="utf-8-sig")

    print(f"[SAVE][PARTIAL-ACC] {acc_path}")
    print(f"[SAVE][PARTIAL-TIME] {tt_path}")

    return acc_path, tt_path


def plot_accuracy_time_multiseed_from_logs(
    log_dir,
    seeds,
    method_prefix_map,
    labels,
    seconds_per_round=60,
    time_unit="min",
    max_time_min=300,
    save_path=None,
):
    """
    기존 accuracy.csv 로그만 읽어서 10-seed 평균 accuracy-time curve 생성.
    실험 재실행 없음.
    """
    import matplotlib.pyplot as plt

    os.makedirs(log_dir, exist_ok=True)

    # 1분 단위 grid
    grid = np.arange(0, max_time_min + 1, 1, dtype=float)

    plt.figure(figsize=(8.5, 5.2))

    style_map = {
        "FedAvg": {
            "color": "#4C78A8",
            "marker": "o",
            "linestyle": "-",
        },
        "FedISL": {
            "color": "#72B7B2",
            "marker": "s",
            "linestyle": "--",
        },
        "LH-FL": {
            "color": "#E45756",
            "marker": "D",
            "linestyle": "-",
        },
    }

    for label in labels:
        curves = []

        for run_idx, _seed in enumerate(seeds, start=1):
            prefix = method_prefix_map[label].format(idx=run_idx)
            acc_path = os.path.join(log_dir, f"{prefix}_accuracy.csv")

            if not os.path.exists(acc_path):
                print(f"[WARN][ACC-TIME] missing: {acc_path}")
                continue

            df = pd.read_csv(acc_path)

            if "round" not in df.columns or "accuracy" not in df.columns:
                print(f"[WARN][ACC-TIME] invalid columns: {acc_path}")
                continue

            df = df.sort_values("round")
            t_min = df["round"].to_numpy(dtype=float) * seconds_per_round / 60.0
            acc = df["accuracy"].to_numpy(dtype=float)

            # 중복 time 제거
            uniq_t, uniq_idx = np.unique(t_min, return_index=True)
            uniq_acc = acc[uniq_idx]

            # grid에 보간
            acc_interp = np.interp(
                grid,
                uniq_t,
                uniq_acc,
                left=uniq_acc[0],
                right=uniq_acc[-1],
            )
            curves.append(acc_interp)

        if len(curves) == 0:
            continue

        curves = np.vstack(curves)
        mean_acc = curves.mean(axis=0)
        std_acc = curves.std(axis=0)

        st = style_map.get(label, {})
        color = st.get("color", None)
        marker = st.get("marker", None)
        linestyle = st.get("linestyle", "-")

        (line,) = plt.plot(
            grid,
            mean_acc,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=3.0,
            marker=marker,
            markevery=25,
            markersize=6.5,
        )

        plt.fill_between(
            grid,
            mean_acc - std_acc,
            mean_acc + std_acc,
            alpha=0.12,
            color=line.get_color(),
            linewidth=0,
        )

    plt.xlabel(f"Time ({time_unit})")
    plt.ylabel("Accuracy (%)")
    plt.xlim(0, max_time_min)
    plt.ylim(0, 100)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(
            log_dir,
            "starlink10p_accuracy_time_multiseed_10runs_nofedleo.png",
        )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVE][ACC-TIME] {save_path}")


def plot_starlink10p_load_balancing_from_logs(
    log_dir,
    prefix="starlink10p_seed1_lhfl",
):
    """
    기존 LH-FL 로그만 읽어서 load balancing figure 생성.
    실험 재실행 없음.
    """
    before_csv = os.path.join(log_dir, f"{prefix}_classdist_before.csv")
    after_csv = os.path.join(log_dir, f"{prefix}_classdist_after.csv")
    cap_csv = os.path.join(log_dir, f"{prefix}_capacity.csv")

    missing = [p for p in [before_csv, after_csv, cap_csv] if not os.path.exists(p)]

    if len(missing) > 0:
        print("[WARN][LOAD] missing files:")
        for p in missing:
            print(f"  {p}")
        return

    save_path = os.path.join(
        log_dir,
        "starlink10p_lhfl_load_balancing_seed1.png",
    )

    plot_normalized_load(
        before_csv,
        after_csv,
        cap_csv,
        save_path=save_path,
    )

    print(f"[SAVE][LOAD] {save_path}")


def main():
    GLOBAL_SEED = 20240301
    set_global_seed(GLOBAL_SEED)
    set_paper_plot_style()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 3-layer classifier(fc1-100-100-10)로 구조가 바뀌면서 기존 2-layer 로그와
    # 호환되지 않으므로, 별도 폴더에 격리해서 기존 logs_eurosat/를 보존한다.
    LOG_DIR = "./logs_eurosat_3layer"
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1) EuroSAT -> Embedding dataset (캐시 있으면 즉시 로드)
    full_ds = build_eurosat_embeddings(
        root="LeoFL/EuroSAT/data",
        cache_path="LeoFL/EuroSAT/cache/eurosat_resnet18_512.npz",
        batch_size=256,
        num_workers=4,
        device=device,
    )

    # 2) train/test split
    central_train, test_dataset = split_train_test(
        full_ds, train_ratio=0.8, seed=GLOBAL_SEED
    )

    # 3) constellation / clients
    # Starlink-like full shell parameter를 기반으로 전체 shell을 만들고,
    # 그중 10개 orbital plane만 active FL participants로 사용한다.
    USE_STARLINK_ACTIVE_10P = True

    if USE_STARLINK_ACTIVE_10P:
        CONSTELLATION_CONFIGS, FULL_CONSTELLATION_CONFIGS, ACTIVE_ORIG_PLANE_IDS = (
            make_starlink_shell_active10_constellation(
                full_num_planes=72,
                sats_per_plane=22,
                active_num_planes=10,
                h_km=552.0,
                incl_deg=53.0,
                phase_factor=0,
            )
        )
    else:
        CONSTELLATION_CONFIGS = FIXED_CONSTELLATION
        FULL_CONSTELLATION_CONFIGS = FIXED_CONSTELLATION
        ACTIVE_ORIG_PLANE_IDS = None

    NUM_CLIENTS = len(CONSTELLATION_CONFIGS)

    print(f"[INFO] NUM_CLIENTS = {NUM_CLIENTS}")
    print(
        f"[INFO] active constellation planes = {sorted(set(c['plane_id'] for c in CONSTELLATION_CONFIGS))}"
    )

    if ACTIVE_ORIG_PLANE_IDS is not None:
        print(
            f"[INFO] active original Starlink-like plane ids = {ACTIVE_ORIG_PLANE_IDS}"
        )

    # 위성별 학습시간(분) - 기존 로직 그대로
    rng = np.random.default_rng(GLOBAL_SEED)
    per_sat_minutes = rng.integers(low=5, high=31, size=NUM_CLIENTS).tolist()

    # 4) non-IID split
    ALPHA = 0.5
    client_datasets = make_clients_from_dataset(
        dataset=central_train,
        num_clients=NUM_CLIENTS,
        split_mode="non",
        alpha=ALPHA,
        seed=GLOBAL_SEED,
    )

    # 5) 시뮬레이션 실행
    N_ROUNDS = 2000

    RUN_EXISTING_SIMULATION = False
    RUN_PARTIAL_FEDAVG = False
    RUN_FEDISL_EXTRA = False

    # 최종 Fig.1, Fig.4용 10회 반복 실험만 실행
    RUN_MULTI_SEED_FINAL = True
    RUN_MULTI_SEED_COMM = False

    PLOT_ONLY_FROM_EXISTING_LOGS = False
    PLOT_EXTRA_STARLINK10P_FIGS = True

    MULTI_SEEDS = [
        20240301,
        20240302,
        20240303,
        20240304,
        20240305,
        20240306,
        20240307,
        20240308,
        20240309,
        20240310,
    ]

    MULTI_N_ROUNDS = 1000

    if RUN_EXISTING_SIMULATION:
        # (1) Centralized upper bound (always)
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
            log_prefix="eurosat_ideal",
            constellation_configs=CONSTELLATION_CONFIGS,
        )

        # (2) Hybrid-Offload
        # fresh split for fairness
        client_datasets = make_clients_from_dataset(
            dataset=central_train,
            num_clients=NUM_CLIENTS,
            split_mode="non",
            alpha=ALPHA,
            seed=GLOBAL_SEED,
        )
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
            log_prefix="eurosat_offload",
            constellation_configs=CONSTELLATION_CONFIGS,
        )

        # (3) Fully decentralized
        client_datasets = make_clients_from_dataset(
            dataset=central_train,
            num_clients=NUM_CLIENTS,
            split_mode="non",
            alpha=ALPHA,
            seed=GLOBAL_SEED,
        )
        run_simulation(
            mode="decentral",
            client_datasets=client_datasets,
            test_dataset=test_dataset,
            per_sat_minutes=per_sat_minutes,
            num_rounds=N_ROUNDS,
            agg_name="fedavg",
            target_acc=70.0,
            device=device,
            seconds_per_round=60,
            log_dir=LOG_DIR,
            log_prefix="eurosat_decentral",
            constellation_configs=CONSTELLATION_CONFIGS,
        )

        # (4) FedISL baseline
        client_datasets = make_clients_from_dataset(
            dataset=central_train,
            num_clients=NUM_CLIENTS,
            split_mode="non",
            alpha=ALPHA,
            seed=GLOBAL_SEED,
        )

        run_simulation(
            mode="fedisl",
            client_datasets=client_datasets,
            test_dataset=test_dataset,
            per_sat_minutes=per_sat_minutes,
            num_rounds=N_ROUNDS,
            agg_name="fedavg",
            target_acc=70.0,
            device=device,
            seconds_per_round=60,
            log_dir=LOG_DIR,
            log_prefix="eurosat_fedisl",
            constellation_configs=CONSTELLATION_CONFIGS,
        )

    # ===============================
    # Additional FedISL comparison
    # ===============================
    if RUN_FEDISL_EXTRA:
        fedisl_acc_path = os.path.join(LOG_DIR, "eurosat_fedisl_accuracy.csv")

        if os.path.exists(fedisl_acc_path):
            print(f"[SKIP][FedISL] existing log found: {fedisl_acc_path}")
        else:
            print("[RUN][FedISL] additional comparison experiment")

            fedisl_client_datasets = make_clients_from_dataset(
                dataset=central_train,
                num_clients=NUM_CLIENTS,
                split_mode="non",
                alpha=ALPHA,
                seed=GLOBAL_SEED,
            )

            run_simulation(
                mode="fedisl",
                client_datasets=fedisl_client_datasets,
                test_dataset=test_dataset,
                per_sat_minutes=per_sat_minutes,
                num_rounds=N_ROUNDS,
                agg_name="fedavg",
                target_acc=70.0,
                device=device,
                seconds_per_round=60,
                log_dir=LOG_DIR,
                log_prefix="eurosat_fedisl",
                constellation_configs=CONSTELLATION_CONFIGS,
            )

        # ===============================
    # Multi-seed final experiment for Fig.1 and Fig.4
    # ===============================
    if RUN_MULTI_SEED_FINAL:
        print("\n[RUN][MULTI-SEED FINAL] Fig.1 and Fig.4")

        def run_one_if_missing(mode, prefix, seed, run_per_sat_minutes):
            acc_path = os.path.join(LOG_DIR, f"{prefix}_accuracy.csv")
            comm_path = os.path.join(LOG_DIR, f"{prefix}_comm.csv")

            if os.path.exists(acc_path) and os.path.exists(comm_path):
                print(f"[SKIP][MULTI-SEED] existing logs found: {prefix}")
                return

            # plot-only 모드에서는 missing log가 있어도 실험을 다시 돌리지 않음
            if PLOT_ONLY_FROM_EXISTING_LOGS:
                print(f"[PLOT-ONLY][WARN] missing logs, skip simulation: {prefix}")
                print(f"  acc exists?  {os.path.exists(acc_path)} -> {acc_path}")
                print(f"  comm exists? {os.path.exists(comm_path)} -> {comm_path}")
                return

            print(f"[RUN][MULTI-SEED] mode={mode}, seed={seed}, prefix={prefix}")

            set_global_seed(seed)

            seeded_clients = make_clients_from_dataset(
                dataset=central_train,
                num_clients=NUM_CLIENTS,
                split_mode="non",
                alpha=ALPHA,
                seed=seed,
            )

            run_simulation(
                mode=mode,
                client_datasets=seeded_clients,
                test_dataset=test_dataset,
                per_sat_minutes=run_per_sat_minutes,
                num_rounds=MULTI_N_ROUNDS,
                agg_name="fedavg",
                target_acc=85.0,
                device=device,
                seconds_per_round=60,
                log_dir=LOG_DIR,
                log_prefix=prefix,
                constellation_configs=CONSTELLATION_CONFIGS,
            )

        def run_lhfl_b_if_missing(prefix, seed, run_per_sat_minutes, do_offload=True):
            acc_path = os.path.join(LOG_DIR, f"{prefix}_accuracy.csv")
            comm_path = os.path.join(LOG_DIR, f"{prefix}_comm.csv")
            if os.path.exists(acc_path) and os.path.exists(comm_path):
                print(f"[SKIP][LHFL-B] {prefix}")
                return
            if PLOT_ONLY_FROM_EXISTING_LOGS:
                print(f"[PLOT-ONLY] skip {prefix}")
                return
            print(
                f"[RUN][LHFL-B] seed={seed}, prefix={prefix}, do_offload={do_offload}"
            )
            set_global_seed(seed)
            seeded = make_clients_from_dataset(
                dataset=central_train,
                num_clients=NUM_CLIENTS,
                split_mode="non",
                alpha=ALPHA,
                seed=seed,
            )
            run_contact_aware_lhfl(
                client_datasets=seeded,
                test_dataset=test_dataset,
                per_sat_minutes=run_per_sat_minutes,
                num_rounds=MULTI_N_ROUNDS,
                constellation_configs=CONSTELLATION_CONFIGS,
                device=device,
                log_dir=LOG_DIR,
                log_prefix=prefix,
                inter_plane_mode="mixing",
                use_real_contacts=True,
                max_dist_km=5000.0,
                target_acc=85.0,
                seconds_per_round=60,
                do_offload=do_offload,
            )

        def run_decentral_if_missing(prefix, seed, run_per_sat_minutes, do_offload):
            acc_path = os.path.join(LOG_DIR, f"{prefix}_accuracy.csv")
            comm_path = os.path.join(LOG_DIR, f"{prefix}_comm.csv")
            if os.path.exists(acc_path) and os.path.exists(comm_path):
                print(f"[SKIP][DECENTRAL] {prefix}")
                return
            if PLOT_ONLY_FROM_EXISTING_LOGS:
                print(f"[PLOT-ONLY] skip {prefix}")
                return
            print(
                f"[RUN][DECENTRAL] seed={seed}, prefix={prefix}, do_offload={do_offload}"
            )
            set_global_seed(seed)
            seeded = make_clients_from_dataset(
                dataset=central_train,
                num_clients=NUM_CLIENTS,
                split_mode="non",
                alpha=ALPHA,
                seed=seed,
            )
            run_simulation(
                mode="decentral",
                client_datasets=seeded,
                test_dataset=test_dataset,
                per_sat_minutes=run_per_sat_minutes,
                num_rounds=MULTI_N_ROUNDS,
                agg_name="fedavg",
                target_acc=85.0,
                device=device,
                seconds_per_round=60,
                log_dir=LOG_DIR,
                log_prefix=prefix,
                constellation_configs=CONSTELLATION_CONFIGS,
                # LH-FL의 ISL 사거리(5000km)와 통일 — 사다리 4칸이 같은 접촉 그래프를
                # 공유해야 리더 계층/offload 효과만 격리해서 비교할 수 있음.
                decentral_max_dist_km=5000.0,
                decentral_do_offload=do_offload,
                decentral_offload_ratio=0.1,
                decentral_use_capacity_time_model=True,
            )

        def run_fedisl_cap_if_missing(prefix, seed, run_per_sat_minutes, do_offload):
            acc_path = os.path.join(LOG_DIR, f"{prefix}_accuracy.csv")
            comm_path = os.path.join(LOG_DIR, f"{prefix}_comm.csv")
            if os.path.exists(acc_path) and os.path.exists(comm_path):
                print(f"[SKIP][FedISL-CAP] {prefix}")
                return
            if PLOT_ONLY_FROM_EXISTING_LOGS:
                print(f"[PLOT-ONLY] skip {prefix}")
                return
            print(
                f"[RUN][FedISL-CAP] seed={seed}, prefix={prefix}, do_offload={do_offload}"
            )
            set_global_seed(seed)
            seeded = make_clients_from_dataset(
                dataset=central_train,
                num_clients=NUM_CLIENTS,
                split_mode="non",
                alpha=ALPHA,
                seed=seed,
            )
            run_fedisl_capacity(
                client_datasets=seeded,
                test_dataset=test_dataset,
                per_sat_minutes=run_per_sat_minutes,
                num_rounds=MULTI_N_ROUNDS,
                constellation_configs=CONSTELLATION_CONFIGS,
                device=device,
                log_dir=LOG_DIR,
                log_prefix=prefix,
                do_offload=do_offload,
                offload_ratio=0.1,
                target_acc=85.0,
                seconds_per_round=60,
            )

        def run_fedisl_mix_if_missing(prefix, seed, run_per_sat_minutes, do_offload):
            acc_path = os.path.join(LOG_DIR, f"{prefix}_accuracy.csv")
            comm_path = os.path.join(LOG_DIR, f"{prefix}_comm.csv")
            if os.path.exists(acc_path) and os.path.exists(comm_path):
                print(f"[SKIP][FedISL-MIX] {prefix}")
                return
            if PLOT_ONLY_FROM_EXISTING_LOGS:
                print(f"[PLOT-ONLY] skip {prefix}")
                return
            print(
                f"[RUN][FedISL-MIX] seed={seed}, prefix={prefix}, do_offload={do_offload}"
            )
            set_global_seed(seed)
            seeded = make_clients_from_dataset(
                dataset=central_train,
                num_clients=NUM_CLIENTS,
                split_mode="non",
                alpha=ALPHA,
                seed=seed,
            )
            run_fedisl_leader_mixing(
                client_datasets=seeded,
                test_dataset=test_dataset,
                per_sat_minutes=run_per_sat_minutes,
                num_rounds=MULTI_N_ROUNDS,
                constellation_configs=CONSTELLATION_CONFIGS,
                device=device,
                log_dir=LOG_DIR,
                log_prefix=prefix,
                do_offload=do_offload,
                offload_ratio=0.1,
                max_dist_km=5000.0,
                leader_mixing_beta=0.5,
                target_acc=85.0,
                seconds_per_round=60,
            )

        def run_hier_central_if_missing(prefix, seed, run_per_sat_minutes):
            acc_path = os.path.join(LOG_DIR, f"{prefix}_accuracy.csv")
            comm_path = os.path.join(LOG_DIR, f"{prefix}_comm.csv")
            if os.path.exists(acc_path) and os.path.exists(comm_path):
                print(f"[SKIP][HIER-CENTRAL] existing logs: {prefix}")
                return
            if PLOT_ONLY_FROM_EXISTING_LOGS:
                print(f"[PLOT-ONLY][WARN] missing logs, skip: {prefix}")
                return
            print(f"[RUN][HIER-CENTRAL] seed={seed}, prefix={prefix}")
            set_global_seed(seed)
            seeded_clients = make_clients_from_dataset(
                dataset=central_train,
                num_clients=NUM_CLIENTS,
                split_mode="non",
                alpha=ALPHA,
                seed=seed,
            )
            run_contact_aware_lhfl(
                client_datasets=seeded_clients,
                test_dataset=test_dataset,
                per_sat_minutes=run_per_sat_minutes,
                num_rounds=MULTI_N_ROUNDS,
                constellation_configs=CONSTELLATION_CONFIGS,
                device=device,
                log_dir=LOG_DIR,
                log_prefix=prefix,
                inter_plane_mode="central",  # B2: 이상적 계층 중앙집계
                use_real_contacts=True,
                max_dist_km=5000.0,
                target_acc=85.0,
                seconds_per_round=60,
            )

        # ==================================================
        # 1) 먼저 seed별 실험 실행
        # ==================================================
        for run_idx, seed in enumerate(MULTI_SEEDS, start=1):
            print(
                f"\n========== MULTI-SEED {run_idx}/{len(MULTI_SEEDS)} | seed={seed} =========="
            )

            # 데이터 분포는 공유(공정성) — make_clients_from_dataset(seed=seed) 그대로
            # 학습시간 분포 범위(5~30분)도 공유하되, 각 방법이 뽑는 실제 값은 독립
            def draw_psm(sub):
                return (
                    np.random.default_rng(seed + sub)
                    .integers(low=5, high=31, size=NUM_CLIENTS)
                    .tolist()
                )

            # 메인 3-way는 기존대로 공유 (헤드라인 유지).
            # sub-seed 400004: 지상국 없는 4칸 사다리의 LH-FL(=lhfl_b5000 재사용)과
            # 동일한 draw로 통일 — 사다리와 헤드라인 어디서 봐도 LH-FL 숫자가 같도록.
            run_per_sat_minutes = draw_psm(400004)

            # 사다리 4개 rung: 각각 독립 draw (동일 분포, 다른 실현값)
            psm_cap = draw_psm(200001)  # FedISL
            psm_mix = draw_psm(200002)  # FedISL+decentral
            psm_cap_off = draw_psm(200003)  # FedISL+offload
            psm_mix_off = draw_psm(200004)  # FedISL+decentral+off

            # 지상국 없는(all-decentral) 4칸 사다리: 각 rung 독립 draw.
            # LH-FL(4번 칸)은 별도로 돌리지 않고 헤드라인 3-way의 lhfl_b5000 결과를
            # 그대로 재사용한다 — 같은 보고서에서 LH-FL 수치가 두 개로 갈리는 걸 방지.
            psm_decentral = draw_psm(400001)  # Decentral
            psm_decentral_leader = draw_psm(400002)  # Decentral+Leader
            psm_decentral_off = draw_psm(400003)  # Decentral+Offload

            # --- 메인 3-way (공유 per_sat_minutes) ---
            run_one_if_missing(
                mode="always",
                prefix=f"starlink10p_seed{run_idx}_fedavg",
                seed=seed,
                run_per_sat_minutes=run_per_sat_minutes,
            )
            run_one_if_missing(
                mode="fedisl",
                prefix=f"starlink10p_seed{run_idx}_fedisl",
                seed=seed,
                run_per_sat_minutes=run_per_sat_minutes,
            )
            run_lhfl_b_if_missing(
                prefix=f"starlink10p_seed{run_idx}_lhfl_b5000",
                seed=seed,
                run_per_sat_minutes=run_per_sat_minutes,
            )

            # --- 사다리 (각 rung 독립 per_sat_minutes) ---
            run_fedisl_cap_if_missing(
                prefix=f"starlink10p_seed{run_idx}_fedisl_cap",
                seed=seed,
                run_per_sat_minutes=psm_cap,
                do_offload=False,
            )
            run_fedisl_mix_if_missing(
                prefix=f"starlink10p_seed{run_idx}_fedisl_mix",
                seed=seed,
                run_per_sat_minutes=psm_mix,
                do_offload=False,
            )
            run_fedisl_cap_if_missing(
                prefix=f"starlink10p_seed{run_idx}_fedisl_cap_off",
                seed=seed,
                run_per_sat_minutes=psm_cap_off,
                do_offload=True,
            )
            run_fedisl_mix_if_missing(
                prefix=f"starlink10p_seed{run_idx}_fedisl_mix_off",
                seed=seed,
                run_per_sat_minutes=psm_mix_off,
                do_offload=True,
            )

            # --- 지상국 없는 4칸 사다리: Decentral / +Leader / +Offload / LH-FL ---
            run_decentral_if_missing(
                prefix=f"starlink10p_seed{run_idx}_decentral",
                seed=seed,
                run_per_sat_minutes=psm_decentral,
                do_offload=False,
            )
            run_lhfl_b_if_missing(
                prefix=f"starlink10p_seed{run_idx}_decentral_leader",
                seed=seed,
                run_per_sat_minutes=psm_decentral_leader,
                do_offload=False,
            )
            run_decentral_if_missing(
                prefix=f"starlink10p_seed{run_idx}_decentral_off",
                seed=seed,
                run_per_sat_minutes=psm_decentral_off,
                do_offload=True,
            )
            # LH-FL: 헤드라인 starlink10p_seed{run_idx}_lhfl_b5000을 그대로 재사용 (위에서 이미 실행됨)

        # ==================================================
        # 2) 실험 파일 생성 후 summary/plot 생성
        # ==================================================
        method_prefix_map = {
            "FedAvg": "starlink10p_seed{idx}_fedavg",
            "FedISL": "starlink10p_seed{idx}_fedisl",
            "LH-FL": "starlink10p_seed{idx}_lhfl_b5000",
        }

        multiseed_labels = ["FedAvg", "FedISL", "LH-FL"]

        # -------------------------------
        # Fig.1: round completion time
        # run-level mean -> 10-run mean ± std
        # -------------------------------
        round_raw_df, round_summary_df = make_round_completion_multiseed(
            log_dir=LOG_DIR,
            seeds=MULTI_SEEDS,
            method_prefix_map=method_prefix_map,
            labels=multiseed_labels,
            seconds_per_round=60,
            time_unit="min",
            save_csv_path=os.path.join(
                LOG_DIR,
                "table_round_completion_time_multiseed_10runs.csv",
            ),
        )

        print("[ROUND MULTI-SEED RAW]")
        print(round_raw_df)

        print("[ROUND MULTI-SEED SUMMARY]")
        print(round_summary_df)

        plot_round_completion_multiseed(
            summary_df=round_summary_df,
            time_unit="min",
            save_path=os.path.join(
                LOG_DIR,
                "eurosat_round_completion_time_multiseed_10runs.png",
            ),
        )

        # -------------------------------
        # Fig.4: cumulative communication cost through round MULTI_N_ROUNDS
        # (target accuracy 도달 여부와 무관 — 85%에 도달 못하는 방법도 비교 가능)
        # stacked: model communication + one-shot offloading overhead
        # -------------------------------
        comm_raw_df, comm_summary_df = make_comm_cost_components_to_target_multiseed(
            log_dir=LOG_DIR,
            seeds=MULTI_SEEDS,
            method_prefix_map=method_prefix_map,
            labels=multiseed_labels,
            final_round=True,
            save_csv_path=os.path.join(
                LOG_DIR,
                f"table_comm_cost_cum{MULTI_N_ROUNDS}rounds_multiseed_10runs_components.csv",
            ),
        )

        print("[COMM COMPONENT MULTI-SEED RAW]")
        print(comm_raw_df)

        print("[COMM COMPONENT MULTI-SEED SUMMARY]")
        print(comm_summary_df)

        plot_comm_cost_components_to_target_multiseed(
            summary_df=comm_summary_df,
            xlabel=f"Cumulative communication cost through round {MULTI_N_ROUNDS}",
            save_path=os.path.join(
                LOG_DIR,
                f"eurosat_comm_cost_cum{MULTI_N_ROUNDS}rounds_multiseed_10runs_components.png",
            ),
        )

        # -------------------------------
        # Fig.4-B: byte-level cumulative communication cost through round MULTI_N_ROUNDS
        # model_bytes + offload_bytes 기준
        # -------------------------------
        comm_byte_raw_df, comm_byte_summary_df = make_comm_cost_to_target_multiseed(
            log_dir=LOG_DIR,
            seeds=MULTI_SEEDS,
            method_prefix_map=method_prefix_map,
            labels=multiseed_labels,
            comm_col="total_mb",
            final_round=True,
            save_csv_path=os.path.join(
                LOG_DIR,
                f"table_comm_cost_cum{MULTI_N_ROUNDS}rounds_multiseed_10runs_mb.csv",
            ),
        )

        print("[COMM BYTE MULTI-SEED RAW]")
        print(comm_byte_raw_df)

        print("[COMM BYTE MULTI-SEED SUMMARY]")
        print(comm_byte_summary_df)

        plot_comm_cost_to_target_multiseed(
            summary_df=comm_byte_summary_df,
            metric_label="Communication cost (MB)",
            xlabel=f"Cumulative communication cost (MB) through round {MULTI_N_ROUNDS}",
            save_path=os.path.join(
                LOG_DIR,
                f"eurosat_comm_cost_cum{MULTI_N_ROUNDS}rounds_multiseed_10runs_mb.png",
            ),
        )

        # -------------------------------
        # Round-averaged accuracy (main 3-way): accuracy-round 곡선 + 방법별
        # "1000라운드 전체 평균 정확도" 요약 테이블
        # -------------------------------
        acc_round_curve_df, acc_round_summary_df = make_accuracy_round_multiseed(
            log_dir=LOG_DIR,
            seeds=MULTI_SEEDS,
            method_prefix_map=method_prefix_map,
            labels=multiseed_labels,
            num_rounds=MULTI_N_ROUNDS,
            save_csv_path=os.path.join(
                LOG_DIR,
                f"table_accuracy_by_round_multiseed_10runs_{MULTI_N_ROUNDS}rounds.csv",
            ),
            save_summary_csv_path=os.path.join(
                LOG_DIR,
                f"table_round_averaged_accuracy_multiseed_10runs_{MULTI_N_ROUNDS}rounds.csv",
            ),
        )

        print("[ROUND-AVERAGED ACCURACY SUMMARY]")
        print(acc_round_summary_df)

        plot_accuracy_round_multiseed(
            curve_df=acc_round_curve_df,
            labels=multiseed_labels,
            num_rounds=MULTI_N_ROUNDS,
            save_path=os.path.join(
                LOG_DIR,
                f"eurosat_accuracy_round_multiseed_10runs_{MULTI_N_ROUNDS}rounds.png",
            ),
        )

        if PLOT_EXTRA_STARLINK10P_FIGS:
            starlink10p_method_prefix_map = {
                "FedAvg": "starlink10p_seed{idx}_fedavg",
                "FedISL": "starlink10p_seed{idx}_fedisl",
                "LH-FL": "starlink10p_seed{idx}_lhfl_b5000",
            }

        starlink10p_labels = ["FedAvg", "FedISL", "LH-FL"]

        # 1) Convergence Performance: 10-seed average accuracy-time curve (1000라운드 전체)
        plot_accuracy_time_multiseed_from_logs(
            log_dir=LOG_DIR,
            seeds=MULTI_SEEDS,
            method_prefix_map=starlink10p_method_prefix_map,
            labels=starlink10p_labels,
            seconds_per_round=60,
            time_unit="min",
            max_time_min=MULTI_N_ROUNDS,
            save_path=os.path.join(
                LOG_DIR,
                "starlink10p_accuracy_time_multiseed_10runs_nofedleo.png",
            ),
        )

        # 2) Load Balancing: representative seed 1 LH-FL before/after
        plot_starlink10p_load_balancing_from_logs(
            log_dir=LOG_DIR,
            prefix="starlink10p_seed1_lhfl_b5000",
        )

        # ================= B안 사다리 그림 (4칸) =================
        ladder_prefix_map = {
            "FedISL": "starlink10p_seed{idx}_fedisl_cap",
            "FedISL+offload": "starlink10p_seed{idx}_fedisl_cap_off",
            "Hier-central": "starlink10p_seed{idx}_hier_central",
            "LH-FL": "starlink10p_seed{idx}_lhfl_b5000",
        }
        ladder_labels = ["FedISL", "FedISL+offload", "Hier-central", "LH-FL"]

        # (1) 사다리 round completion time
        _, ladder_round_summary = make_round_completion_multiseed(
            log_dir=LOG_DIR,
            seeds=MULTI_SEEDS,
            method_prefix_map=ladder_prefix_map,
            labels=ladder_labels,
            seconds_per_round=60,
            time_unit="min",
            save_csv_path=os.path.join(LOG_DIR, "table_ladder_round_completion.csv"),
        )
        print("[LADDER ROUND SUMMARY]")
        print(ladder_round_summary)
        plot_round_completion_multiseed(
            summary_df=ladder_round_summary,
            time_unit="min",
            save_path=os.path.join(LOG_DIR, "eurosat_ladder_round_completion.png"),
        )

        # (2) 사다리 accuracy vs elapsed time
        plot_accuracy_time_multiseed_from_logs(
            log_dir=LOG_DIR,
            seeds=MULTI_SEEDS,
            method_prefix_map=ladder_prefix_map,
            labels=ladder_labels,
            seconds_per_round=60,
            time_unit="min",
            max_time_min=300,
            save_path=os.path.join(LOG_DIR, "eurosat_ladder_accuracy_time.png"),
        )

        # (3) 사다리 통신량 (hop, model + offload stacked)
        _, ladder_comm_summary = make_comm_cost_components_to_target_multiseed(
            log_dir=LOG_DIR,
            seeds=MULTI_SEEDS,
            method_prefix_map=ladder_prefix_map,
            labels=ladder_labels,
            target_acc=85.0,
            save_csv_path=os.path.join(LOG_DIR, "table_ladder_comm_cost.csv"),
        )
        print("[LADDER COMM SUMMARY]")
        print(ladder_comm_summary)
        plot_comm_cost_components_to_target_multiseed(
            summary_df=ladder_comm_summary,
            target_acc=85.0,
            save_path=os.path.join(LOG_DIR, "eurosat_ladder_comm_cost.png"),
        )

        # ================= 사다리 그림 (5칸) =================
        ladder_prefix_map = {
            "FedISL": "starlink10p_seed{idx}_fedisl_cap",
            "FedISL+decentral": "starlink10p_seed{idx}_fedisl_mix",
            "FedISL+offload": "starlink10p_seed{idx}_fedisl_cap_off",
            "FedISL+decentral+off": "starlink10p_seed{idx}_fedisl_mix_off",
            "LH-FL": "starlink10p_seed{idx}_lhfl_b5000",
        }
        ladder_labels = [
            "FedISL",
            "FedISL+decentral",
            "FedISL+offload",
            "FedISL+decentral+off",
            "LH-FL",
        ]

        # (1) round completion
        _, ladder_round = make_round_completion_multiseed(
            log_dir=LOG_DIR,
            seeds=MULTI_SEEDS,
            method_prefix_map=ladder_prefix_map,
            labels=ladder_labels,
            seconds_per_round=60,
            time_unit="min",
            save_csv_path=os.path.join(LOG_DIR, "table_ladder_round_completion.csv"),
        )
        print(ladder_round)
        plot_round_completion_multiseed(
            summary_df=ladder_round,
            time_unit="min",
            save_path=os.path.join(LOG_DIR, "eurosat_ladder_round_completion.png"),
        )

        # (2) accuracy vs time
        plot_accuracy_time_multiseed_from_logs(
            log_dir=LOG_DIR,
            seeds=MULTI_SEEDS,
            method_prefix_map=ladder_prefix_map,
            labels=ladder_labels,
            seconds_per_round=60,
            time_unit="min",
            max_time_min=300,
            save_path=os.path.join(LOG_DIR, "eurosat_ladder_accuracy_time.png"),
        )

        # (3) comm cost
        _, ladder_comm = make_comm_cost_components_to_target_multiseed(
            log_dir=LOG_DIR,
            seeds=MULTI_SEEDS,
            method_prefix_map=ladder_prefix_map,
            labels=ladder_labels,
            target_acc=85.0,
            save_csv_path=os.path.join(LOG_DIR, "table_ladder_comm_cost.csv"),
        )
        print(ladder_comm)
        plot_comm_cost_components_to_target_multiseed(
            summary_df=ladder_comm,
            target_acc=85.0,
            save_path=os.path.join(LOG_DIR, "eurosat_ladder_comm_cost.png"),
        )

        # ================= 지상국 없는 4칸 사다리 (all-decentral, fair ISL-only) =================
        # 모든 rung이 지상국 접촉/지연 없이 ISL로만 통신. (1->2) 리더 계층 추가 효과,
        # (1->3) offload 효과, (2->4 또는 3->4) 나머지 추가 = LH-FL.
        decentral_ladder_prefix_map = {
            "Decentral": "starlink10p_seed{idx}_decentral",
            "Decentral+Leader": "starlink10p_seed{idx}_decentral_leader",
            "Decentral+Offload": "starlink10p_seed{idx}_decentral_off",
            "LH-FL": "starlink10p_seed{idx}_lhfl_b5000",
        }
        decentral_ladder_labels = [
            "Decentral",
            "Decentral+Leader",
            "Decentral+Offload",
            "LH-FL",
        ]

        # (1) round completion time
        _, decentral_ladder_round = make_round_completion_multiseed(
            log_dir=LOG_DIR,
            seeds=MULTI_SEEDS,
            method_prefix_map=decentral_ladder_prefix_map,
            labels=decentral_ladder_labels,
            seconds_per_round=60,
            time_unit="min",
            save_csv_path=os.path.join(
                LOG_DIR, "table_decentral_ladder_round_completion.csv"
            ),
        )
        print("[DECENTRAL LADDER ROUND SUMMARY]")
        print(decentral_ladder_round)
        plot_round_completion_multiseed(
            summary_df=decentral_ladder_round,
            time_unit="min",
            save_path=os.path.join(
                LOG_DIR, "eurosat_decentral_ladder_round_completion.png"
            ),
        )

        # (2) accuracy vs elapsed time (1000라운드 전체)
        plot_accuracy_time_multiseed_from_logs(
            log_dir=LOG_DIR,
            seeds=MULTI_SEEDS,
            method_prefix_map=decentral_ladder_prefix_map,
            labels=decentral_ladder_labels,
            seconds_per_round=60,
            time_unit="min",
            max_time_min=MULTI_N_ROUNDS,
            save_path=os.path.join(
                LOG_DIR, "eurosat_decentral_ladder_accuracy_time.png"
            ),
        )

        # (3) cumulative communication cost through round MULTI_N_ROUNDS
        #     (hop, model + offload stacked)
        _, decentral_ladder_comm = make_comm_cost_components_to_target_multiseed(
            log_dir=LOG_DIR,
            seeds=MULTI_SEEDS,
            method_prefix_map=decentral_ladder_prefix_map,
            labels=decentral_ladder_labels,
            final_round=True,
            save_csv_path=os.path.join(
                LOG_DIR,
                f"table_decentral_ladder_comm_cost_cum{MULTI_N_ROUNDS}rounds.csv",
            ),
        )
        print("[DECENTRAL LADDER COMM SUMMARY]")
        print(decentral_ladder_comm)
        plot_comm_cost_components_to_target_multiseed(
            summary_df=decentral_ladder_comm,
            xlabel=f"Cumulative communication cost through round {MULTI_N_ROUNDS}",
            save_path=os.path.join(
                LOG_DIR,
                f"eurosat_decentral_ladder_comm_cost_cum{MULTI_N_ROUNDS}rounds.png",
            ),
        )

        # (3-B) cumulative communication cost through round MULTI_N_ROUNDS (byte/MB 기준)
        _, decentral_ladder_comm_mb = make_comm_cost_to_target_multiseed(
            log_dir=LOG_DIR,
            seeds=MULTI_SEEDS,
            method_prefix_map=decentral_ladder_prefix_map,
            labels=decentral_ladder_labels,
            comm_col="total_mb",
            final_round=True,
            save_csv_path=os.path.join(
                LOG_DIR,
                f"table_decentral_ladder_comm_cost_cum{MULTI_N_ROUNDS}rounds_mb.csv",
            ),
        )
        print("[DECENTRAL LADDER COMM SUMMARY (MB)]")
        print(decentral_ladder_comm_mb)
        plot_comm_cost_to_target_multiseed(
            summary_df=decentral_ladder_comm_mb,
            metric_label="Communication cost (MB)",
            xlabel=f"Cumulative communication cost (MB) through round {MULTI_N_ROUNDS}",
            save_path=os.path.join(
                LOG_DIR,
                f"eurosat_decentral_ladder_comm_cost_cum{MULTI_N_ROUNDS}rounds_mb.png",
            ),
        )

        # (4) round-averaged accuracy summary table (4개 방법)
        _, decentral_ladder_acc_round_summary = make_accuracy_round_multiseed(
            log_dir=LOG_DIR,
            seeds=MULTI_SEEDS,
            method_prefix_map=decentral_ladder_prefix_map,
            labels=decentral_ladder_labels,
            num_rounds=MULTI_N_ROUNDS,
            save_csv_path=os.path.join(
                LOG_DIR,
                f"table_decentral_ladder_accuracy_by_round_{MULTI_N_ROUNDS}rounds.csv",
            ),
            save_summary_csv_path=os.path.join(
                LOG_DIR,
                f"table_decentral_ladder_round_averaged_accuracy_{MULTI_N_ROUNDS}rounds.csv",
            ),
        )
        print("[DECENTRAL LADDER ROUND-AVERAGED ACCURACY SUMMARY]")
        print(decentral_ladder_acc_round_summary)

        print("[DONE] Multi-seed final experiment completed.")
        return

    # ===============================
    # Partial FedAvg Ratio Analysis
    # ===============================
    if RUN_PARTIAL_FEDAVG:
        partial_ratios = [1.0, 0.75, 0.5]

        for rho in partial_ratios:
            # fairness를 위해 같은 non-IID split을 ratio마다 다시 생성
            partial_client_datasets = make_clients_from_dataset(
                dataset=central_train,
                num_clients=NUM_CLIENTS,
                split_mode="non",
                alpha=ALPHA,
                seed=GLOBAL_SEED,
            )

            run_partial_fedavg(
                client_datasets=partial_client_datasets,
                test_dataset=test_dataset,
                per_sat_minutes=per_sat_minutes,
                rho=rho,
                num_rounds=PARTIAL_ROUNDS,
                log_dir=LOG_DIR,
                log_prefix=f"eurosat_partial_rho{int(rho * 100)}",
                device=device,
                hidden_size=100,
                lr=1e-3,
                batch_size=32,
                local_epochs=1,
                seed=GLOBAL_SEED,
            )

    # ===============================
    # Common paths and labels for plots
    # Final comparison: FedAvg vs FedISL vs LH-FL
    # ===============================
    acc_csvs = [
        os.path.join(LOG_DIR, "eurosat_ideal_accuracy.csv"),
        os.path.join(LOG_DIR, "eurosat_fedisl_accuracy.csv"),
        os.path.join(LOG_DIR, "eurosat_offload_accuracy.csv"),
    ]

    comm_csvs = [
        os.path.join(LOG_DIR, "eurosat_ideal_comm.csv"),
        os.path.join(LOG_DIR, "eurosat_fedisl_comm.csv"),
        os.path.join(LOG_DIR, "eurosat_offload_comm.csv"),
    ]

    train_time_csvs = [
        os.path.join(LOG_DIR, "eurosat_ideal_train_time.csv"),
        os.path.join(LOG_DIR, "eurosat_fedisl_train_time.csv"),
        os.path.join(LOG_DIR, "eurosat_offload_train_time.csv"),
    ]

    labels = [
        "FedAvg",
        "FedISL",
        "LH-FL",
    ]

    # plot_multiple_accuracy_vs_time(
    #    acc_csv_paths=acc_csv_paths,
    #    train_time_csv_paths=train_time_csv_paths,
    #    labels=[
    #        "CENTRALIZED FL",
    #        "HYBRID-OFFLOAD",
    #        "DECENTRALIZED FL",
    #    ],
    #    time_unit="min",
    #    title="Accuracy Convergence Over Actual #Time",
    #    max_x=1000,
    #    save_path=os.path.join(LOG_DIR, #"eurosat_accuracy_comparison_time.png"),
    # )

    # ===============================
    # Offloading Before/After Plots
    # ===============================
    before_csv = os.path.join(LOG_DIR, "eurosat_offload_classdist_before.csv")
    after_csv = os.path.join(LOG_DIR, "eurosat_offload_classdist_after.csv")
    cap_csv = os.path.join(LOG_DIR, "eurosat_offload_capacity.csv")
    meeting_csv = os.path.join(LOG_DIR, "eurosat_offload_meeting_counts.csv")

    if os.path.exists(before_csv):
        plot_class_distribution_from_csv(
            before_csv,
            title=None,
            save_path=before_csv.replace(".csv", ".png"),
        )
        print(f"[SAVE] {before_csv.replace('.csv', '.png')}")

    if os.path.exists(after_csv):
        plot_class_distribution_from_csv(
            after_csv,
            title=None,
            save_path=after_csv.replace(".csv", ".png"),
        )
        print(f"[SAVE] {after_csv.replace('.csv', '.png')}")

    if os.path.exists(meeting_csv):
        plot_meeting_counts_heatmap(
            meeting_csv,
            save_path=meeting_csv.replace(".csv", ".png"),
        )
        print(f"[SAVE] {meeting_csv.replace('.csv', '.png')}")

    if os.path.exists(cap_csv):
        plot_capacity_from_csv(
            cap_csv,
            save_path=cap_csv.replace(".csv", ".png"),
        )
        print(f"[SAVE] {cap_csv.replace('.csv', '.png')}")

    if (
        os.path.exists(before_csv)
        and os.path.exists(after_csv)
        and os.path.exists(cap_csv)
    ):
        norm_save = os.path.join(LOG_DIR, "eurosat_offload_normalized_load.png")
        plot_normalized_load(
            before_csv,
            after_csv,
            cap_csv,
            save_path=norm_save,
        )
        print(f"[SAVE] {norm_save}")

    # ===============================
    # Communication Comparison Plot
    # ===============================
    comm_csvs = [
        os.path.join(LOG_DIR, "eurosat_ideal_comm.csv"),
        os.path.join(LOG_DIR, "eurosat_decentral_comm.csv"),
        os.path.join(LOG_DIR, "eurosat_offload_comm.csv"),
    ]

    comm_labels = [
        "CENTRALIZED FL",
        "DECENTRALIZED FL",
        "LH-FL",
    ]

    valid_comm_csvs = []
    valid_comm_labels = []

    for p, l in zip(comm_csvs, comm_labels):
        if os.path.exists(p):
            valid_comm_csvs.append(p)
            valid_comm_labels.append(l)
        else:
            print(f"[WARN] comm csv not found: {p}")

    if len(valid_comm_csvs) >= 2:
        # 1) per-round communication
        plot_multiple_communication(
            csv_paths=valid_comm_csvs,
            labels=valid_comm_labels,
            which="model_hops",
            mode="per_round",
            max_x=30,
            seconds_per_round=60,
            time_unit="min",
            title=None,
            save_path=os.path.join(LOG_DIR, "eurosat_comm_model_hops_comparison.png"),
        )
        print("[SAVE] eurosat_comm_model_hops_comparison.png")

        # 2) cumulative communication
        plot_multiple_communication(
            csv_paths=valid_comm_csvs,
            labels=valid_comm_labels,
            which="model_hops",
            mode="cumulative",
            max_x=30,
            seconds_per_round=60,
            time_unit="min",
            title=None,
            save_path=os.path.join(LOG_DIR, "eurosat_comm_model_hops_cumulative.png"),
        )
        print("[SAVE] eurosat_comm_model_hops_cumulative.png")

    else:
        print("[WARN] Not enough communication CSV files for comparison.")

    offload_acc_path = os.path.join(LOG_DIR, "eurosat_offload_accuracy.csv")
    if os.path.exists(offload_acc_path):
        df = pd.read_csv(offload_acc_path)
        print("[DEBUG] eurosat_offload_accuracy.csv")
        print(df.head())
        print(df.tail())
        print("[DEBUG] shape =", df.shape)

        # ===============================
    # Bottleneck Mitigation Plot
    # ===============================
    plot_global_update_intervals(
        csv_paths=acc_csvs,
        labels=labels,
        seconds_per_round=60,
        time_unit="min",
        title=None,
        save_path=os.path.join(LOG_DIR, "eurosat_global_update_intervals.png"),
    )

    # ===============================
    # Accuracy Comparison (Elapsed-time based)
    # ===============================
    plot_multiple_accuracy_sim_time(
        acc_csv_paths=acc_csvs,
        labels=labels,
        seconds_per_round=60,
        time_unit="min",
        title=None,
        save_path=os.path.join(LOG_DIR, "eurosat_accuracy_comparison_elapsed_time.png"),
        use_step=False,
        max_x=300,
    )

    # ===============================
    # Accuracy Comparison (Update-round based)
    # ===============================
    plot_multiple_accuracy_update_round(
        csv_paths=acc_csvs,
        labels=labels,
        title=None,
        save_path=os.path.join(LOG_DIR, "eurosat_accuracy_comparison_update_round.png"),
        use_step=False,
        max_x=None,
    )

    # ===============================
    # Accuracy vs Cumulative Communication
    # ===============================
    plot_multiple_accuracy_vs_cumulative_comm(
        acc_csv_paths=acc_csvs,
        comm_csv_paths=comm_csvs,
        labels=labels,
        title=None,
        save_path=os.path.join(
            LOG_DIR, "eurosat_accuracy_vs_cumulative_transmission.png"
        ),
        comm_col="model_hops",
        xscale_log=False,
        max_x=None,
    )

    # ===============================
    # Communication Cost to Target Accuracy
    # ===============================
    target_comm_table = plot_comm_cost_to_target_bar(
        acc_csv_paths=acc_csvs,
        comm_csv_paths=comm_csvs,
        labels=[
            "Centralized FL",
            "Decentralized FL",
            "LH-FL",
        ],
        target_acc=85.0,
        save_path=os.path.join(LOG_DIR, "eurosat_comm_cost_to_85acc_bar.png"),
        save_csv_path=os.path.join(LOG_DIR, "table_comm_cost_to_85acc_new.csv"),
        seconds_per_round=60,
        yscale_log=False,
    )

    print(target_comm_table)

    # ===============================
    # Partial Aggregation Ratio Plot
    # ===============================
    partial_acc_csvs = [
        os.path.join(LOG_DIR, "eurosat_partial_rho100_accuracy.csv"),
        os.path.join(LOG_DIR, "eurosat_partial_rho75_accuracy.csv"),
        os.path.join(LOG_DIR, "eurosat_partial_rho50_accuracy.csv"),
        os.path.join(LOG_DIR, "eurosat_offload_accuracy.csv"),
    ]

    partial_train_time_csvs = [
        os.path.join(LOG_DIR, "eurosat_partial_rho100_train_time.csv"),
        os.path.join(LOG_DIR, "eurosat_partial_rho75_train_time.csv"),
        os.path.join(LOG_DIR, "eurosat_partial_rho50_train_time.csv"),
        os.path.join(LOG_DIR, "eurosat_offload_train_time.csv"),
    ]

    partial_labels = [
        r"FedAvg-$\rho=1.0$",
        r"FedAvg-$\rho=0.75$",
        r"FedAvg-$\rho=0.5$",
        "LH-FL",
    ]

    valid_partial_acc = []
    valid_partial_time = []
    valid_partial_labels = []

    for acc_p, time_p, lab in zip(
        partial_acc_csvs, partial_train_time_csvs, partial_labels
    ):
        if os.path.exists(acc_p) and os.path.exists(time_p):
            valid_partial_acc.append(acc_p)
            valid_partial_time.append(time_p)
            valid_partial_labels.append(lab)
        else:
            print(f"[WARN][PARTIAL-PLOT] missing: {acc_p} or {time_p}")

    if len(valid_partial_acc) >= 2:
        plot_multiple_accuracy_time(
            acc_csv_paths=valid_partial_acc,
            train_time_csv_paths=valid_partial_time,
            labels=valid_partial_labels,
            time_unit="min",
            title=None,
            save_path=os.path.join(LOG_DIR, "eurosat_partial_ratio_accuracy_time.png"),
            use_step=False,
            max_x=300,
        )
        print("[SAVE] eurosat_partial_ratio_accuracy_time.png")

    # ===============================
    # Cycle Completion Time
    # ===============================
    plot_cycle_completion_time(
        train_time_csv_paths=train_time_csvs,
        labels=labels,
        time_unit="min",
        title=None,
        save_path=os.path.join(
            LOG_DIR, "eurosat_cycle_completion_time_round_based.png"
        ),
        x_axis="simulation_round",
        max_x=600,
        smoothing_window=5,
    )

    plot_update_completion_summary_from_accuracy(
        acc_csv_paths=acc_csvs,
        labels=labels,
        seconds_per_round=60,
        time_unit="min",
        title=None,
        save_path=os.path.join(
            LOG_DIR, "eurosat_update_completion_time_bar_zoomed.png"
        ),
        save_csv_path=os.path.join(LOG_DIR, "table_update_completion_time_summary.csv"),
        plot_type="bar",
        ylim=(23, 32),
    )

    plot_update_completion_saved_segment_bar(
        summary_csv_path=os.path.join(
            LOG_DIR, "table_update_completion_time_summary.csv"
        ),
        baseline_methods=("CENTRALIZED FL", "DECENTRALIZED FL"),
        proposed_method="LH-FL",
        save_path=os.path.join(LOG_DIR, "eurosat_update_completion_saved_segment.png"),
    )

    plot_update_completion_reduction_bar(
        summary_csv_path=os.path.join(
            LOG_DIR, "table_update_completion_time_summary.csv"
        ),
        baseline_methods=("CENTRALIZED FL", "DECENTRALIZED FL"),
        proposed_method="LH-FL",
        save_path=os.path.join(LOG_DIR, "eurosat_update_completion_reduction.png"),
    )

    plot_update_completion_time_from_accuracy(
        acc_csv_paths=acc_csvs,
        labels=labels,
        seconds_per_round=60,
        time_unit="min",
        title=None,
        save_path=os.path.join(LOG_DIR, "eurosat_update_completion_time.png"),
        max_x=None,
        smoothing_window=None,
    )

    plot_normalized_update_completion_time(
        summary_csv_path=os.path.join(
            LOG_DIR, "table_update_completion_time_summary.csv"
        ),
        reference_method="FedAvg",
        save_path=os.path.join(
            LOG_DIR, "eurosat_normalized_update_completion_time.png"
        ),
    )

    plot_update_completion_dumbbell(
        summary_csv_path=os.path.join(
            LOG_DIR, "table_update_completion_time_summary.csv"
        ),
        baseline_methods=("CENTRALIZED FL", "DECENTRALIZED FL"),
        proposed_method="LH-FL",
        save_path=os.path.join(LOG_DIR, "eurosat_update_completion_dumbbell.png"),
    )

    # ===============================
    # Additional comparison: Centralized FL vs FedISL vs LH-FL
    # ===============================
    fedisl_acc_csvs = [
        os.path.join(LOG_DIR, "eurosat_ideal_accuracy.csv"),
        os.path.join(LOG_DIR, "eurosat_fedisl_accuracy.csv"),
        os.path.join(LOG_DIR, "eurosat_offload_accuracy.csv"),
    ]

    fedisl_comm_csvs = [
        os.path.join(LOG_DIR, "eurosat_ideal_comm.csv"),
        os.path.join(LOG_DIR, "eurosat_fedisl_comm.csv"),
        os.path.join(LOG_DIR, "eurosat_offload_comm.csv"),
    ]

    fedisl_train_time_csvs = [
        os.path.join(LOG_DIR, "eurosat_ideal_train_time.csv"),
        os.path.join(LOG_DIR, "eurosat_fedisl_train_time.csv"),
        os.path.join(LOG_DIR, "eurosat_offload_train_time.csv"),
    ]

    fedisl_labels = [
        "FedAvg",
        "FedISL",
        "LH-FL",
    ]

    # 존재하는 파일만 검사
    fedisl_ready = all(os.path.exists(p) for p in fedisl_acc_csvs)

    if fedisl_ready:
        # 1) Accuracy vs elapsed time
        plot_multiple_accuracy_sim_time(
            acc_csv_paths=fedisl_acc_csvs,
            labels=fedisl_labels,
            seconds_per_round=60,
            time_unit="min",
            title=None,
            save_path=os.path.join(LOG_DIR, "eurosat_extra_fedisl_accuracy_time.png"),
            use_step=False,
            max_x=300,
        )

        # 2) Communication cost to reach 85% accuracy
        fedisl_comm_ready = all(os.path.exists(p) for p in fedisl_comm_csvs)

        if fedisl_comm_ready:
            fedisl_target_comm_table = plot_comm_cost_to_target_bar(
                acc_csv_paths=fedisl_acc_csvs,
                comm_csv_paths=fedisl_comm_csvs,
                labels=[
                    "FedAvg",
                    "FedISL",
                    "LH-FL",
                ],
                target_acc=85.0,
                save_path=os.path.join(
                    LOG_DIR, "eurosat_extra_fedisl_comm_cost_to_85acc.png"
                ),
                save_csv_path=os.path.join(
                    LOG_DIR, "table_extra_fedisl_comm_cost_to_85acc.csv"
                ),
                seconds_per_round=60,
                yscale_log=False,
            )
            print(fedisl_target_comm_table)

        # 3) Average update completion time
        plot_update_completion_summary_from_accuracy(
            acc_csv_paths=fedisl_acc_csvs,
            labels=fedisl_labels,
            seconds_per_round=60,
            time_unit="min",
            title=None,
            save_path=os.path.join(
                LOG_DIR, "eurosat_extra_fedisl_update_completion_time.png"
            ),
            save_csv_path=os.path.join(
                LOG_DIR, "table_extra_fedisl_update_completion_time.csv"
            ),
            plot_type="bar",
            ylim=None,
        )

    else:
        print(
            "[WARN][FedISL-extra] Missing one or more accuracy CSVs. Skip FedISL extra plots."
        )
        for p in fedisl_acc_csvs:
            if not os.path.exists(p):
                print(f"  missing: {p}")


if __name__ == "__main__":
    main()
