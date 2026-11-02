# fl_case2_always_connected_hetero_train.py
# [초기 프로토타입] 항상 연결 + 위성별 학습시간이 다른(heterogeneous) 조건의 FedAvg 실험 및 이벤트 요약·그래프.
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn import preprocessing
from pandas import read_csv, read_feather
import matplotlib.pyplot as plt
import os

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

# =========================
# 고정 별자리(분 단위 위상 오프셋)
# =========================
FIXED_CONSTELLATION = [
    {"h_km": 537, "incl_deg": 42, "offset_min": 4},  # 235s ≈ 4 min
    {"h_km": 572, "incl_deg": 35, "offset_min": 15},  # 905s ≈ 15 min
    {"h_km": 644, "incl_deg": 36, "offset_min": 70},  # 4225s ≈ 70 min
    {"h_km": 781, "incl_deg": 50, "offset_min": 20},  # 1202s ≈ 20 min
    {"h_km": 754, "incl_deg": 48, "offset_min": 74},  # 4453s ≈ 74 min
    {"h_km": 968, "incl_deg": 58, "offset_min": 118},  # 7051s ≈ 118 min
    {"h_km": 898, "incl_deg": 43, "offset_min": 9},  # 562s ≈ 9 min
]


class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def Zipf(a: np.float64, minv: np.uint64, maxv: np.uint64, size=None):
    if minv == 0:
        raise ZeroDivisionError("min must be >= 1")
    v = np.arange(minv, maxv + 1)
    p = 1.0 / np.power(v, a)
    p /= np.sum(p)
    return np.random.choice(v, size=size, replace=True, p=p)


class NIDDataset(Dataset):
    def __init__(
        self,
        data_file,
        proto_encoder=None,
        category_encoder=None,
        scaler=None,
        n_best_features=15,
        selected_features=[],
        excluded_classes=[],
        selected_classes=[],
        bBinary=False,
    ):
        super().__init__()
        ext = data_file.split(".")[-1].lower()
        if ext == "csv":
            df = read_csv(data_file, header=0, sep=",")
        elif ext == "feather":
            df = read_feather(data_file)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        lbl_col = {
            "unsw": "attack_cat",
            "kdd": "label",
            "cicids": "Label",
            "unr": "Label",
        }
        drops = {
            "unsw": ["srcip", "sport", "dstip", "dsport", "state", "service", "label"],
            "kdd": [
                "index",
                "service",
                "flag",
                "land",
                "is_host_login",
                "is_guest_login",
            ],
            "cicids": ["index", "Timestamp"],
            "unr": ["index", "Switch ID", "Port Number", "is_valid", "Binary Label"],
        }

        dataset_name = None
        lowpath = data_file.lower()
        for k in lbl_col.keys():
            if k in lowpath:
                dataset_name = k
                break
        if dataset_name is None:
            raise ValueError(
                "파일명에 데이터셋 단서(unsw/kdd/cicids/unr)가 필요합니다."
            )

        label_col = lbl_col[dataset_name]
        df = df.drop(drops[dataset_name], axis=1, errors="ignore")

        if category_encoder is None:
            self.category_encoder = preprocessing.OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            self.category_encoder.fit(df[label_col].values.reshape((-1, 1)))
        else:
            self.category_encoder = category_encoder

        df[label_col] = self.category_encoder.transform(
            df[label_col].values.reshape((-1, 1))
        )
        Y_series = df[label_col].values

        self.class_weight = torch.from_numpy(
            compute_class_weight("balanced", classes=np.unique(Y_series), y=Y_series)
        ).type(torch.float)

        self.X = df.drop([label_col], axis=1)
        self.features = self.X.columns.to_list()
        self.n_features = len(self.features)
        self.n_classes = len(np.unique(Y_series))

        if scaler is None:
            self.scaler = preprocessing.MinMaxScaler()
            self.scaler.fit(self.X)
        else:
            self.scaler = scaler

        Xn = self.scaler.transform(self.X)
        Xn = np.clip(Xn, 0.0, 1.0)
        self.X = torch.from_numpy(Xn).type(torch.float32)
        self.Y = torch.from_numpy(Y_series).type(torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train_one_epoch(model, dataset, device, batch_size=32, lr=1e-3, class_weight=None):
    model.to(device)
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss(
        weight=class_weight.to(device) if class_weight is not None else None
    )
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    total_loss = 0.0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optim.zero_grad()
        out = model(X)
        loss = criterion(out, Y)
        loss.backward()
        optim.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=64):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        out = model(X)
        pred = out.argmax(dim=1)
        correct += (pred == Y).sum().item()
        total += Y.size(0)
    return 100.0 * correct / max(1, total)


def fedavg_aggregate(global_model, client_models, participating_indices, weights=None):
    if len(participating_indices) == 0:
        return
    with torch.no_grad():
        g = global_model.state_dict()
        for k in g.keys():
            g[k].zero_()
        if weights is None:
            for idx in participating_indices:
                s = client_models[idx].state_dict()
                for k in g.keys():
                    g[k].add_(s[k])
            for k in g.keys():
                g[k].div_(float(len(participating_indices)))
        else:
            assert len(weights) == len(participating_indices)
            for idx, w in zip(participating_indices, weights):
                s = client_models[idx].state_dict()
                for k in g.keys():
                    g[k].add_(s[k] * float(w))
        global_model.load_state_dict(g)


def federated_learning_always_connected(
    client_datasets,
    test_dataset,
    per_client_train_seconds,
    max_time=200000,
    base_tick_seconds=1,
    local_epochs=1,
    device=None,
    hidden_size=100,
    lr=1e-3,
    target_acc=None,
    weighted_avg=False,
    start_offset_ticks=None,
):
    """
    항상 연결 + 클라이언트별 학습시간 이질.
    이벤트 로그에 학습 시작/종료/대기시간/라운드 요약이 가능하도록 version/시각 저장.
    """
    n_clients = len(client_datasets)
    input_size = test_dataset.n_features
    num_classes = test_dataset.n_classes
    global_model = DNN(
        input_size=input_size, hidden_size=hidden_size, num_classes=num_classes
    ).to(device)
    client_models = [copy.deepcopy(global_model).to(device) for _ in range(n_clients)]

    if isinstance(per_client_train_seconds, dict):
        train_ticks_needed = [
            max(1, int(np.ceil(per_client_train_seconds[i] / base_tick_seconds)))
            for i in range(n_clients)
        ]
    else:
        assert len(per_client_train_seconds) == n_clients
        train_ticks_needed = [
            max(1, int(np.ceil(s / base_tick_seconds)))
            for s in per_client_train_seconds
        ]

    global_version = 0
    downloaded_version = [-1] * n_clients
    remaining_train = [0] * n_clients
    trained_ready = [False] * n_clients
    uploaded_for_version = set()
    train_start_time = [-1] * n_clients

    events = []  # (time, type, payload)
    accuracy_trace = []  # (time, accuracy)

    client_sizes = [len(ds) for ds in client_datasets]
    total_size = float(sum(client_sizes))
    client_weights = [sz / total_size for sz in client_sizes] if weighted_avg else None

    for t in range(max_time):
        # 1) 다운로드 (이번 버전을 아직 못 받은 클라에게 1회 전송)
        for i in range(n_clients):
            if downloaded_version[i] < global_version:
                if start_offset_ticks is not None:
                    if start_offset_ticks[i] is not None and t < start_offset_ticks[i]:
                        continue
                client_models[i].load_state_dict(
                    copy.deepcopy(global_model.state_dict())
                )
                downloaded_version[i] = global_version
                remaining_train[i] = train_ticks_needed[i]
                trained_ready[i] = False
                train_start_time[i] = t
                events.append((t, "download", {"client": i, "version": global_version}))
                events.append(
                    (
                        t,
                        "train_start",
                        {
                            "client": i,
                            "version": global_version,
                            "ticks": remaining_train[i],
                            "start_time": t,
                            "start_time_seconds": t * base_tick_seconds,
                        },
                    )
                )

        # 2) 학습 카운트다운
        for i in range(n_clients):
            if downloaded_version[i] == global_version and remaining_train[i] > 0:
                remaining_train[i] -= 1
                if remaining_train[i] == 0:
                    last_loss = None
                    for _ in range(local_epochs):
                        last_loss = train_one_epoch(
                            client_models[i],
                            client_datasets[i],
                            device=device,
                            batch_size=32,
                            lr=lr,
                            class_weight=client_datasets[i].class_weight,
                        )
                    trained_ready[i] = True
                    elapsed_ticks = (
                        t - train_start_time[i] if train_start_time[i] >= 0 else None
                    )
                    elapsed_seconds = (
                        (elapsed_ticks * base_tick_seconds)
                        if elapsed_ticks is not None
                        else None
                    )
                    events.append(
                        (
                            t,
                            "train_done",
                            {
                                "client": i,
                                "version": downloaded_version[i],
                                "last_epoch_loss": float(last_loss)
                                if last_loss is not None
                                else None,
                                "elapsed_ticks": int(elapsed_ticks)
                                if elapsed_ticks is not None
                                else None,
                                "elapsed_seconds": float(elapsed_seconds)
                                if elapsed_seconds is not None
                                else None,
                            },
                        )
                    )

        # 3) 업링크 (항상 연결이므로 즉시 업로드)
        for i in range(n_clients):
            if (
                (downloaded_version[i] == global_version)
                and trained_ready[i]
                and (i not in uploaded_for_version)
            ):
                uploaded_for_version.add(i)
                events.append((t, "upload", {"client": i, "version": global_version}))

        # 4) 전원 업링크 → FedAvg + 평가 + 로그
        if len(uploaded_for_version) == n_clients:
            w = client_weights if weighted_avg else None
            fedavg_aggregate(
                global_model, client_models, list(uploaded_for_version), weights=w
            )
            acc = evaluate(global_model, test_dataset, device=device)
            accuracy_trace.append((t, float(acc)))
            events.append(
                (
                    t,
                    "global_update",
                    {
                        "version": global_version,
                        "fed_round": global_version + 1,
                        "accuracy": float(acc),
                    },
                )
            )
            print(
                f"[Global Update @ t={t}min] FedRound {global_version + 1}  Accuracy: {acc:.2f}%"
            )

            # (가독용) 각 클라 총 경과시간(업데이트까지)만 간단 요약
            for i in range(n_clients):
                if train_start_time[i] >= 0:
                    et_ticks = t - train_start_time[i]
                    et_secs = et_ticks * base_tick_seconds
                    print(
                        f"  - client {i}: train_window {et_ticks} ticks (~{et_secs:.1f}s)"
                    )

            if (target_acc is not None) and (acc >= float(target_acc)):
                print("==> 목표 정확도 도달. 종료.")
                return global_model, events, accuracy_trace, base_tick_seconds

            global_version += 1
            uploaded_for_version.clear()
            for i in range(n_clients):
                remaining_train[i] = 0
                trained_ready[i] = False

    return global_model, events, accuracy_trace, base_tick_seconds


# -------- 추가: 이벤트 집계/리포트 (라운드 주기, 학습 vs 대기, 유휴비율) --------
def summarize_events(events, n_clients, base_tick_seconds=1):
    """
    events에서 라운드별/클라별 지표를 계산해 콘솔 출력하고 DataFrame 반환.
    - round_period_ticks, round_period_sec
    - per client: train_ticks, wait_ticks, idle_ratio
    """
    # 라운드(=version) 별 글로벌 업데이트 시각 수집
    round_update_time = {}  # version -> t
    for t, etype, payload in events:
        if etype == "global_update":
            v = payload.get("version")
            round_update_time[v] = t

    # 각 라운드에서 클라별 train_done 시각/학습시간 수집
    train_done_time = {}  # version -> {client: t}
    train_elapsed = {}  # version -> {client: elapsed_ticks}
    for t, etype, payload in events:
        if etype == "train_done":
            v = payload.get("version")
            i = payload.get("client")
            if v is None:
                continue
            train_done_time.setdefault(v, {})[i] = t
            train_elapsed.setdefault(v, {})[i] = payload.get("elapsed_ticks")

    # 라운드별 요약 출력
    rows = []
    versions = sorted(round_update_time.keys())
    for idx, v in enumerate(versions):
        t_update = round_update_time[v]
        # 라운드 주기: 첫 라운드는 baseline 0으로 계산
        if idx == 0:
            period_ticks = t_update - 0
        else:
            prev_t = round_update_time[versions[idx - 1]]
            period_ticks = t_update - prev_t
        period_sec = period_ticks * base_tick_seconds

        print(f"\n=== Round {v + 1} Summary ===")
        print(f"- round period: {period_ticks} ticks (~{period_sec:.1f}s)")

        idle_ratios = []
        for i in range(n_clients):
            tr_el = (train_elapsed.get(v, {}) or {}).get(i, None)
            tr_done_t = (train_done_time.get(v, {}) or {}).get(i, None)
            if (tr_el is None) or (tr_done_t is None):
                # 데이터 없으면 스킵/0 처리
                continue
            wait_ticks = max(0, t_update - tr_done_t)
            idle_ratio = (wait_ticks / period_ticks) if period_ticks > 0 else 0.0
            idle_ratios.append(idle_ratio)

            print(
                f"  client {i}: train={tr_el}t (~{tr_el * base_tick_seconds:.1f}s) "
                f"wait={wait_ticks}t (~{wait_ticks * base_tick_seconds:.1f}s) "
                f"idle_ratio={idle_ratio:.2%}"
            )

            rows.append(
                {
                    "version": v,
                    "round": v + 1,
                    "client": i,
                    "round_period_ticks": period_ticks,
                    "round_period_sec": period_sec,
                    "train_ticks": tr_el,
                    "train_sec": tr_el * base_tick_seconds,
                    "wait_ticks": wait_ticks,
                    "wait_sec": wait_ticks * base_tick_seconds,
                    "idle_ratio": idle_ratio,
                }
            )

        if idle_ratios:
            avg_idle = np.mean(idle_ratios)
            med_idle = np.median(idle_ratios)
            print(f"- idle ratio (mean/median): {avg_idle:.2%} / {med_idle:.2%}")

    df = pd.DataFrame(rows)
    return df


def plot_accuracy_curve(csv_path: str, save_path: str = None):
    """
    time-accuracy CSV를 시각화하는 함수
    Args:
        csv_path (str): 'time,accuracy' 형식의 CSV 파일 경로
        save_path (str, optional): 그래프 저장 경로. None이면 화면에만 표시
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일이 존재하지 않습니다: {csv_path}")

    # CSV 로드
    df = pd.read_csv(csv_path)
    if "time" not in df.columns or "accuracy" not in df.columns:
        raise ValueError("CSV 파일에 'time'과 'accuracy' 컬럼이 있어야 합니다.")

    # 그래프 그리기
    plt.figure(figsize=(8, 5))
    plt.plot(df["time"], df["accuracy"], marker="o", color="orange", linewidth=2)
    plt.title("Accuracy over Training Time", fontsize=14)
    plt.xlabel("Time (steps)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    # 저장 or 표시
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ 그래프 저장 완료: {save_path}")
    else:
        plt.show()


def plot_total_train_wait_all_rounds(csv_path: str, save_path: str = None):
    """
    case2_round_efficiency.csv 기반:
    전체 라운드 동안 각 클라이언트가 학습/유휴에 쓴 총 시간을 시각화.
    """
    df = pd.read_csv(csv_path)

    # 라운드별 합계 -> 클라이언트별 누적합
    totals = df.groupby("client")[["train_sec", "wait_sec"]].sum().reset_index()
    totals["total_sec"] = totals["train_sec"] + totals["wait_sec"]

    # 그래프
    plt.figure(figsize=(9, 5))
    clients = totals["client"].astype(str)
    plt.bar(clients, totals["train_sec"], label="Train (sec)", color="#4A90E2")
    plt.bar(
        clients,
        totals["wait_sec"],
        bottom=totals["train_sec"],
        label="Wait/Idle (sec)",
        color="#D0021B",
    )

    plt.title("Total Time per Client Across All Rounds", fontsize=14)
    plt.xlabel("Client (Satellite)", fontsize=12)
    plt.ylabel("Total Time (seconds)", fontsize=12)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ 그래프 저장 완료: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    num_clients = len(FIXED_CONSTELLATION)
    max_time = 200000
    base_tick_seconds = 60
    local_epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_acc = 70.0

    train_path = "LeoFL/UNR-IDD/UNR-IDD_train.feather"
    test_path = "LeoFL/UNR-IDD/UNR-IDD_test.feather"

    central_train = NIDDataset(train_path, bBinary=False)

    client_datasets = []
    for i in range(num_clients):
        d = NIDDataset(
            train_path,
            bBinary=False,
            category_encoder=central_train.category_encoder,
            scaler=central_train.scaler,
        )
        total_len = len(d)
        indices = np.random.choice(total_len, 2000, replace=False)
        d.X = d.X[indices]
        d.Y = d.Y[indices]
        print(
            f"Client {i}: {len(d.Y)} samples, class dist = {np.bincount(d.Y.numpy().astype(int))}"
        )
        client_datasets.append(d)

    test_dataset = NIDDataset(
        test_path,
        bBinary=False,
        category_encoder=central_train.category_encoder,
        scaler=central_train.scaler,
    )

    # 학습 시간: "분" 단위로 사람이 읽기 좋게 정의
    per_client_train_minutes = [3, 4, 5, 6, 7, 8, 10]
    # 내부 사용은 초 단위이므로 변환해서 넘김
    per_client_train_seconds = [m * 60 for m in per_client_train_minutes]

    # 위성 위상 오프셋(분) → 시작 지연 틱(=분). 틱 단위가 분이므로 그대로 사용.
    start_offset_ticks = [cfg["offset_min"] for cfg in FIXED_CONSTELLATION]

    print("=== Fixed LEO Constellation (phase offset in minutes) ===")
    for i, c in enumerate(FIXED_CONSTELLATION):
        print(
            f"[client {i}] h={c['h_km']} km, inc={c['incl_deg']}°, phase_offset={c['offset_min']} min"
        )

    _, events, acc_trace, tick_sec = federated_learning_always_connected(
        client_datasets=client_datasets,
        test_dataset=test_dataset,
        per_client_train_seconds=per_client_train_seconds,
        max_time=max_time,
        base_tick_seconds=base_tick_seconds,
        local_epochs=local_epochs,
        device=device,
        hidden_size=100,
        lr=1e-3,
        target_acc=target_acc,
        weighted_avg=False,
        start_offset_ticks=start_offset_ticks,
    )

    # 정확도 시계열 간단 확인
    if acc_trace:
        df_acc = pd.DataFrame(acc_trace, columns=["time", "accuracy"])
        print("\n=== Accuracy Trace (head) ===")
        print(df_acc.head())

    # ★ 추가 집계: 라운드 주기 / 학습 vs 대기 / 유휴비율
    print("\n=== Per-round Efficiency Summary ===")
    df_eff = summarize_events(events, n_clients=num_clients, base_tick_seconds=tick_sec)

    df_eff.to_csv("case2_round_efficiency.csv", index=False)
    df_acc.to_csv("case2_accuracy_trace.csv", index=False)

    csv_file = "case2_accuracy_trace.csv"
    plot_accuracy_curve(csv_file, save_path="accuracy_curve.png")

    plot_total_train_wait_all_rounds(
        "case2_round_efficiency.csv", save_path="total_train_wait_all_rounds.png"
    )
