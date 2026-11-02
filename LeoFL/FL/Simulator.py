# [초기 프로토타입] Leocompare.py의 3가지 시나리오를 run_simulation()으로 묶은 버전. 이후 LeoFL/Simulator/ 패키지로 발전.
import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn import preprocessing
from pandas import read_csv, read_feather

# ---- 위성/지구 상수 및 커버리지 계산 유틸 ----
from LeoFL.Leo.leosat import LEOSat, is_point_in_coverage

# 지상국 좌표(서울 시청 근처; 필요시 바꾸세요)
GS_LAT = 37.5665
GS_LON = 126.9780


# =========================
# 1) 모델 정의
# =========================
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


# =========================
# 2) 데이터셋 & 샘플링
# =========================
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

        # 라벨 인코딩
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

        # 클래스 가중치
        self.class_weight = torch.from_numpy(
            compute_class_weight("balanced", classes=np.unique(Y_series), y=Y_series)
        ).type(torch.float)

        # 피처 스케일링
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


def stratified_sample_fixed_counts(
    dataset, class_counts, rng: np.random.Generator, replace=True, shuffle=True
):
    """
    dataset.Y(클래스 인덱스, torch.long)를 기준으로
    class_counts = {cls_id: count, ...} 만큼 샘플링해 (X, Y) 반환.
    """
    Y_np = dataset.Y.numpy().astype(int)

    X_parts, Y_parts = [], []
    for cls_id, count in class_counts.items():
        idx = np.where(Y_np == cls_id)[0]
        if len(idx) == 0:
            raise ValueError(f"class {cls_id} 데이터가 없습니다.")
        sel = rng.choice(len(idx), size=count, replace=(replace or count > len(idx)))
        take = idx[sel]
        X_parts.append(dataset.X[take])
        Y_parts.append(dataset.Y[take])

    X_new = torch.cat(X_parts, dim=0)
    Y_new = torch.cat(Y_parts, dim=0)

    if shuffle:
        perm = rng.permutation(len(Y_new))
        X_new = X_new[perm]
        Y_new = Y_new[perm]

    return X_new, Y_new


def to_counts_dict(counts_array):
    """[c0, c1, ...] -> {0:c0, 1:c1, ...}"""
    return {i: int(c) for i, c in enumerate(counts_array)}


# =========================
# 3) 학습/평가 유틸
# =========================
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


# =========================
# 4) 집계 함수 (전략 선택 가능)
# =========================
def fedavg_aggregate_fn(global_model, client_models, participating_indices):
    """
    표준 FedAvg: 참여한 클라이언트들의 파라미터 평균.
    """
    if len(participating_indices) == 0:
        return
    with torch.no_grad():
        global_state = global_model.state_dict()
        for k in global_state.keys():
            global_state[k].zero_()

        for idx in participating_indices:
            local_state = client_models[idx].state_dict()
            for k in global_state.keys():
                global_state[k].add_(local_state[k])

        for k in global_state.keys():
            global_state[k].div_(float(len(participating_indices)))

        global_model.load_state_dict(global_state)


AGGREGATORS = {
    "fedavg": fedavg_aggregate_fn,
    # 나중에 "median": fedmedian_fn 이런 식으로 추가 가능
}


# =========================
# 5) 수렴(중단) 조건 함수
# =========================
def stop_on_target_accuracy(target):
    """
    정확도가 target 이상이면 True 반환하는 stop 함수 생성.
    """

    def _fn(minute, acc, best_acc):
        return acc >= target

    return _fn


def stop_never():
    """
    무한정 도는(정책적으로 num_rounds까지만) stop 함수.
    """

    def _fn(minute, acc, best_acc):
        return False

    return _fn


# =========================
# 6) 위성/궤도 유틸
# =========================
FIXED_CONSTELLATION = [
    {"h_km": 537, "incl_deg": 42, "offset_min": 4},
    {"h_km": 572, "incl_deg": 35, "offset_min": 15},
    {"h_km": 644, "incl_deg": 36, "offset_min": 70},
    {"h_km": 781, "incl_deg": 50, "offset_min": 20},
    {"h_km": 754, "incl_deg": 48, "offset_min": 74},
    {"h_km": 968, "incl_deg": 58, "offset_min": 118},
    {"h_km": 898, "incl_deg": 43, "offset_min": 9},
]


def build_fixed_constellation(configs):
    sats = []
    for i, c in enumerate(configs):
        offset_sec = int(round(c["offset_min"] * 60))
        sats.append(
            LEOSat(
                h_km=c["h_km"], incl_deg=c["incl_deg"], offset_sec=offset_sec, id=i + 1
            )
        )
    return sats


def compute_orbits_for_rounds(sat_list, num_rounds, seconds_per_round=60):
    times = np.arange(0, num_rounds * seconds_per_round, seconds_per_round)
    for sat in sat_list:
        sat.compute_leo_orbit(times)


def connected_clients_at_round(sat_list, round_idx, gs_lat=GS_LAT, gs_lon=GS_LON):
    connected = []
    for i, sat in enumerate(sat_list):
        pos = sat.get_leo_orbit_position(round_idx)
        inside, _, _ = is_point_in_coverage(pos, gs_lat, gs_lon)
        if inside:
            connected.append(i)
    return connected


def _minutes_to_rounds_list(training_minutes_list, seconds_per_round):
    return [
        int(math.ceil(m * 60.0 / float(seconds_per_round)))
        for m in training_minutes_list
    ]


# =========================
# 7) 이벤트/정확도 CSV 저장
# =========================
def _save_events_only_csv(events_log, log_dir, prefix):
    if not log_dir:
        log_dir = os.path.dirname(os.path.abspath(__file__))
    elif not os.path.isabs(log_dir):
        base = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base, log_dir)

    os.makedirs(log_dir, exist_ok=True)
    df = pd.DataFrame(events_log, columns=["time", "sat", "event"])
    path = os.path.join(log_dir, f"{prefix}_events_only.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _save_time_accuracy_csv(acc_log, save_csv_path: str):
    if not acc_log:
        print("accuracy log가 비어있어 CSV를 생성하지 않습니다.")
        return
    df = pd.DataFrame(acc_log, columns=["time", "accuracy"])
    df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")


# =========================
# 8) FL 시뮬레이터 함수들
# =========================
def federated_learning_always_connected(
    client_datasets,
    test_dataset,
    num_rounds=10000,
    local_epochs=1,
    device=None,
    hidden_size=100,
    lr=1e-3,
    train_minutes=10,
    log_dir=".",
    log_filename_prefix="sat_events_always",
    aggregate_fn=fedavg_aggregate_fn,
    stop_fn=None,
):
    """
    항상 지상국과 연결된다고 가정한 일반적인 FL.
    - aggregate_fn: 집계 함수 (ex. fedavg_aggregate_fn)
    - stop_fn: 수렴/중단 조건 함수 (minute, acc, best_acc) -> bool
    """
    assert len(client_datasets) > 0
    n_clients = len(client_datasets)
    if stop_fn is None:
        stop_fn = stop_never()

    input_size = test_dataset.n_features
    num_classes = test_dataset.n_classes
    global_model = DNN(
        input_size=input_size, hidden_size=hidden_size, num_classes=num_classes
    ).to(device)
    client_models = [copy.deepcopy(global_model).to(device) for _ in range(n_clients)]

    train_rounds_needed = int(train_minutes)
    global_version = 0
    downloaded_version = [-1] * n_clients
    remaining_train = [0] * n_clients
    trained_ready = [False] * n_clients
    uploaded_for_version = set()

    best_acc = -1.0
    acc_log = []
    acc_log.append((0, 0.0))

    for minute in range(num_rounds):
        # 다운링크
        for i in range(n_clients):
            if downloaded_version[i] < global_version:
                client_models[i].load_state_dict(
                    copy.deepcopy(global_model.state_dict())
                )
                downloaded_version[i] = global_version
                remaining_train[i] = train_rounds_needed
                trained_ready[i] = False

        # 로컬 학습
        for i in range(n_clients):
            if downloaded_version[i] == global_version and remaining_train[i] > 0:
                remaining_train[i] -= 1
                if remaining_train[i] == 0:
                    _ = train_one_epoch(
                        client_models[i],
                        client_datasets[i],
                        device=device,
                        batch_size=32,
                        lr=lr,
                        class_weight=client_datasets[i].class_weight,
                    )
                    trained_ready[i] = True
                    uploaded_for_version.add(i)

        # 전역 업데이트
        if len(uploaded_for_version) == n_clients:
            aggregate_fn(global_model, client_models, list(uploaded_for_version))
            acc = evaluate(global_model, test_dataset, device=device)
            acc_log.append((minute, acc))
            best_acc = max(best_acc, acc)

            if stop_fn(minute, acc, best_acc):
                acc_csv = os.path.join(
                    log_dir or ".", f"{log_filename_prefix}_accuracy.csv"
                )
                _save_time_accuracy_csv(acc_log, acc_csv)
                return global_model, best_acc

            global_version += 1
            uploaded_for_version.clear()
            for i in range(n_clients):
                remaining_train[i] = 0
                trained_ready[i] = False
                downloaded_version[i] = -1

    acc_csv = os.path.join(log_dir or ".", f"{log_filename_prefix}_accuracy.csv")
    _save_time_accuracy_csv(acc_log, acc_csv)
    return global_model, best_acc


def federated_learning_with_connectivity(  # class로 설계 파일 따로 만들기 -> 호출해서 쓰기
    client_datasets,
    test_dataset,
    num_rounds=10000,
    local_epochs=1,
    device=None,
    hidden_size=100,
    lr=1e-3,
    seconds_per_round=60,
    log_dir=None,
    log_filename_prefix="sat_events_base",
    train_minutes=10,
    aggregate_fn=fedavg_aggregate_fn,
    stop_fn=None,
    constellation_configs=FIXED_CONSTELLATION,
):
    """
    기존 BASE 시나리오:
    - 모든 위성 동일 학습 시간(train_minutes)
    - LEO 연결 기반
    """
    assert len(client_datasets) > 0
    n_clients = len(client_datasets)
    if stop_fn is None:
        stop_fn = stop_never()

    acc_log = []
    acc_log.append((0, 0.0))

    input_size = test_dataset.n_features
    num_classes = test_dataset.n_classes
    global_model = DNN(
        input_size=input_size, hidden_size=hidden_size, num_classes=num_classes
    ).to(device)
    client_models = [copy.deepcopy(global_model).to(device) for _ in range(n_clients)]

    sats = build_fixed_constellation(constellation_configs)
    compute_orbits_for_rounds(
        sats, num_rounds=num_rounds, seconds_per_round=seconds_per_round
    )

    TRAIN_MINUTES = train_minutes
    train_rounds_needed = TRAIN_MINUTES

    global_version = 0
    downloaded_version = [-1] * n_clients
    uploaded_for_version = set()

    remaining_train = [0] * n_clients
    trained_ready = [False] * n_clients

    best_acc = -1.0

    events_log = []

    def log(i, t, label):
        events_log.append((t, f"sat_{i + 1}", label))

    prev_connected = [False] * n_clients

    for minute in range(num_rounds):
        connected = connected_clients_at_round(sats, minute, GS_LAT, GS_LON)

        for i in range(n_clients):
            is_conn = i in connected
            if is_conn and not prev_connected[i]:
                log(i, minute, "GS connect")
            prev_connected[i] = is_conn

        # 다운링크
        for i in connected:
            if downloaded_version[i] < global_version:
                client_models[i].load_state_dict(
                    copy.deepcopy(global_model.state_dict())
                )
                downloaded_version[i] = global_version
                remaining_train[i] = train_rounds_needed
                trained_ready[i] = False
                log(i, minute, "get_global")

        # 로컬학습
        for i in range(n_clients):
            if downloaded_version[i] == global_version and remaining_train[i] > 0:
                remaining_train[i] -= 1
                if remaining_train[i] == 0:
                    _ = train_one_epoch(
                        client_models[i],
                        client_datasets[i],
                        device=device,
                        batch_size=32,
                        lr=lr,
                        class_weight=client_datasets[i].class_weight,
                    )
                    trained_ready[i] = True
                    log(i, minute, "local")

        # 업링크
        for i in connected:
            if (
                (downloaded_version[i] == global_version)
                and trained_ready[i]
                and (i not in uploaded_for_version)
            ):
                uploaded_for_version.add(i)

        # 전역 업데이트
        if len(uploaded_for_version) == n_clients:
            aggregate_fn(global_model, client_models, list(uploaded_for_version))
            acc = evaluate(global_model, test_dataset, device=device)
            acc_log.append((minute, acc))
            best_acc = max(best_acc, acc)

            if stop_fn(minute, acc, best_acc):
                _save_time_accuracy_csv(
                    acc_log,
                    os.path.join(log_dir or ".", f"{log_filename_prefix}_accuracy.csv"),
                )
                _save_events_only_csv(events_log, log_dir, log_filename_prefix)
                return global_model, best_acc

            global_version += 1
            uploaded_for_version.clear()
            for i in range(n_clients):
                remaining_train[i] = 0
                trained_ready[i] = False

    _save_events_only_csv(events_log, log_dir, log_filename_prefix)
    _save_time_accuracy_csv(
        acc_log, os.path.join(log_dir or ".", f"{log_filename_prefix}_accuracy.csv")
    )
    return global_model, best_acc


def federated_learning_with_connectivity_vartrain(
    client_datasets,
    test_dataset,
    training_minutes_list,
    num_rounds=10000,
    local_epochs=1,
    device=None,
    hidden_size=100,
    lr=1e-3,
    seconds_per_round=60,
    log_dir=None,
    log_filename_prefix="sat_events_vartrain",
    aggregate_fn=fedavg_aggregate_fn,
    stop_fn=None,
    constellation_configs=FIXED_CONSTELLATION,
):
    """
    VAR 시나리오:
    - 위성별 학습시간이 training_minutes_list로 다름
    """
    acc_log = []
    acc_log.append((0, 0.0))
    assert len(client_datasets) > 0
    n_clients = len(client_datasets)
    assert len(training_minutes_list) == n_clients
    if stop_fn is None:
        stop_fn = stop_never()

    input_size = test_dataset.n_features
    num_classes = test_dataset.n_classes
    global_model = DNN(
        input_size=input_size, hidden_size=hidden_size, num_classes=num_classes
    ).to(device)
    client_models = [copy.deepcopy(global_model).to(device) for _ in range(n_clients)]

    sats = build_fixed_constellation(constellation_configs)
    compute_orbits_for_rounds(
        sats, num_rounds=num_rounds, seconds_per_round=seconds_per_round
    )

    train_rounds_needed = _minutes_to_rounds_list(
        training_minutes_list, seconds_per_round
    )

    global_version = 0
    downloaded_version = [-1] * n_clients
    uploaded_for_version = set()
    remaining_train = [0] * n_clients
    trained_ready = [False] * n_clients
    best_acc = -1.0

    events_log = []

    def log(i, t, label):
        events_log.append((t, f"sat_{i + 1}", label))

    prev_connected = [False] * n_clients

    for minute in range(num_rounds):
        connected = connected_clients_at_round(sats, minute, GS_LAT, GS_LON)
        connected = [int(ci) for ci in connected if 0 <= int(ci) < n_clients]

        for i in range(n_clients):
            is_conn = i in connected
            if is_conn and not prev_connected[i]:
                log(i, minute, "GS connect")
            prev_connected[i] = is_conn

        # 다운링크
        for i in connected:
            if downloaded_version[i] < global_version:
                client_models[i].load_state_dict(
                    copy.deepcopy(global_model.state_dict())
                )
                downloaded_version[i] = global_version
                remaining_train[i] = train_rounds_needed[i]
                trained_ready[i] = False
                log(i, minute, "get_global")

        # 로컬학습
        for i in range(n_clients):
            if downloaded_version[i] == global_version and remaining_train[i] > 0:
                remaining_train[i] -= 1
                if remaining_train[i] == 0:
                    _ = train_one_epoch(
                        client_models[i],
                        client_datasets[i],
                        device=device,
                        batch_size=32,
                        lr=lr,
                        class_weight=client_datasets[i].class_weight,
                    )
                    trained_ready[i] = True
                    log(i, minute, "local")

        # 업링크
        for i in connected:
            if (
                (downloaded_version[i] == global_version)
                and trained_ready[i]
                and (i not in uploaded_for_version)
            ):
                uploaded_for_version.add(i)

        if len(uploaded_for_version) == n_clients:
            aggregate_fn(global_model, client_models, list(uploaded_for_version))
            acc = evaluate(global_model, test_dataset, device=device)
            acc_log.append((minute, acc))
            best_acc = max(best_acc, acc)

            if stop_fn(minute, acc, best_acc):
                _save_time_accuracy_csv(
                    acc_log,
                    os.path.join(log_dir or ".", f"{log_filename_prefix}_accuracy.csv"),
                )
                _save_events_only_csv(events_log, log_dir, log_filename_prefix)
                return global_model, best_acc

            global_version += 1
            uploaded_for_version.clear()
            for i in range(n_clients):
                remaining_train[i] = 0
                trained_ready[i] = False

    _save_events_only_csv(events_log, log_dir, log_filename_prefix)
    _save_time_accuracy_csv(
        acc_log, os.path.join(log_dir or ".", f"{log_filename_prefix}_accuracy.csv")
    )
    return global_model, best_acc


# =========================
# 9) 전체 시뮬레이터 엔트리포인트 예시
# =========================
def run_simulation(
    mode: str,
    client_datasets,
    test_dataset,
    num_rounds=20000,
    per_sat_minutes=None,
    train_minutes=10,
    agg_name="fedavg",
    target_acc=70.0,
    device=None,
    hidden_size=100,
    lr=1e-3,
    seconds_per_round=60,
    log_dir=".",
    log_prefix_base="sat_events_base",
    log_prefix_var="sat_events_vartrain",
    log_prefix_always="sat_events_always",
    constellation_configs=FIXED_CONSTELLATION,
):
    """
    mode: "base", "vartrain", "always"
    agg_name: AGGREGATORS에서 선택 (예: "fedavg")
    """
    aggregate_fn = AGGREGATORS[agg_name]
    stop_fn = stop_on_target_accuracy(target_acc)

    if mode == "base":
        return federated_learning_with_connectivity(
            client_datasets=client_datasets,
            test_dataset=test_dataset,
            num_rounds=num_rounds,
            local_epochs=1,
            device=device,
            hidden_size=hidden_size,
            lr=lr,
            seconds_per_round=seconds_per_round,
            log_dir=log_dir,
            log_filename_prefix=log_prefix_base,
            train_minutes=train_minutes,
            aggregate_fn=aggregate_fn,
            stop_fn=stop_fn,
            constellation_configs=constellation_configs,
        )
    elif mode == "vartrain":
        if per_sat_minutes is None:
            raise ValueError("VAR 모드에서는 per_sat_minutes가 필요합니다.")
        return federated_learning_with_connectivity_vartrain(
            client_datasets=client_datasets,
            test_dataset=test_dataset,
            training_minutes_list=per_sat_minutes,
            num_rounds=num_rounds,
            local_epochs=1,
            device=device,
            hidden_size=hidden_size,
            lr=lr,
            seconds_per_round=seconds_per_round,
            log_dir=log_dir,
            log_filename_prefix=log_prefix_var,
            aggregate_fn=aggregate_fn,
            stop_fn=stop_fn,
            constellation_configs=constellation_configs,
        )
    elif mode == "always":
        return federated_learning_always_connected(
            client_datasets=client_datasets,
            test_dataset=test_dataset,
            num_rounds=num_rounds,
            local_epochs=1,
            device=device,
            hidden_size=hidden_size,
            lr=lr,
            train_minutes=train_minutes,
            log_dir=log_dir,
            log_filename_prefix=log_prefix_always,
            aggregate_fn=aggregate_fn,
            stop_fn=stop_fn,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
