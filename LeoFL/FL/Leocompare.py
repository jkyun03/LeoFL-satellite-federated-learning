# [초기 프로토타입] 항상 연결 / 간헐 연결 / 간헐 연결+학습시간 상이 3가지 FL 시나리오를 UNR-IDD로 비교한다.
import copy
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn import preprocessing
from pandas import read_csv, read_feather
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ---- 위성/지구 상수 및 커버리지 계산 유틸 ----
from LeoFL.Leo.leosat import LEOSat, is_point_in_coverage

# 지상국 좌표(서울 시청 근처; 필요시 바꾸세요)
GS_LAT = 37.5665
GS_LON = 126.9780


# =========================
# 1) 모델 정의 (단일 파일)
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

        # 라벨 인코딩(범주 → 정수)
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

        # 클래스 가중치(필요시 사용)
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


SAT_CLASS_COUNTS = [
    [441, 270, 200, 62, 519, 508],  # sat_1
    [475, 327, 192, 50, 520, 436],  # sat_2
    [451, 288, 238, 55, 487, 481],  # sat_3
    [466, 304, 205, 51, 496, 478],  # sat_4
    [437, 288, 196, 64, 537, 478],  # sat_5
    [446, 293, 216, 57, 488, 500],  # sat_6
    [451, 306, 181, 54, 527, 481],  # sat_7
]


def stratified_sample_fixed_counts(
    dataset, class_counts, rng: np.random.Generator, replace=True, shuffle=True
):
    """
    dataset.Y(클래스 인덱스, torch.long)를 기준으로
    class_counts = {cls_id: count, ...} 만큼 샘플링해 (X, Y) 반환.

    - replace=True: 원본 클래스 샘플 수 < count 일 때 중복 허용해 맞춤
    - shuffle=True: 최종 합친 뒤 전체를 섞어서 반환
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
    """[c0, c1, ...] -> {0:c0, 1:c1, ...} 로 변환"""
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


def fedavg_aggregate(global_model, client_models, participating_indices):
    """표준 FedAvg: 참여한 클라이언트들의 파라미터 평균."""
    if len(participating_indices) == 0:
        return  # 아무도 업링크 못함 → 유지
    # 합 초기화
    with torch.no_grad():
        # 글로벌 파라미터 0으로 초기화 후 누산
        global_state = global_model.state_dict()
        for k in global_state.keys():
            global_state[k].zero_()

        for idx in participating_indices:
            local_state = client_models[idx].state_dict()
            for k in global_state.keys():
                global_state[k].add_(local_state[k])

        # 평균으로 스케일
        for k in global_state.keys():
            global_state[k].div_(float(len(participating_indices)))

        global_model.load_state_dict(global_state)


FIXED_CONSTELLATION = [
    #  [위성 01] 고도=537 km, 경사각=42°, 오프셋=235s ≈ 4 min
    {"h_km": 537, "incl_deg": 42, "offset_min": 4},
    #  [위성 02] 905s ≈ 15 min
    {"h_km": 572, "incl_deg": 35, "offset_min": 15},
    #  [위성 03] 4225s ≈ 70 min
    {"h_km": 644, "incl_deg": 36, "offset_min": 70},
    #  [위성 04] 1202s ≈ 20 min
    {"h_km": 781, "incl_deg": 50, "offset_min": 20},
    #  [위성 05] 4453s ≈ 74 min
    {"h_km": 754, "incl_deg": 48, "offset_min": 74},
    #  [위성 06] 7051s ≈ 118 min
    {"h_km": 968, "incl_deg": 58, "offset_min": 118},
    #  [위성 07] 562s ≈ 9 min
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


# =========================
# 4) 위성 연결 스케줄링
# =========================


def compute_orbits_for_rounds(sat_list, num_rounds, seconds_per_round=60):
    """매 라운드를 1분 간격의 '프레임'으로 보고 위성 궤도 위치를 미리 계산."""
    times = np.arange(0, num_rounds * seconds_per_round, seconds_per_round)
    for sat in sat_list:
        sat.compute_leo_orbit(times)  # 내부에 x,y,z 저장됨


def connected_clients_at_round(sat_list, round_idx, gs_lat=GS_LAT, gs_lon=GS_LON):
    """라운드 시점에 지상국 커버리지 내에 있는 위성 인덱스 반환."""
    connected = []
    for i, sat in enumerate(sat_list):
        pos = sat.get_leo_orbit_position(round_idx)  # (x,y,z) [m]
        inside, _, _ = is_point_in_coverage(pos, gs_lat, gs_lon)
        if inside:
            connected.append(i)
    return connected


# =========================
# 5) 연합학습 (이벤트 구동 · 전역 업데이트 시에만 출력 · 조기종료 70%)
# =========================


def federated_learning_always_connected(
    client_datasets,
    test_dataset,
    num_rounds=10000,  # 최대 시뮬레이션 분
    local_epochs=1,
    device=None,
    hidden_size=100,
    lr=1e-3,
    train_minutes=10,  # 한 라운드당(=한 글로벌 버전) 로컬학습에 필요한 분
    log_dir=".",
    log_filename_prefix="sat_events_always",  # 저장 파일 접두어
    plot_accuracy=False,
    plot_path=None,
):
    """
    항상 지상국과 연결된다고 가정한 일반적인 FL.
    - 모든 클라이언트가 매 글로벌 버전마다 동일한 전역모델을 즉시 받음(get_global)
    - 각 클라이언트는 train_minutes 만큼 시간이 흐른 뒤 1 epoch 학습(local)
    - 모든 클라이언트가 학습 완료되면 FedAvg 수행 → 다음 글로벌 버전
    """
    assert len(client_datasets) > 0
    n_clients = len(client_datasets)

    # ---- 모델 초기화 ----
    input_size = test_dataset.n_features
    num_classes = test_dataset.n_classes
    global_model = DNN(
        input_size=input_size, hidden_size=hidden_size, num_classes=num_classes
    ).to(device)
    client_models = [copy.deepcopy(global_model).to(device) for _ in range(n_clients)]

    # ---- 상태 변수 ----
    train_rounds_needed = int(train_minutes)  # 1분 단위 시뮬
    global_version = 0
    downloaded_version = [-1] * n_clients
    remaining_train = [0] * n_clients
    trained_ready = [False] * n_clients
    uploaded_for_version = set()  # 항상 연결이라 업링크=학습완료와 동치로 사용

    best_acc = -1.0
    acc_log = []
    acc_log.append((0, 0.0))

    for minute in range(num_rounds):
        # 항상 연결되어 있으므로, 새 글로벌 버전이면 즉시 다운링크
        for i in range(n_clients):
            if downloaded_version[i] < global_version:
                client_models[i].load_state_dict(
                    copy.deepcopy(global_model.state_dict())
                )
                downloaded_version[i] = global_version
                remaining_train[i] = train_rounds_needed
                trained_ready[i] = False

        # 로컬 학습 카운트다운
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
                    uploaded_for_version.add(i)  # 항상 연결 → 곧바로 업링크했다고 간주

        # 모든 클라가 업로드했으면 전역 업데이트
        if len(uploaded_for_version) == n_clients:
            fedavg_aggregate(global_model, client_models, list(uploaded_for_version))
            acc = evaluate(global_model, test_dataset, device=device)
            acc_log.append((minute, acc))
            best_acc = max(best_acc, acc)
            # print(f" Global Update (FedRound {global_version+1}) at {minute} min  "
            # f"Accuracy: {acc:.2f}% (Best: {best_acc:.2f}%)")

            # 조기 종료 기준(원하면 조정/삭제 가능)
            if acc >= 70.0:
                acc_csv = os.path.join(
                    log_dir or ".", f"{log_filename_prefix}_accuracy.csv"
                )
                _save_time_accuracy_csv(acc_log, acc_csv)
                if plot_accuracy:
                    out_png = plot_path or os.path.join(
                        log_dir or ".", f"{log_filename_prefix}_accuracy.png"
                    )

                return global_model, best_acc

            # 다음 글로벌 버전 준비
            global_version += 1
            uploaded_for_version.clear()
            for i in range(n_clients):
                remaining_train[i] = 0
                trained_ready[i] = False
                downloaded_version[i] = -1  # 새 전역모델을 다시 받도록 리셋

    # 종료 시 로그 저장
    acc_csv = os.path.join(log_dir or ".", f"{log_filename_prefix}_accuracy.csv")
    _save_time_accuracy_csv(acc_log, acc_csv)
    if plot_accuracy:
        out_png = plot_path or os.path.join(
            log_dir or ".", f"{log_filename_prefix}_accuracy.png"
        )

    return global_model, best_acc


def federated_learning_with_connectivity(
    client_datasets,
    test_dataset,
    num_rounds=10000,  # 최대 시뮬레이션 시간 (분)
    local_epochs=1,
    device=None,
    hidden_size=100,
    lr=1e-3,
    seconds_per_round=60,  # 1 스텝 = 60초 = 1분
    log_dir=None,
    log_filename_prefix="sat_events",
):
    """
    - 시뮬레이션 단위: 1분
    - 학습시간: 10분(=10 스텝)
    - 전역 업데이트(FedAvg)는 10개 위성 업링크 후에만 수행
    - Accuracy >= 70%면 즉시 종료
    - 이벤트는 (time[min], sat, event)만 기록: 'GS connect'(연결 시작), 'get_global', 'local'(+원하면 'uplink')
    """
    assert len(client_datasets) > 0
    n_clients = len(client_datasets)
    acc_log = []
    acc_log.append((0, 0.0))

    # ---- 모델 초기화 ----
    input_size = test_dataset.n_features
    num_classes = test_dataset.n_classes
    global_model = DNN(
        input_size=input_size, hidden_size=hidden_size, num_classes=num_classes
    ).to(device)
    client_models = [copy.deepcopy(global_model).to(device) for _ in range(n_clients)]

    # ---- 궤도 계산(분 단위) ----
    sats = build_fixed_constellation(FIXED_CONSTELLATION)
    compute_orbits_for_rounds(
        sats, num_rounds=num_rounds, seconds_per_round=seconds_per_round
    )

    # ---- 상태 변수 ----
    TRAIN_MINUTES = 10
    train_rounds_needed = TRAIN_MINUTES  # 1분 단위 → 5라운드 학습

    global_version = 0
    downloaded_version = [-1] * n_clients
    uploaded_for_version = set()

    remaining_train = [0] * n_clients
    trained_ready = [False] * n_clients

    best_acc = -1.0

    # ---- 이벤트 로그 ----
    events_log = []

    def log(i, t, label):
        # i: 0-based index, 저장은 sat_1..sat_n
        events_log.append((t, f"sat_{i + 1}", label))

    #  연결 상태 추적(직전 스텝 연결 여부)
    prev_connected = [False] * n_clients

    # ---- 시뮬레이션 ----
    for minute in range(num_rounds):
        connected = connected_clients_at_round(sats, minute, GS_LAT, GS_LON)

        #  연결 '시작' 이벤트만 기록 (off → on 전이 시에만)
        for i in range(n_clients):
            is_conn = i in connected
            if is_conn and not prev_connected[i]:
                log(i, minute, "GS connect")
            prev_connected[i] = is_conn

        # (다운링크) 전역모델 처음 수신
        for i in connected:
            if downloaded_version[i] < global_version:
                client_models[i].load_state_dict(
                    copy.deepcopy(global_model.state_dict())
                )
                downloaded_version[i] = global_version
                remaining_train[i] = train_rounds_needed
                trained_ready[i] = False
                log(i, minute, "get_global")

        # (로컬학습 진행)
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

        # (업링크)
        for i in connected:
            if (
                (downloaded_version[i] == global_version)
                and trained_ready[i]
                and (i not in uploaded_for_version)
            ):
                uploaded_for_version.add(i)
                # log(i, minute, "uplink")  # 필요하면 주석 해제

                # (전역 업데이트)
        if len(uploaded_for_version) == n_clients:
            fedavg_aggregate(global_model, client_models, list(uploaded_for_version))
            acc = evaluate(global_model, test_dataset, device=device)
            acc_log.append((minute, acc))
            best_acc = max(best_acc, acc)
            """print(f" Global Update (FedRound {global_version+1}) at {minute} min  "
                  f"Accuracy: {acc:.2f}% (Best: {best_acc:.2f}%)")
            """
            if acc >= 70.0:
                #  조기 종료 시에도 정확도 CSV 저장
                _save_time_accuracy_csv(
                    acc_log,
                    os.path.join(log_dir or ".", f"{log_filename_prefix}_accuracy.csv"),
                )
                return global_model, best_acc

            global_version += 1
            uploaded_for_version.clear()
            for i in range(n_clients):
                remaining_train[i] = 0
                trained_ready[i] = False
                # (선택) 이벤트 CSV도 필요 없으면 아래 줄은 주석 처리
    _save_events_only_csv(events_log, log_dir, log_filename_prefix)
    return global_model, best_acc


def _minutes_to_rounds_list(training_minutes_list, seconds_per_round):
    # 분 → 라운드 (ceil)
    return [
        int(math.ceil(m * 60.0 / float(seconds_per_round)))
        for m in training_minutes_list
    ]


def federated_learning_with_connectivity_vartrain(
    client_datasets,
    test_dataset,
    training_minutes_list,  # ← 위성별 학습시간(분) 필수
    num_rounds=10000,
    local_epochs=1,
    device=None,
    hidden_size=100,
    lr=1e-3,
    seconds_per_round=60,
    log_dir=None,
    log_filename_prefix="sat_events_vartrain",  # ← 디폴트로 별도 파일명
):
    """
    기존 로직 동일. 차이점:
    - get_global 시점에 remaining_train[i] = per-sat 학습 라운드로 세팅
    """
    acc_log = []
    acc_log.append((0, 0.0))
    assert len(client_datasets) > 0
    n_clients = len(client_datasets)
    assert len(training_minutes_list) == n_clients, (
        f"training_minutes_list 길이({len(training_minutes_list)}) != n_clients({n_clients})"
    )

    # ---- 모델 초기화 (기존과 동일) ----
    input_size = test_dataset.n_features
    num_classes = test_dataset.n_classes
    global_model = DNN(
        input_size=input_size, hidden_size=hidden_size, num_classes=num_classes
    ).to(device)
    client_models = [copy.deepcopy(global_model).to(device) for _ in range(n_clients)]

    # ---- 궤도 계산 (기존과 동일) ----
    sats = build_fixed_constellation(FIXED_CONSTELLATION)
    compute_orbits_for_rounds(
        sats, num_rounds=num_rounds, seconds_per_round=seconds_per_round
    )

    # ---- 위성별 학습 라운드 수 ----
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
        # 안전 가드(개수 불일치 대비)
        connected = [int(ci) for ci in connected if 0 <= int(ci) < n_clients]

        # 연결 시작 이벤트만 기록
        for i in range(n_clients):
            is_conn = i in connected
            if is_conn and not prev_connected[i]:
                log(i, minute, "GS connect")
            prev_connected[i] = is_conn

        # (다운링크) 전역모델 수신 → 위성별 학습시간으로 카운트다운 설정
        for i in connected:
            if downloaded_version[i] < global_version:
                client_models[i].load_state_dict(
                    copy.deepcopy(global_model.state_dict())
                )
                downloaded_version[i] = global_version
                remaining_train[i] = train_rounds_needed[i]  # ★ 위성별 학습시간 반영
                trained_ready[i] = False
                log(i, minute, "get_global")

        # (로컬학습) 카운트다운 0 되면 1 epoch 학습
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

        # (업링크)
        for i in connected:
            if (
                (downloaded_version[i] == global_version)
                and trained_ready[i]
                and (i not in uploaded_for_version)
            ):
                uploaded_for_version.add(i)

        if len(uploaded_for_version) == n_clients:
            fedavg_aggregate(global_model, client_models, list(uploaded_for_version))
            acc = evaluate(global_model, test_dataset, device=device)
            acc_log.append((minute, acc))
            best_acc = max(best_acc, acc)
            """print(f" Global Update (FedRound {global_version+1}) at {minute} min  "
                  f"Accuracy: {acc:.2f}% (Best: {best_acc:.2f}%)")"""

            if acc >= 70.0:
                # ✅ 조기 종료 시에도 정확도 CSV 저장
                _save_time_accuracy_csv(
                    acc_log,
                    os.path.join(log_dir or ".", f"{log_filename_prefix}_accuracy.csv"),
                )
                return global_model, best_acc

            global_version += 1
            uploaded_for_version.clear()
            for i in range(n_clients):
                remaining_train[i] = 0
                trained_ready[i] = False

                # (선택) 이벤트 CSV도 필요 없으면 아래 줄은 주석 처리
    _save_events_only_csv(events_log, log_dir, log_filename_prefix)
    return global_model, best_acc


def _save_events_only_csv(events_log, log_dir, prefix):
    """이벤트 발생 시점만 (time[min], sat, event) CSV로 저장."""
    if not log_dir:
        # None 또는 "" 이면 현재 파일 옆에 저장
        log_dir = os.path.dirname(os.path.abspath(__file__))
    elif not os.path.isabs(log_dir):
        # 상대경로면 현재 파일 기준으로 변환
        base = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base, log_dir)

    os.makedirs(log_dir, exist_ok=True)
    df = pd.DataFrame(events_log, columns=["time", "sat", "event"])
    path = os.path.join(log_dir, f"{prefix}_events_only.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    # print(f" Event log saved: {path}  (rows={len(df)})")


# =========================
# 6 ) matplotlib
# =========================


def _save_time_accuracy_csv(acc_log, save_csv_path: str):
    """(minute, accuracy) 튜플 리스트를 CSV로 저장"""
    if not acc_log:
        print(" accuracy log가 비어있어 CSV를 생성하지 않습니다.")
        return
    df = pd.DataFrame(acc_log, columns=["time", "accuracy"])
    df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
    # print(f" accuracy curve saved: {save_csv_path} (rows={len(df)})")


def plot_accuracy_over_time(
    base_csv: str,
    var_csv: str,
    always_csv: str = None,
    labels=("BASE", "VAR", "ALWAYS"),
    save_path: str = None,
):
    if not os.path.exists(base_csv) or not os.path.exists(var_csv):
        raise FileNotFoundError("정확도 CSV 파일이 존재하지 않습니다.")

    df_base = pd.read_csv(base_csv)
    df_var = pd.read_csv(var_csv)
    df_alw = (
        pd.read_csv(always_csv) if always_csv and os.path.exists(always_csv) else None
    )

    # ---- 메인 플롯 ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_base["time"], df_base["accuracy"], label=labels[0], linewidth=2)
    ax.plot(df_var["time"], df_var["accuracy"], label=labels[1], linewidth=2)
    if df_alw is not None:
        ax.plot(df_alw["time"], df_alw["accuracy"], label=labels[2], linewidth=2)

    ax.set_title("Accuracy over Time (BASE vs VAR vs ALWAYS)")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(left=0, right=3500)
    ax.set_ylim(0, 70)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    # ax.set_xticks([0, 500, 1500, 2500, 3500])

    # ---- ★ inset(0~500분 확대) 추가 ----
    axins = inset_axes(
        ax,
        width="40%",
        height="45%",
        loc="lower left",
        bbox_to_anchor=(0.08, 0.08, 1, 1),
        bbox_transform=ax.transAxes,
    )

    axins.plot(df_base["time"], df_base["accuracy"], linewidth=2, color="tab:blue")
    axins.plot(df_var["time"], df_var["accuracy"], linewidth=2, color="tab:orange")
    if df_alw is not None:
        axins.plot(df_alw["time"], df_alw["accuracy"], linewidth=2, color="tab:green")

    # 확대 구간 지정
    axins.set_xlim(0, 500)
    axins.set_ylim(0, 70)
    axins.set_title("Zoom: 0–500 min", fontsize=10)
    axins.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"그래프 저장 완료: {save_path}")
    else:
        plt.show()


# =========================
# 7) 메인 (이벤트 기반 · Zipf 제거 · 조기종료 포함)
# =========================
if __name__ == "__main__":

    def set_global_seed(seed: int) -> None:
        """Fix all stochastic sources so dataset splits stay identical run-to-run."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    GLOBAL_SEED = 20240301
    set_global_seed(GLOBAL_SEED)

    # ---- 하이퍼파라미터 ----
    num_clients = len(FIXED_CONSTELLATION)
    max_simulation_minutes = 20000  # 최대 시뮬레이션 '분' (가드)
    local_epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """print("=== Fixed LEO Constellation (phase offset in minutes) ===")
    for i, c in enumerate(FIXED_CONSTELLATION, start=1):
        print(f"[sat_{i:02d}] h={c['h_km']} km, inc={c['incl_deg']}°, phase_offset={c['offset_min']} min")"""

    # ---- 데이터 경로 ----
    train_path = "LeoFL/UNR-IDD/UNR-IDD_train.feather"
    test_path = "LeoFL/UNR-IDD/UNR-IDD_test.feather"

    # ---- 공통 인코더/스케일러 설정 ----
    central_train = NIDDataset(train_path, bBinary=False)

    # ---- 위성별 균등 샘플 데이터셋 (2000개씩) ----
    client_datasets = []
    for i, counts in enumerate(SAT_CLASS_COUNTS, start=1):
        d = NIDDataset(
            train_path,
            bBinary=False,
            category_encoder=central_train.category_encoder,
            scaler=central_train.scaler,
        )

        # 위성별로 샘플링은 랜덤(클래스 내부만), '분포'는 고정
        rng = np.random.default_rng(GLOBAL_SEED + i)

        counts_dict = to_counts_dict(counts)
        assert sum(counts_dict.values()) == 2000, f"sat_{i} 총합이 2000이 아님"

        X_new, Y_new = stratified_sample_fixed_counts(
            d, counts_dict, rng, replace=True, shuffle=True
        )
        d.X, d.Y = X_new, Y_new

        binc = np.bincount(d.Y.numpy().astype(int), minlength=len(counts))
        # print(f"위성 {i} Dataset → {len(d.Y)} samples, class dist = {binc}")
        client_datasets.append(d)
    # ---- 테스트 데이터셋 ----
    test_dataset = NIDDataset(
        test_path,
        bBinary=False,
        category_encoder=central_train.category_encoder,
        scaler=central_train.scaler,
    )

    # (권장) 위성/클라이언트 개수 일치 확인
    assert len(client_datasets) == len(FIXED_CONSTELLATION)

    # ===== 실험 1: 기존 코드 (학습시간 고정; 네 현재 함수 그대로) =====
    _, best_acc_base = federated_learning_with_connectivity(
        client_datasets=client_datasets,
        test_dataset=test_dataset,
        num_rounds=max_simulation_minutes,
        local_epochs=local_epochs,
        device=device,
        hidden_size=100,
        lr=1e-3,
        seconds_per_round=60,
        log_dir=".",
        log_filename_prefix="sat_events_base",  # ← CSV: sat_events_base_events_only.csv
    )

    # ===== 실험 2: 위성별 학습시간 다르게 =====
    per_sat_minutes = [10, 20, 30, 40, 50, 60, 70]  # 위성 개수만큼
    assert len(per_sat_minutes) == len(client_datasets)

    _, best_acc_var = federated_learning_with_connectivity_vartrain(
        client_datasets=client_datasets,
        test_dataset=test_dataset,
        training_minutes_list=per_sat_minutes,  # ★ 여기만 추가
        num_rounds=max_simulation_minutes,
        local_epochs=local_epochs,
        device=device,
        hidden_size=100,
        lr=1e-3,
        seconds_per_round=60,
        log_dir=".",
        log_filename_prefix="sat_events_vartrain",  # ← CSV: sat_events_vartrain_events_only.csv
    )
    # ===== 실험 3: 항상 지상국과 연결 (일반 FL) =====
    _, best_acc_always = federated_learning_always_connected(
        client_datasets=client_datasets,
        test_dataset=test_dataset,
        num_rounds=max_simulation_minutes,
        local_epochs=local_epochs,
        device=device,
        hidden_size=100,
        lr=1e-3,
        train_minutes=10,  # BASE와 공정 비교용
        log_dir=".",
        log_filename_prefix="sat_events_always",
    )

    print("\n=== Comparison ===")
    print(f"[BASE] Best Accuracy: {best_acc_base:.2f}%")
    print(f"[VAR ] Best Accuracy: {best_acc_var:.2f}%")
    # === ✅ 여기부터 방법 A 추가 ===
    LOG_DIR = os.path.dirname(
        os.path.abspath(__file__)
    )  # 현재 파일(Leocompare.py) 폴더
    BASE_EVENTS = os.path.join(LOG_DIR, "sat_events_base_events_only.csv")
    VAR_EVENTS = os.path.join(LOG_DIR, "sat_events_vartrain_events_only.csv")

    plot_accuracy_over_time(
        base_csv="sat_events_base_accuracy.csv",
        var_csv="sat_events_vartrain_accuracy.csv",
        always_csv="sat_events_always_accuracy.csv",
        labels=("BASE", "VAR", "ALWAYS"),
        save_path="base_var_always_accuracy.png",
    )
