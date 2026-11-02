# [초기 프로토타입] 고정 위성 구성에서 간헐 연결 + 동일 학습시간 조건의 FL을 실험한다.
import copy
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    - 학습시간: 5분(=5 스텝)
    - 전역 업데이트(FedAvg)는 10개 위성 업링크 후에만 수행
    - Accuracy >= 70%면 즉시 종료
    - 이벤트는 (time[min], sat, event)만 기록: 'GS connect'(연결 시작), 'get_global', 'local'(+원하면 'uplink')
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

    # ---- 궤도 계산(분 단위) ----
    sats = build_fixed_constellation(FIXED_CONSTELLATION)
    compute_orbits_for_rounds(
        sats, num_rounds=num_rounds, seconds_per_round=seconds_per_round
    )

    # ---- 상태 변수 ----
    TRAIN_MINUTES = 5
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
            best_acc = max(best_acc, acc)
            print(
                f" Global Update (FedRound {global_version + 1}) at {minute} min  "
                f"Accuracy: {acc:.2f}% (Best: {best_acc:.2f}%)"
            )

            if acc >= 70.0:
                _save_events_only_csv(events_log, log_dir, log_filename_prefix)
                return global_model, best_acc

            global_version += 1
            uploaded_for_version.clear()
            for i in range(n_clients):
                remaining_train[i] = 0
                trained_ready[i] = False

    # 종료 후 로그 저장
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
    print(f" Event log saved: {path}  (rows={len(df)})")


# =========================
# 6) 메인 (이벤트 기반 · Zipf 제거 · 조기종료 포함)
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

    print("=== Fixed LEO Constellation (phase offset in minutes) ===")
    for i, c in enumerate(FIXED_CONSTELLATION, start=1):
        print(
            f"[sat_{i:02d}] h={c['h_km']} km, inc={c['incl_deg']}°, phase_offset={c['offset_min']} min"
        )

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
        print(f"위성 {i} Dataset → {len(d.Y)} samples, class dist = {binc}")
        client_datasets.append(d)
    # ---- 테스트 데이터셋 ----
    test_dataset = NIDDataset(
        test_path,
        bBinary=False,
        category_encoder=central_train.category_encoder,
        scaler=central_train.scaler,
    )

    # ---- 연합학습 (이벤트 구동 기반 + 정확도 70% 도달 시 종료) ----
    _, best_acc = federated_learning_with_connectivity(
        client_datasets=client_datasets,
        test_dataset=test_dataset,
        num_rounds=max_simulation_minutes,
        local_epochs=local_epochs,
        device=device,
        hidden_size=100,
        lr=1e-3,
        seconds_per_round=60,
        log_dir=".",  # ← 현재 파일 폴더(LeoFL/FL)에 저장
        log_filename_prefix="sat_events",
    )

    print("\n=== Federated Learning (with GS connectivity · event-based) ===")
    print(f"Best Accuracy: {best_acc:.2f}%")
