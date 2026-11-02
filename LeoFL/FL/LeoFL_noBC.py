# -*- coding: utf-8 -*-
# [초기 프로토타입] UNR-IDD 데이터로 지상국 간헐 연결 하의 동기식 FedAvg를 실험하는 단일 파일 스크립트(LeoFLyj.py의 변형).
import copy
import numpy as np
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

    def apply_zipf_sampling(
        self, alpha=1.0, num_samples=2000, cls_seq=[], name="Dataset"
    ):

        # 1) Zipf 분포 기반 클래스 인덱스 생성
        temp = Zipf(alpha, 1, self.n_classes, num_samples) - 1
        cnt_list = np.bincount(temp, minlength=self.n_classes)

        # 2) 클래스 순서 시드 고정
        if len(cls_seq) == 0:
            cls_sequence = np.random.permutation(self.n_classes)
        else:
            cls_sequence = np.array(cls_seq)

        # 3) 각 클래스별로 샘플링
        sampled_X_list, sampled_Y_list = [], []
        for idx in range(self.n_classes):
            cnt = int(cnt_list[idx])
            if cnt == 0:
                continue
            cls = int(cls_sequence[idx])
            cls_mask = self.Y == cls
            cls_X = self.X[cls_mask]
            cls_Y = self.Y[cls_mask]
            if cls_X.shape[0] == 0:
                continue
            indices = np.random.choice(cls_X.shape[0], cnt, replace=True)
            sampled_X_list.append(cls_X[indices])
            sampled_Y_list.append(cls_Y[indices])

        # 4) 새로운 샘플로 재구성
        if len(sampled_X_list) == 0:
            raise RuntimeError("Zipf 샘플링 결과가 비어 있습니다.")
        self.X = torch.cat(sampled_X_list, dim=0)
        self.Y = torch.cat(sampled_Y_list, dim=0)

        # 5) 클래스 분포 출력
        counts = np.bincount(self.Y.numpy().astype(int))
        print(f"[{name}] 클래스 분포: {counts} (총 {len(self.Y)}개 샘플)")

        return cls_sequence


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


# =========================
# 4) 위성 연결 스케줄링
# =========================
def build_satellite_constellation(num_clients, seed=1, h_min_km=500, h_max_km=1200):
    np.random.seed(seed)
    sats = []
    for i in range(num_clients):
        h_km = int(np.random.randint(h_min_km, h_max_km + 1))
        offset_sec = int(np.random.randint(0, 60 * 120))  # 0~2h
        incl_deg = int(np.random.randint(30, 98))  # SSOs 포함 대략 범위
        sats.append(
            LEOSat(h_km=h_km, offset_sec=offset_sec, incl_deg=incl_deg, id=i + 1)
        )
    return sats


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
# 5) 연합학습 (위성-지상국 업링크 제약 반영)
# =========================
def federated_learning_with_connectivity(
    client_datasets,
    test_dataset,
    num_rounds=10,
    local_epochs=1,
    device=None,
    hidden_size=100,
    lr=1e-3,
    seconds_per_round=60,
):
    assert len(client_datasets) > 0
    n_clients = len(client_datasets)

    # 글로벌/클라이언트 모델 초기화
    input_size = test_dataset.n_features
    num_classes = test_dataset.n_classes
    global_model = DNN(
        input_size=input_size, hidden_size=hidden_size, num_classes=num_classes
    ).to(device)
    client_models = [copy.deepcopy(global_model).to(device) for _ in range(n_clients)]

    # 위성 성좌 및 궤도 미리 계산
    sats = build_satellite_constellation(n_clients, seed=1)
    compute_orbits_for_rounds(
        sats, num_rounds=num_rounds, seconds_per_round=seconds_per_round
    )

    best_acc = -1.0
    for r in range(num_rounds):
        # 1) 현재 라운드에 지상국과 연결된 위성만 선별
        participating = connected_clients_at_round(sats, r, GS_LAT, GS_LON)

        # 2) 연결된 위성만 '다운링크'로 최신 글로벌 모델 수신
        for i in participating:
            client_models[i].load_state_dict(copy.deepcopy(global_model.state_dict()))

        # 3) 로컬 업데이트(연결된 위성만)
        for i in participating:
            _ = train_one_epoch(
                client_models[i],
                client_datasets[i],
                device=device,
                batch_size=32,
                lr=lr,
                class_weight=client_datasets[i].class_weight,
            )

        # 4) 업링크 + FedAvg 집계(연결된 위성만)
        fedavg_aggregate(global_model, client_models, participating)

        # 5) (중요) 라운드 종료 후 일괄 브로드캐스트는 하지 않음
        #    -> 다음 라운드에 '연결된' 위성만 새 글로벌을 받을 수 있음
        #    (비연결 위성은 이전에 갖고 있던 로컬 파라미터를 계속 보유)

        # 6) 평가
        acc = evaluate(global_model, test_dataset, device=device)
        best_acc = max(best_acc, acc)
        if r % 5 == 0:
            print(
                f"[Round {r + 1:02d}] connected={participating}  acc={acc:.2f}%  best={best_acc:.2f}%"
            )

    return global_model, best_acc


# =========================
# 6) 메인
# =========================
if __name__ == "__main__":
    # ---- 하이퍼파라미터 ----
    alpha = 2.0  # Zipf 계수(클래스 불균형 강도)
    num_clients = 10
    rounds = 200
    local_epochs = 1
    seconds_per_round = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 공통 스케일러/라벨인코더 학습용 중앙 데이터셋(파라미터만 맞춤) ----
    # 경로는 환경에 맞게 수정
    train_path = "LeoFL/UNR-IDD/UNR-IDD_train.feather"
    test_path = "LeoFL/UNR-IDD/UNR-IDD_test.feather"

    # 중앙 데이터셋
    central_train = NIDDataset(train_path, bBinary=False)
    cls_seq = central_train.apply_zipf_sampling(
        alpha=alpha, num_samples=25000, name="중앙 샘플 분포"
    )

    # 위성별 데이터
    client_datasets = []
    for i in range(num_clients):
        d = NIDDataset(
            train_path,
            bBinary=False,
            category_encoder=central_train.category_encoder,
            scaler=central_train.scaler,
        )
        d.apply_zipf_sampling(
            alpha=alpha, num_samples=2000, cls_seq=cls_seq, name=f"위성 {i + 1} Dataset"
        )
        client_datasets.append(d)

    # 테스트 데이터
    test_dataset = NIDDataset(
        test_path,
        bBinary=False,
        category_encoder=central_train.category_encoder,
        scaler=central_train.scaler,
    )
    # test_dataset.apply_zipf_sampling(alpha=0.0, num_samples=10000, name="테스트셋 분포")

    # ---- 연합학습 (지상국 연결 제약 반영) ----
    _, best_acc = federated_learning_with_connectivity(
        client_datasets=client_datasets,
        test_dataset=test_dataset,
        num_rounds=rounds,
        local_epochs=local_epochs,
        device=device,
        hidden_size=100,
        lr=1e-3,
        seconds_per_round=seconds_per_round,
    )

    print("\n=== Federated Learning (with GS connectivity) ===")
    print(f"Best Accuracy: {best_acc:.2f}%")
