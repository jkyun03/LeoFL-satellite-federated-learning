# fl_core.py
# FL 시뮬레이터 핵심부: 모델(DNN/CNN), 데이터 분할(non-IID), 위성군 생성(Walker-Delta/Starlink), 학습·평가·집계, 궤도면 내 오프로딩 공식, SimConfig.
import os, math, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn import preprocessing
from pandas import read_csv, read_feather
from typing import Tuple

# ---- 위성/지구 상수 및 커버리지 계산 유틸 ----
from LeoFL.Leo.leosat import LEOSat, is_point_in_coverage


def offload_within_plane(
    Ni: np.ndarray, Si: np.ndarray, max_iters: int = 1000
) -> np.ndarray:
    """
    Ni : 각 위성의 현재 데이터 개수 (float 또는 int)
    Si : 각 위성의 train capacity (학습 속도, 클수록 빠름)
    max_iters : greedy offloading 반복 횟수 한도

    반환 : offloading 후 각 위성의 데이터 개수 (int)
    """
    Ni = Ni.astype(float).copy()
    Si = Si.astype(float).copy()

    total = Ni.sum()
    # 각 위성의 목표 샘플 수: capacity 비율 * 총 샘플 수
    target = total * (Si / Si.sum())

    # 처음 excess 계산 (양수: 너무 많이 갖고 있음, 음수: 너무 적음)
    excess = Ni - target

    # greedy하게 sender -> receiver로 샘플 이동
    for _ in range(max_iters):
        sender = int(np.argmax(excess))  # 가장 많이 가진 애
        receiver = int(np.argmin(excess))  # 가장 적게 가진 애

        # 더 이상 의미 있는 이동이 없으면 종료
        if excess[sender] <= 0 or excess[receiver] >= 0:
            break

        # 이번에 옮길 개수 (정수) - 최소 1개
        # sender의 남는 양, receiver의 부족한 양 중 작은 쪽
        delta = min(excess[sender], -excess[receiver])

        # 너무 작으면(0.x) 안 옮기고 종료
        if delta < 0.5:
            break

        delta = int(round(delta))
        if delta <= 0:
            break

        Ni[sender] -= delta
        Ni[receiver] += delta

        excess[sender] -= delta
        excess[receiver] += delta

    # 최종 결과 int로 캐스팅 (총합 보존 보정)
    Ni_new = np.rint(Ni).astype(int)
    diff = int(total - Ni_new.sum())
    if diff != 0:
        # 합이 안 맞으면 제일 capacity 큰 애에 보정
        k = int(np.argmax(Si))
        Ni_new[k] += diff

    return Ni_new


def build_transfer_plan_within_plane(
    Ni: np.ndarray, Ni_new: np.ndarray
) -> List[Tuple[int, int, int]]:
    """
    Ni -> Ni_new를 만족시키기 위한 전송 계획 생성.
    반환: (sender_local_idx, receiver_local_idx, delta_samples)
    - delta_samples는 참고용(로깅/디버그). A 기준에서는 hop 계산 시 delta 무시하고 페어 1건으로 처리.
    """
    Ni = Ni.astype(int).copy()
    Ni_new = Ni_new.astype(int).copy()
    diff = Ni - Ni_new  # +면 sender(초과), -면 receiver(부족)

    senders = [(i, diff[i]) for i in range(len(diff)) if diff[i] > 0]
    receivers = [(i, -diff[i]) for i in range(len(diff)) if diff[i] < 0]

    plan = []
    si = 0
    ri = 0
    while si < len(senders) and ri < len(receivers):
        s_idx, s_excess = senders[si]
        r_idx, r_need = receivers[ri]

        delta = min(s_excess, r_need)
        plan.append((s_idx, r_idx, int(delta)))

        s_excess -= delta
        r_need -= delta

        if s_excess == 0:
            si += 1
        else:
            senders[si] = (s_idx, s_excess)

        if r_need == 0:
            ri += 1
        else:
            receivers[ri] = (r_idx, r_need)

    return plan


# 지상국 좌표 (서울)
GS_LAT = 37.5665
GS_LON = 126.9780


# ---------------------------
# 1) 모델
# ---------------------------
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


class SimpleCNN(nn.Module):
    """
    CIFAR-10 같은 3x32x32 이미지용 간단한 CNN
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                3, 32, kernel_size=3, padding=1
            ),  # [B, 3, 32, 32] -> [B, 32, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [B, 32, 16, 16]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> [B, 64, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [B, 64, 8, 8]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(
    model_name: str,
    num_classes: int,
    input_size: Optional[int] = None,
    hidden_size: int = 100,
):
    """
    model_name:
      - 'dnn' : tabular / embedding vector
      - 'cnn' : image tensor (e.g. CIFAR-10)
    """
    name = model_name.lower().strip()

    if name == "dnn":
        if input_size is None:
            raise ValueError("DNN 모델은 input_size가 필요합니다.")
        return DNN(
            input_size=input_size, hidden_size=hidden_size, num_classes=num_classes
        )

    elif name == "cnn":
        return SimpleCNN(num_classes=num_classes)

    else:
        raise ValueError(f"지원하지 않는 model_name: {model_name}")


# ---------------------------
# 2) 데이터셋
# ---------------------------
class NIDDataset(Dataset):
    def __init__(
        self, data_file, proto_encoder=None, category_encoder=None, scaler=None
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
        dataset_name = next((k for k in lbl_col if k in data_file.lower()), None)
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
            self.scaler = preprocessing.MinMaxScaler().fit(self.X)
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


# ==== (IID/Non-IID/Strat) ====
def split_dataset_stratified_iid(dataset: NIDDataset, num_clients: int, seed: int = 42):
    """
    Stratified IID:
    - 각 클래스의 전체 비율을 유지하면서
    - 각 클라이언트에게 균등하게 분배하는 방식
    """
    rng = np.random.default_rng(seed)
    Y_np = dataset.Y.numpy().astype(int)
    X = dataset.X
    clients = [[] for _ in range(num_clients)]

    for cls in np.unique(Y_np):
        idx = np.where(Y_np == cls)[0]
        rng.shuffle(idx)

        # 클래스별로 균등하게 잘라서 분배
        chunks = np.array_split(idx, num_clients)
        for c in range(num_clients):
            clients[c].extend(chunks[c])

    # 최종적으로 클라이언트 dataset 생성
    client_datasets = []
    for c in range(num_clients):
        inds = np.array(clients[c], dtype=int)
        Xc = X[inds]
        Yc = dataset.Y[inds]
        client_datasets.append(InMemoryClientDataset(dataset, Xc, Yc))

    return client_datasets


class InMemoryClientDataset(Dataset):
    """
    원본 NIDDataset의 스케일러/인코더/메타 유지하면서
    (X, Y) 텐서만 분할해서 담는 경량 Dataset
    """

    def __init__(
        self, base_dataset: NIDDataset, X_tensor: torch.Tensor, Y_tensor: torch.Tensor
    ):
        self.X = X_tensor
        self.Y = Y_tensor

        # 메타/속성 승계
        self.features = getattr(base_dataset, "features", None)
        if X_tensor.ndim == 2:
            self.n_features = X_tensor.shape[1]
        else:
            self.n_features = None
        self.n_classes = base_dataset.n_classes
        self.class_weight = (
            base_dataset.class_weight.clone()
            if hasattr(base_dataset, "class_weight")
            else None
        )
        self.scaler = getattr(base_dataset, "scaler", None)
        self.category_encoder = getattr(base_dataset, "category_encoder", None)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def split_dataset_iid(
    dataset: NIDDataset, num_clients: int, seed: int = 42, shuffle: bool = True
):
    """
    IID 분할: 전체 인덱스를 균등하게 겹치지 않게 나눔
    """
    rng = np.random.default_rng(seed)
    N = len(dataset)
    inds = np.arange(N)
    if shuffle:
        rng.shuffle(inds)
    chunks = np.array_split(inds, num_clients)

    clients = []
    for ch in chunks:
        Xc = dataset.X[ch]
        Yc = dataset.Y[ch]
        clients.append(InMemoryClientDataset(dataset, Xc, Yc))
    return clients


def split_dataset_dirichlet(
    dataset: NIDDataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
    shuffle_within: bool = True,
):
    """
    Non-IID(Dirichlet). 각 클래스별로 p~Dir(alpha) 비율을 뽑아서
    '겹치지 않게' 인덱스를 분배
    - alpha ↓ → 더 불균형(Non-IID 강함)
    """
    rng = np.random.default_rng(seed)
    Y_np = dataset.Y.numpy().astype(int)
    classes = np.unique(Y_np)

    client_indices = [[] for _ in range(num_clients)]

    for cls in classes:
        idx = np.where(Y_np == cls)[0]
        rng.shuffle(idx)
        p = rng.dirichlet([alpha] * num_clients)
        counts = rng.multinomial(len(idx), p)
        start = 0
        for c in range(num_clients):
            take = counts[c]
            if take > 0:
                client_indices[c].extend(idx[start : start + take])
            start += take

    clients = []
    for c in range(num_clients):
        inds = np.array(client_indices[c], dtype=int)
        if shuffle_within:
            rng.shuffle(inds)
        Xc = dataset.X[inds]
        Yc = dataset.Y[inds]
        clients.append(InMemoryClientDataset(dataset, Xc, Yc))
    return clients


def make_clients_from_dataset(
    dataset: NIDDataset,
    num_clients: int,
    split_mode: str = "non",
    alpha: float = 0.5,
    seed: int = 42,
):
    """
    split_mode:
        - "iid" : 랜덤 IID 균등 분배
        - "strat" : Stratified IID 균등 분배 (클래스 비율 동일하게 유지)
        - "non": Dirichlet Non-IID 분배
    """
    mode = split_mode.lower().strip()

    if mode == "strat":
        return split_dataset_stratified_iid(dataset, num_clients=num_clients, seed=seed)
    elif mode == "non":
        return split_dataset_dirichlet(
            dataset, num_clients=num_clients, alpha=alpha, seed=seed
        )
    elif mode == "iid":
        return split_dataset_iid(dataset, num_clients=num_clients, seed=seed)
    else:
        raise ValueError(f"지원하지 않는 split_mode: {split_mode}")


# ---------------------------
# 3) 학습/평가/집계
# ---------------------------
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
        loss = criterion(model(X), Y)
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
        pred = model(X).argmax(dim=1)
        correct += (pred == Y).sum().item()
        total += Y.size(0)
    return 100.0 * correct / max(1, total)


def fedavg_aggregate_fn(global_model, client_models, participating_indices):
    if len(participating_indices) == 0:
        return
    with torch.no_grad():
        g = global_model.state_dict()
        for k in g:
            g[k].zero_()
        for i in participating_indices:
            ls = client_models[i].state_dict()
            for k in g:
                g[k].add_(ls[k])
        for k in g:
            g[k].div_(float(len(participating_indices)))
        global_model.load_state_dict(g)


def decentralized_ring_aggregate_fn(global_model, client_models, participating_indices):
    """
    완전 분산형 FL을 근사하는 ring/gossip 스타일 집계 함수.
    - 실제 네트워크 상에서는 이웃끼리 차례로 평균을 내는 ring all-reduce와 유사.
    - 여기서는 구현 단순화를 위해:
        1) 참여 클라이언트들의 모델을 모두 더하고
        2) 개수로 나눠서 평균
        3) 그 결과를 global_model과 모든 참여 클라이언트에 뿌린다.
    - 수학적으로는 FedAvg와 동일한 결과지만,
      run_simulation()에서 이 모드에 대해
      "지상국 없이 peer-to-peer로 평균한다"는 프로토콜을 정의하고,
      통신 비용/구조 분석에서 Decentralized FL로 해석할 수 있다.
    """
    if len(participating_indices) == 0:
        return

    with torch.no_grad():
        # 기준이 될 첫 클라이언트의 state_dict 복사
        base_idx = participating_indices[0]
        g = copy.deepcopy(client_models[base_idx].state_dict())

        # 나머지 클라이언트들의 모델 파라미터를 누적
        for idx in participating_indices[1:]:
            st = client_models[idx].state_dict()
            for k in g.keys():
                g[k].add_(st[k])

        # 평균
        for k in g.keys():
            g[k].div_(float(len(participating_indices)))

        # 글로벌 모델에 반영
        global_model.load_state_dict(g)

        # 분산 FL 가정: 참여한 모든 클라이언트도 같은 평균 모델을 공유
        for idx in participating_indices:
            client_models[idx].load_state_dict(g)


AGGREGATORS: Dict[str, Callable] = {
    "fedavg": fedavg_aggregate_fn,
    "decentral": decentralized_ring_aggregate_fn,
}


def stop_on_target_accuracy(target: float):
    def _fn(minute, acc, best_acc):
        return acc >= target

    return _fn


def stop_never():
    def _fn(minute, acc, best_acc):
        return False

    return _fn


# ---------------------------
# 4) 위성/궤도 유틸
# ---------------------------
"""
FIXED_CONSTELLATION = [
    # ===== Plane 0: 저고도, 중간 경사 =====
    {"h_km": 550, "incl_deg": 50, "offset_min": 0,   "plane_id": 0},
    {"h_km": 550, "incl_deg": 50, "offset_min": 10,  "plane_id": 0},
    {"h_km": 550, "incl_deg": 50, "offset_min": 20,  "plane_id": 0},
    {"h_km": 550, "incl_deg": 50, "offset_min": 30,  "plane_id": 0},
    {"h_km": 550, "incl_deg": 50, "offset_min": 40,  "plane_id": 0},
    {"h_km": 550, "incl_deg": 50, "offset_min": 50,  "plane_id": 0},

    # ===== Plane 1: 살짝 더 높은 고도 & 다른 경사 =====
    {"h_km": 600, "incl_deg": 53, "offset_min": 5,   "plane_id": 1},
    {"h_km": 600, "incl_deg": 53, "offset_min": 15,  "plane_id": 1},
    {"h_km": 600, "incl_deg": 53, "offset_min": 25,  "plane_id": 1},
    {"h_km": 600, "incl_deg": 53, "offset_min": 35,  "plane_id": 1},
    {"h_km": 600, "incl_deg": 53, "offset_min": 45,  "plane_id": 1},
    {"h_km": 600, "incl_deg": 53, "offset_min": 55,  "plane_id": 1},

    # ===== Plane 2: 더 높은 고도 =====
    {"h_km": 650, "incl_deg": 56, "offset_min": 2,   "plane_id": 2},
    {"h_km": 650, "incl_deg": 56, "offset_min": 12,  "plane_id": 2},
    {"h_km": 650, "incl_deg": 56, "offset_min": 22,  "plane_id": 2},
    {"h_km": 650, "incl_deg": 56, "offset_min": 32,  "plane_id": 2},
    {"h_km": 650, "incl_deg": 56, "offset_min": 42,  "plane_id": 2},
    {"h_km": 650, "incl_deg": 56, "offset_min": 52,  "plane_id": 2},

    # ===== Plane 3: 가장 높은 고도 =====
    {"h_km": 700, "incl_deg": 60, "offset_min": 8,   "plane_id": 3},
    {"h_km": 700, "incl_deg": 60, "offset_min": 18,  "plane_id": 3},
    {"h_km": 700, "incl_deg": 60, "offset_min": 28,  "plane_id": 3},
    {"h_km": 700, "incl_deg": 60, "offset_min": 38,  "plane_id": 3},
    {"h_km": 700, "incl_deg": 60, "offset_min": 48,  "plane_id": 3},
    {"h_km": 700, "incl_deg": 60, "offset_min": 58,  "plane_id": 3},
]
"""
# ----------------------------------------------
# Starlink-like small constellation (8 planes × 10 sats)
# ----------------------------------------------

STARLINK_LIKE_SMALL = []


def make_starlink_shell1_active10_constellation(
    full_num_planes: int = 72,
    sats_per_plane: int = 22,
    active_num_planes: int = 10,
    h_km: float = 552.0,
    incl_deg: float = 53.0,
    phase_factor: int = 0,
):
    """
    Starlink-like Shell 1 구조를 만든 뒤,
    그중 10개 plane만 active FL plane으로 선택.

    반환:
        active_configs: 실제 FL에 참여하는 satellite configs
        full_configs: 전체 Starlink-like shell configs
        active_plane_ids: 원래 shell 기준 active plane id
    """
    full_configs = make_walker_delta_constellation(
        num_planes=full_num_planes,
        sats_per_plane=sats_per_plane,
        h_km=h_km,
        incl_deg=incl_deg,
        phase_factor=phase_factor,
    )

    # 전체 72개 plane 중 10개를 최대한 균등하게 선택
    active_plane_ids = (
        np.round(np.linspace(0, full_num_planes - 1, active_num_planes))
        .astype(int)
        .tolist()
    )

    # 혹시 중복 생기면 보정
    active_plane_ids = sorted(list(dict.fromkeys(active_plane_ids)))
    while len(active_plane_ids) < active_num_planes:
        for p in range(full_num_planes):
            if p not in active_plane_ids:
                active_plane_ids.append(p)
            if len(active_plane_ids) >= active_num_planes:
                break
    active_plane_ids = sorted(active_plane_ids[:active_num_planes])

    active_configs = select_active_planes_from_constellation(
        full_configs=full_configs,
        active_plane_ids=active_plane_ids,
        reindex_plane_ids=True,
    )

    print(
        "[CONSTELLATION] full Starlink-like shell:",
        f"{full_num_planes} planes × {sats_per_plane} sats = {len(full_configs)} sats",
    )
    print(
        "[CONSTELLATION] active FL planes:",
        f"{len(active_plane_ids)} planes × {sats_per_plane} sats = {len(active_configs)} sats",
    )
    print("[CONSTELLATION] active original plane ids:", active_plane_ids)

    return active_configs, full_configs, active_plane_ids


num_planes = 8
sats_per_plane = 10

h_list = [550, 560, 570, 580, 550, 560, 570, 580]
incl_list = [53, 55, 57, 60, 65, 70, 85, 97.6]

# --- 핵심: RAAN을 plane 수만큼 0~360도에 균등 분포 ---
raan_list = np.linspace(0, 360, num_planes, endpoint=False)

for plane_id in range(num_planes):
    h = h_list[plane_id]
    incl = incl_list[plane_id]
    raan = raan_list[plane_id]

    for sat_idx in range(sats_per_plane):
        offset = sat_idx * (60 / sats_per_plane)  # plane 내부 위상 분포

        STARLINK_LIKE_SMALL.append(
            {
                "h_km": h,
                "incl_deg": incl,
                "raan_deg": raan,  # <-- 추가됨: plane 중심의 경도 위치
                "offset_min": offset,
                "plane_id": plane_id,
            }
        )


def make_walker_delta_constellation(
    num_planes: int = 72,
    sats_per_plane: int = 22,
    h_km: float = 552.0,
    incl_deg: float = 53.0,
    phase_factor: int = 0,
):
    """
    Walker-delta-style Starlink-like constellation generator.

    전체 shell을 생성한다.
    예: 72 planes × 22 satellites = 1584 satellites.
    """
    configs = []

    # Walker-delta: RAAN을 0~360도에 균등 분포
    raan_list = np.linspace(0, 360, num_planes, endpoint=False)

    for plane_id in range(num_planes):
        raan = float(raan_list[plane_id])

        for sat_idx in range(sats_per_plane):
            # 기존 코드가 offset_min을 사용하므로,
            # 한 궤도 주기를 60분으로 정규화해서 plane 내부 위상 부여
            phase_shift = phase_factor * plane_id / max(1, num_planes)
            phase_slot = (sat_idx + phase_shift) % sats_per_plane
            offset_min = phase_slot * (60.0 / sats_per_plane)

            configs.append(
                {
                    "h_km": float(h_km),
                    "incl_deg": float(incl_deg),
                    "raan_deg": raan,
                    "offset_min": float(offset_min),
                    "plane_id": int(plane_id),
                    # 원래 shell에서의 정보 보존용
                    "orig_plane_id": int(plane_id),
                    "sat_idx_in_plane": int(sat_idx),
                }
            )

    return configs


def select_active_planes_from_constellation(
    full_configs,
    active_plane_ids,
    reindex_plane_ids: bool = True,
):
    """
    full constellation 중 일부 plane만 active FL participants로 선택.

    reindex_plane_ids=True:
        선택된 plane들을 0,1,2,...로 다시 numbering.
        기존 scenario들이 plane_id로 grouping하므로 이게 안전함.

    orig_plane_id:
        원래 Starlink-like shell에서의 plane 번호는 보존.
    """
    active_plane_ids = list(active_plane_ids)
    active_set = set(active_plane_ids)
    plane_id_map = {p: new_p for new_p, p in enumerate(active_plane_ids)}

    active_configs = []

    for c in full_configs:
        old_p = int(c["plane_id"])

        if old_p not in active_set:
            continue

        new_c = dict(c)
        new_c["orig_plane_id"] = old_p

        if reindex_plane_ids:
            new_c["plane_id"] = int(plane_id_map[old_p])
        else:
            new_c["plane_id"] = old_p

        active_configs.append(new_c)

    return active_configs


def make_starlink_shell_active10_constellation(
    full_num_planes: int = 72,
    sats_per_plane: int = 22,
    active_num_planes: int = 10,
    h_km: float = 552.0,
    incl_deg: float = 53.0,
    phase_factor: int = 0,
):
    """
    Starlink-like full shell을 만든 뒤, 그중 10개 plane만 active FL plane으로 선택.

    반환:
        active_configs: 실제 FL 학습에 참여하는 위성 configs
        full_configs: 전체 shell configs
        active_plane_ids: 원래 shell 기준 active plane id
    """
    full_configs = make_walker_delta_constellation(
        num_planes=full_num_planes,
        sats_per_plane=sats_per_plane,
        h_km=h_km,
        incl_deg=incl_deg,
        phase_factor=phase_factor,
    )

    # 전체 plane에서 10개를 균등하게 선택
    active_plane_ids = (
        np.round(np.linspace(0, full_num_planes - 1, active_num_planes))
        .astype(int)
        .tolist()
    )

    # 중복 방지
    active_plane_ids = sorted(list(dict.fromkeys(active_plane_ids)))

    # 혹시 round 때문에 10개보다 적어지면 앞쪽 plane으로 보충
    while len(active_plane_ids) < active_num_planes:
        for p in range(full_num_planes):
            if p not in active_plane_ids:
                active_plane_ids.append(p)
            if len(active_plane_ids) >= active_num_planes:
                break

    active_plane_ids = sorted(active_plane_ids[:active_num_planes])

    active_configs = select_active_planes_from_constellation(
        full_configs=full_configs,
        active_plane_ids=active_plane_ids,
        reindex_plane_ids=True,
    )

    print(
        "[CONSTELLATION] full Starlink-like shell:",
        f"{full_num_planes} planes × {sats_per_plane} sats = {len(full_configs)} sats",
    )
    print(
        "[CONSTELLATION] active FL planes:",
        f"{len(active_plane_ids)} planes × {sats_per_plane} sats = {len(active_configs)} sats",
    )
    print("[CONSTELLATION] active original plane ids:", active_plane_ids)

    return active_configs, full_configs, active_plane_ids


FIXED_CONSTELLATION = STARLINK_LIKE_SMALL


def build_fixed_constellation(configs):
    sats = []
    for i, c in enumerate(configs):
        sats.append(
            LEOSat(
                h_km=c["h_km"],
                incl_deg=c["incl_deg"],
                offset_sec=int(round(c["offset_min"] * 60)),
                id=i + 1,
                raan_deg=c.get("raan_deg", 0.0),  # ★ 여기서 RAAN 넘겨줌
            )
        )
    return sats


"""def build_fixed_constellation(configs):
    sats = []
    for i, c in enumerate(configs):
        sats.append(LEOSat(
            h_km=c["h_km"],
            incl_deg=c["incl_deg"],
            offset_sec=int(round(c["offset_min"] * 60)),
            id=i+1,
        ))
    return sats
"""


def compute_orbits_for_rounds(sat_list, num_rounds, seconds_per_round=60):
    times = np.arange(0, num_rounds * seconds_per_round, seconds_per_round)
    for sat in sat_list:
        sat.compute_leo_orbit(times)


def reindex_satellites_by_orbital_phase(sat_list, plane_ids, round_idx=0):
    """
    같은 plane 내 위성들을 '같은 시점 round_idx'의 true orbital phase 기준으로
    정렬하여 ID를 재부여한다.

    즉, 물리적으로 가까운 위성끼리 sat_1, sat_2, ...로 가깝게 붙는다.
    논문 Figure 및 hop-path 설명을 직관적으로 하기 위함.
    """
    n = len(sat_list)

    # plane별로 분리
    plane_to_sats = {}
    for i, p in enumerate(plane_ids):
        plane_to_sats.setdefault(p, []).append(i)

    # 새로운 sat_list를 담을 공간
    new_sat_list = [None] * n
    new_plane_ids = [0] * n

    # 재정렬할 때 부여할 새로운 ID (1, 2, 3, ...)
    new_id_counter = 1

    # plane 단위로 처리
    for p, indices in plane_to_sats.items():
        # 각 위성의 그 순간 위치 가져오기
        sat_positions = []
        for idx in indices:
            pos = sat_list[idx].get_leo_orbit_position(round_idx)
            sat_positions.append((idx, pos))

        # 각 위치의 위상(angle) 계산 — x,y 평면 투영 후 arctan2
        phase_list = []
        for idx, pos in sat_positions:
            x, y, z = pos
            phase = np.arctan2(y, x)  # -pi ~ +pi
            phase_list.append((idx, phase))

        # 위상(angle) 순으로 정렬
        phase_list.sort(key=lambda x: x[1])

        # 재정렬된 순서대로 새로운 ID 부여
        for idx, _phase in phase_list:
            sat = sat_list[idx]
            # sat 객체 복사 (ID만 새로 부여)
            new_sat = copy.copy(sat)
            new_sat.id = new_id_counter

            new_index = new_id_counter - 1
            new_sat_list[new_index] = new_sat
            new_plane_ids[new_index] = p

            new_id_counter += 1

    return new_sat_list, new_plane_ids


def connected_clients_at_round(sat_list, round_idx, gs_lat=GS_LAT, gs_lon=GS_LON):
    connected = []
    for i, sat in enumerate(sat_list):
        pos = sat.get_leo_orbit_position(round_idx)
        inside, _, _ = is_point_in_coverage(pos, gs_lat, gs_lon)
        if inside:
            connected.append(i)
    return connected


def minutes_to_rounds(minutes: float, seconds_per_round: int) -> int:
    return int(math.ceil(minutes * 60.0 / float(seconds_per_round)))


# ---------------------------
# 4-1 위성-위성 링크 및 리더 선정 유틸 (OFFLOAD용)
# ---------------------------


def _euclidean_distance_km(pos1, pos2):
    """
    LEOSat.get_leo_orbit_position()이 m 단위 좌표를 준다고 가정하고
    유클리드 거리(km)를 계산.
    """
    p1 = np.asarray(pos1, dtype=float) / 1000.0
    p2 = np.asarray(pos2, dtype=float) / 1000.0
    return float(np.linalg.norm(p1 - p2))


def sat_contacts_at_round(sat_list, round_idx, max_dist_km: float = 2000.0):
    """
    round_idx 시점에 서로 통신 가능한 위성 쌍 (i, j) 리스트를 반환.
    단순히 두 위성 간 거리가 max_dist_km 이하면 연결된 것으로 본다.
    """
    n = len(sat_list)
    positions = [sat.get_leo_orbit_position(round_idx) for sat in sat_list]

    """
    # 디버그: 처음 몇 step에서 좌표/거리 출력
    if round_idx < 3:
        print(f"[DEBUG][sat_contacts] t={round_idx}")
        for idx, pos in enumerate(positions):
            print(f"  sat_{idx+1} pos = {pos}")
    """
    contacts = []
    for i in range(n):
        for j in range(i + 1, n):
            dist_km = _euclidean_distance_km(positions[i], positions[j])
            # if round_idx < 3:
            #    print(f"  dist(sat_{i+1}, sat_{j+1}) = {dist_km:.3f} km")
            if dist_km <= max_dist_km:
                contacts.append((i, j))
    return contacts


def sample_train_rounds_with_exp_jitter(
    mean_minutes: float,
    seconds_per_round: int,
    jitter_sec: float = 0.5,
    clamp: bool = True,
    rng: np.random.Generator = None,
) -> int:
    """
    mean_minutes: 위성 i의 '평균' 학습시간 (분)
    라운드마다 실제 학습시간을:
        T = mean_sec + noise
    noise는 two-sided exponential(Laplace)로 생성:
        noise ~ Laplace(0, b)
        (Laplace는 '지수분포 기반' 양방향 노이즈)

    jitter_sec=0.5이면, 보통 ±0.5초 수준으로 흔들리게 만들고,
    clamp=True면 실제값을 [mean-0.5, mean+0.5]로 강제 제한.
    """

    if rng is None:
        rng = np.random.default_rng()

    mean_sec = float(mean_minutes) * 60.0

    # two-sided exponential = Laplace(0, b)
    # Laplace의 평균절대편차는 b, 분산은 2b^2
    # 여기서는 "흔들림 스케일"을 jitter_sec로 두고 b=jitter_sec로 둠
    noise = rng.laplace(loc=0.0, scale=float(jitter_sec))

    t = mean_sec + noise

    if clamp:
        lo = mean_sec - float(jitter_sec)
        hi = mean_sec + float(jitter_sec)
        t = min(max(t, lo), hi)

    # 학습시간이 0초 이하로 내려가면 안 되므로 안전장치
    t = max(t, 1e-6)

    # 실제 코드는 "몇 라운드 걸리냐"로 모델링하니까 rounds로 변환
    return int(math.ceil(t / float(seconds_per_round)))


# ---------------------------
# 4-0 통신 링크 랜덤성 (link failure / hop jitter)
# ---------------------------
LINK_FAIL_P_DEFAULT = 0.05
HOP_JITTER_PCT_DEFAULT = 0.10


def filter_by_link_failure(rng: Optional[np.random.Generator], p_fail: float, items):
    """
    items(수신 후보 목록)의 각 원소를, 그 방향의 전송이 이번 라운드에 성공했다고 볼 때만
    살아남기는 방식으로 독립적으로 필터링한다 (rng가 None이거나 p_fail<=0이면 원본 그대로 반환
    — 즉 link failure를 켜지 않은 기존 시나리오는 동작이 전혀 바뀌지 않는다).
    """
    if rng is None or p_fail <= 0:
        return list(items)
    return [x for x in items if rng.random() >= p_fail]


def link_success(rng: Optional[np.random.Generator], p_fail: float) -> bool:
    """단일 전송(예: PS<->plane 업/다운링크)이 이번 시도에 성공했는지."""
    if rng is None or p_fail <= 0:
        return True
    return rng.random() >= p_fail


def jitter_hop_count(
    rng: Optional[np.random.Generator], hops: int, pct: float = HOP_JITTER_PCT_DEFAULT
) -> int:
    """
    경로 길이(hop) 개념이 있는 통신(ring 전파, offload 전송 등)에 ±pct 비율의
    소규모 랜덤 변동을 준다. 최소 1홉은 유지. rng가 None이면 원본 그대로 반환.
    """
    if rng is None or hops <= 0:
        return hops
    factor = 1.0 + rng.uniform(-pct, pct)
    return max(1, int(round(hops * factor)))


def precompute_meetings_and_select_leaders(
    sat_list,
    plane_ids,
    num_rounds: int,
    seconds_per_round: int = 60,
    max_dist_km: float = 2000.0,
    log_dir: Optional[str] = None,
    filename: str = "meeting_counts_offload.csv",
):
    """
    - 전체 시뮬레이션 동안 위성 간 접촉(만남) 횟수를 세고,
    - plane(궤도)별로 '다른 plane 위성과 가장 자주 만난' 위성을 leader로 선정.
    """
    n = len(sat_list)
    if len(plane_ids) != n:
        raise ValueError("plane_ids 길이는 sat_list 길이와 같아야 합니다.")

    meeting_counts = np.zeros((n, n), dtype=int)
    for t in range(num_rounds):
        contacts = sat_contacts_at_round(sat_list, t, max_dist_km=max_dist_km)
        for i, j in contacts:
            meeting_counts[i, j] += 1
            meeting_counts[j, i] += 1

    total_meet = int(meeting_counts.sum())
    nonzero_pairs = int((meeting_counts > 0).sum() // 2)
    print("[DEBUG] meeting_counts matrix =")
    print(meeting_counts)
    print("[DEBUG] total meetings =", total_meet)
    print("[DEBUG] nonzero pairs =", nonzero_pairs)

    # 만나는 쌍이 하나도 없으면 synthetic meeting 주입
    if total_meet == 0:
        print(
            "[WARN] no inter-satellite meetings detected; injecting synthetic meetings for offload."
        )
        for i in range(n):
            for j in range(i + 1, n):
                if plane_ids[i] != plane_ids[j]:
                    meeting_counts[i, j] = 1
                    meeting_counts[j, i] = 1
        print("[DEBUG] synthetic meeting_counts =")
        print(meeting_counts)

    plane_to_sats = {}
    for idx, p in enumerate(plane_ids):
        plane_to_sats.setdefault(p, []).append(idx)

    leaders_per_plane = {}
    for p, sat_indices in plane_to_sats.items():
        best_sat = None
        best_meet = -1
        for i in sat_indices:
            inter_meet = 0
            for j in range(n):
                if plane_ids[j] != p:
                    inter_meet += meeting_counts[i, j]
            if inter_meet > best_meet:
                best_meet = inter_meet
                best_sat = i
        leaders_per_plane[p] = best_sat

    # CSV 저장
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, filename)
        np.savetxt(path, meeting_counts, delimiter=",", fmt="%d")
        print(f"[SAVE] meeting_counts matrix saved to {path}")

    return meeting_counts, leaders_per_plane, plane_to_sats


# ---------------------------
# 5) 로깅
# ---------------------------
def save_events_only_csv(events_log, log_dir, prefix):
    if not log_dir:
        log_dir = os.path.dirname(os.path.abspath(__file__))
    elif not os.path.isabs(log_dir):
        base = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{prefix}_events_only.csv")
    pd.DataFrame(events_log, columns=["time", "sat", "event"]).to_csv(
        path, index=False, encoding="utf-8-sig"
    )
    # --- 네가 쓰던 스타일로 로그 ---
    print(f"[SAVE][EVENTS] {path} (rows={len(events_log)})")


def save_time_accuracy_csv(acc_log, save_csv_path: str):
    """
    acc_log: List[Tuple[round, accuracy]]
    NOTE: 기존 함수명은 유지하되, CSV 컬럼은 round로 통일한다.
    """
    if not acc_log:
        print("[WARN] accuracy log가 비어있어 CSV를 생성하지 않습니다.")
        return
    dir_name = os.path.dirname(save_csv_path) or "."
    os.makedirs(dir_name, exist_ok=True)
    pd.DataFrame(acc_log, columns=["round", "accuracy"]).to_csv(
        save_csv_path, index=False, encoding="utf-8-sig"
    )
    print(f"[SAVE][ACCURACY] {save_csv_path} (rows={len(acc_log)})")


# ---------------------------
# 6) 공통 설정 객체
# ---------------------------
@dataclass
class SimConfig:
    num_rounds: int = 10000
    local_epochs: int = 1
    device: Optional[torch.device] = None
    hidden_size: int = 100
    lr: float = 1e-3
    seconds_per_round: int = 60
    log_dir: Optional[str] = "."
    target_acc: float = 70.0
    aggregate_name: str = "fedavg"
    constellation_configs: List[dict] = None
    log_prefix: str = "sat_events"
    model_name: str = "dnn"

    def __post_init__(self):
        if self.constellation_configs is None:
            self.constellation_configs = FIXED_CONSTELLATION


def _swap_dataset_samples(ds_a, ds_b, ratio: float, rng: np.random.Generator):
    """
    두 클라이언트 Dataset의 일부 샘플을 서로 교환하여
    non-IID를 완화하는 offloading 함수 (label-aware 버전).
    - 각 데이터셋의 전체 샘플 수는 유지.
    - 양쪽에서 '같은 개수'만큼만 교환.
    - 가능한 한 '과대표된 클래스'를 내보내고, '과소대표된 클래스'를 받아오도록 시도.
    """
    n_a = len(ds_a)
    n_b = len(ds_b)
    if n_a == 0 or n_b == 0:
        return

    max_possible = min(n_a, n_b)
    k_swap = int(max_possible * ratio)
    if k_swap < 1:
        k_swap = 1

    # 라벨 numpy 배열
    y_a = ds_a.Y.numpy().astype(int)
    y_b = ds_b.Y.numpy().astype(int)

    # 전체 가능한 클래스 (0 ~ C-1 가정)
    n_classes = max(y_a.max(), y_b.max()) + 1
    cnt_a = np.bincount(y_a, minlength=n_classes)
    cnt_b = np.bincount(y_b, minlength=n_classes)

    total_a = cnt_a.sum()
    total_b = cnt_b.sum()
    if total_a == 0 or total_b == 0:
        return

    # 비율
    p_a = cnt_a / total_a
    p_b = cnt_b / total_b

    # A에서 "과다"인 클래스 (p_a > p_b)
    surplus_a = np.where(p_a > p_b)[0]
    surplus_b = np.where(p_b > p_a)[0]

    # 후보 인덱스 뽑기
    def _sample_indices_by_labels(y, label_list, k):
        if len(label_list) == 0:
            return np.array([], dtype=int)
        mask = np.isin(y, label_list)
        cand = np.where(mask)[0]
        if len(cand) == 0:
            return np.array([], dtype=int)
        k_eff = min(k, len(cand))
        return rng.choice(cand, size=k_eff, replace=False)

    idx_a = _sample_indices_by_labels(y_a, surplus_a, k_swap)
    idx_b = _sample_indices_by_labels(y_b, surplus_b, k_swap)

    # 혹시 surplus 클래스가 거의 없다면 → 그냥 random fallback
    if len(idx_a) < k_swap:
        rest = k_swap - len(idx_a)
        others = np.setdiff1d(np.arange(n_a), idx_a)
        if len(others) > 0:
            rest = min(rest, len(others))
            add_idx = rng.choice(others, size=rest, replace=False)
            idx_a = np.concatenate([idx_a, add_idx])

    if len(idx_b) < k_swap:
        rest = k_swap - len(idx_b)
        others = np.setdiff1d(np.arange(n_b), idx_b)
        if len(others) > 0:
            rest = min(rest, len(others))
            add_idx = rng.choice(others, size=rest, replace=False)
            idx_b = np.concatenate([idx_b, add_idx])

    # 최종적으로 양쪽 길이를 맞춰줌
    k_final = min(len(idx_a), len(idx_b))
    if k_final == 0:
        return

    idx_a = idx_a[:k_final]
    idx_b = idx_b[:k_final]

    Xa_sel = ds_a.X[idx_a].clone()
    Ya_sel = ds_a.Y[idx_a].clone()
    Xb_sel = ds_b.X[idx_b].clone()
    Yb_sel = ds_b.Y[idx_b].clone()

    ds_a.X[idx_a] = Xb_sel
    ds_a.Y[idx_a] = Yb_sel
    ds_b.X[idx_b] = Xa_sel
    ds_b.Y[idx_b] = Ya_sel


def _move_samples(ds_from, ds_to, idx_from: np.ndarray):
    """
    ds_from에서 선택된 샘플들을 ds_to로 '이동' (삭제 후 추가)하는 함수.
    - idx_from: ds_from 내부 인덱스 배열
    """
    if len(idx_from) == 0:
        return

    # 선택된 샘플 복사
    X_sel = ds_from.X[idx_from].clone()
    Y_sel = ds_from.Y[idx_from].clone()

    # from에서 제거
    mask = torch.ones(len(ds_from), dtype=torch.bool)
    mask[idx_from] = False
    ds_from.X = ds_from.X[mask]
    ds_from.Y = ds_from.Y[mask]

    # to에 추가
    ds_to.X = torch.cat([ds_to.X, X_sel], dim=0)
    ds_to.Y = torch.cat([ds_to.Y, Y_sel], dim=0)


def offload_balance_datasets(
    client_datasets,
    meeting_counts: np.ndarray,
    plane_ids,
    offload_ratio: float = 0.1,
    top_pairs_per_sat: int = 1,
    seed: int = 0,
    train_capacities: Optional[List[float]] = None,
):
    """
    meeting_counts(i,j)가 큰 (자주 만나는) 위성 페어를 중심으로
    각 위성 데이터셋 일부를 서로 교환/이동하여 분포를 섞는다.

    train_capacities가 None이면: 기존처럼 'swap'으로 클래스 분포 완화 위주.
    train_capacities가 주어지면: 각 plane 내에서
        N_i / s_i (학습시간)가 균등해지도록 'move' 기반 offload 수행.

    [추가]
    - capacity 모드에서는 offloading으로 발생하는 전송횟수 (Transmission Count)를
      plane 내 인접 hop 기반으로 계산하여 반환한다.
    """
    rng = np.random.default_rng(seed)
    n = len(client_datasets)

    Ni_before_global = np.array([len(ds) for ds in client_datasets], dtype=float)
    Ni_target_global = np.zeros(n, dtype=float)  # plane별 target 기록용

    # ---- 공통: plane별 위성 묶기 ----
    plane_to_sats = {}
    for idx, p in enumerate(plane_ids):
        plane_to_sats.setdefault(p, []).append(idx)

    # ---- (A) train_capacities == None: 기존 swap 모드 유지 ----
    if train_capacities is None:
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                # 같은 plane끼리만
                if plane_ids[i] != plane_ids[j]:
                    continue
                c = meeting_counts[i, j]
                if c > 0:
                    pairs.append((c, i, j))

        pairs.sort(reverse=True, key=lambda x: x[0])
        used_count = [0] * n
        used_pairs = []

        for c, i, j in pairs:
            if (
                used_count[i] >= top_pairs_per_sat
                and used_count[j] >= top_pairs_per_sat
            ):
                continue
            _swap_dataset_samples(
                client_datasets[i], client_datasets[j], offload_ratio, rng
            )
            used_count[i] += 1
            used_count[j] += 1
            used_pairs.append((i, j, c))

        print(
            f"[OFFLOAD][swap] ratio={offload_ratio}, top_pairs_per_sat={top_pairs_per_sat}, num_pairs={len(used_pairs)}"
        )
        for i, j, c in used_pairs:
            print(f"  - sat_{i + 1} ↔ sat_{j + 1}, meetings={c}")
    # ---- (B) train_capacities 제공: 학습시간 균등화 모드 ----
    else:
        train_capacities = np.asarray(train_capacities, dtype=float)
        assert len(train_capacities) == n, (
            "train_capacities 길이는 client 개수와 같아야 합니다."
        )

        print("[OFFLOAD][capacity] train_capacities =", train_capacities)

        offload_hops_total = 0

        # plane 단위로 offloading 수행
        for p, sat_indices in plane_to_sats.items():
            sat_indices = list(sat_indices)  # 예: [0,1,2,3,4,5]

            # 1) 현재 각 위성의 샘플 개수(Ni)와 capacity(Si)
            Ni = np.array([len(client_datasets[i]) for i in sat_indices], dtype=float)
            Si = train_capacities[sat_indices]

            N_tot = Ni.sum()
            S_tot = Si.sum()
            if N_tot == 0 or S_tot == 0:
                continue

            print(f"[OFFLOAD][plane {p}] Ni={Ni}, Si={Si}")

            # 2) offload_within_plane로 목표 분배량 계산
            Ni_new = offload_within_plane(Ni, Si)
            print(
                f"[OFFLOAD][plane {p}] target Ni={Ni_new}, diff={Ni_new - Ni.astype(int)}"
            )

            for local_i, cid in enumerate(sat_indices):
                Ni_target_global[cid] = float(Ni_new[local_i])

            # 변화 없으면 스킵
            if np.array_equal(Ni_new, Ni.astype(int)):
                print(f"[OFFLOAD][plane {p}] no change after offload.")
                continue

            # =========================================================
            # [추가] (현실적) 전송횟수 계산:
            # - sat_indices의 "순서"를 plane 내 인접 순서로 간주
            # - (sender -> receiver)로 옮겨야 하는 샘플 수만큼
            #   hop = |sender_pos - receiver_pos| 를 누적
            # - 샘플 단위로 세면 너무 커질 수 있으니,
            #   "이동 이벤트 1회"를 송신자→수신자 1쌍으로 보고
            #   그 이벤트의 hop을 1회로 누적하는 방식(권장)
            #
            # 여기서는 "이동 이벤트 기반"으로 구현:
            #   - sender와 receiver를 매칭하면서
            #   - move가 발생할 때마다 hop을 1회 누적
            # =========================================================

            Ni_int = Ni.astype(int)
            delta = Ni_new - Ni_int

            senders = [
                (si, int(-delta[si])) for si in range(len(sat_indices)) if delta[si] < 0
            ]
            receivers = [
                (ri, int(delta[ri])) for ri in range(len(sat_indices)) if delta[ri] > 0
            ]

            s_ptr = 0
            r_ptr = 0
            plane_offload_hops = 0

            while s_ptr < len(senders) and r_ptr < len(receivers):
                s_pos, s_need = senders[s_ptr]
                r_pos, r_need = receivers[r_ptr]

                move = min(s_need, r_need)
                if move <= 0:
                    break

                hop_cost = abs(s_pos - r_pos)
                plane_offload_hops += hop_cost

                s_need -= move
                r_need -= move

                # 리스트 업데이트
                senders[s_ptr] = (s_pos, s_need)
                receivers[r_ptr] = (r_pos, r_need)

                if senders[s_ptr][1] == 0:
                    s_ptr += 1
                if receivers[r_ptr][1] == 0:
                    r_ptr += 1

            offload_hops_total += plane_offload_hops
            print(
                f"[OFFLOAD][plane {p}] transmission_count(=hop sum) = {plane_offload_hops}"
            )

            # 3) plane 내 모든 샘플을 하나로 모으기
            X_list = []
            Y_list = []
            for idx in sat_indices:
                X_list.append(client_datasets[idx].X)
                Y_list.append(client_datasets[idx].Y)

            X_all = torch.cat(X_list, dim=0)
            Y_all = torch.cat(Y_list, dim=0)

            # 4) 섞어서 공정하게 다시 나누기 (재현성 위해 기존 rng 사용)
            perm = rng.permutation(X_all.size(0))  # numpy array
            perm_t = torch.from_numpy(perm).long()  # torch LongTensor로 변환
            X_all = X_all[perm_t]
            Y_all = Y_all[perm_t]

            # 5) Ni_new에 맞게 잘라서 다시 위성별로 할당
            start = 0
            for local_i, cid in enumerate(sat_indices):
                cnt = int(Ni_new[local_i])
                end = start + cnt
                client_datasets[cid].X = X_all[start:end]
                client_datasets[cid].Y = Y_all[start:end]
                start = end

            # sanity check
            assert start == X_all.size(0), (
                "[OFFLOAD] index split mismatch in plane redistribution!"
            )

            Ni_after = np.array(
                [len(client_datasets[i]) for i in sat_indices], dtype=float
            )
            print(f"[OFFLOAD][plane {p}] after offload: Ni={Ni_after}")

    # ========= 공통: offloading 이후 class_weight 재계산 =========
    all_y = np.concatenate([ds.Y.numpy().astype(int) for ds in client_datasets])
    n_classes = int(all_y.max()) + 1
    counts = np.bincount(all_y, minlength=n_classes)
    class_weight = np.zeros(n_classes, dtype=np.float32)

    nonzero = counts > 0
    class_weight[nonzero] = len(all_y) / (n_classes * counts[nonzero])

    class_weight_t = torch.from_numpy(class_weight).type(torch.float)

    for ds in client_datasets:
        ds.class_weight = class_weight_t.clone()
        ds.n_classes = n_classes

    print("[OFFLOAD] global class_weight recomputed.")

    if train_capacities is None:
        return None  # swap 모드일 땐 안 씀

    Ni_after_global = np.array([len(ds) for ds in client_datasets], dtype=float)
    return {
        "Ni_before": Ni_before_global,
        "Ni_target": Ni_target_global,
        "Ni_after": Ni_after_global,
        "offload_hops": float(offload_hops_total),
    }


def save_class_distribution_csv(client_datasets, save_csv_path: str):
    """
    각 클라이언트(위성)별 클래스 분포를 CSV로 저장.
    컬럼 예시: client_id, class_0, class_1, ...
    """
    rows = []
    all_classes = sorted(
        {int(c) for ds in client_datasets for c in np.unique(ds.Y.numpy().astype(int))}
    )

    for i, ds in enumerate(client_datasets):
        y = ds.Y.numpy().astype(int)
        classes, counts = np.unique(y, return_counts=True)
        row = {"client_id": i}
        for c in all_classes:
            row[f"class_{c}"] = 0
        for c, cnt in zip(classes, counts):
            row[f"class_{c}"] = int(cnt)
        rows.append(row)

    df = pd.DataFrame(rows)
    dir_name = os.path.dirname(save_csv_path) or "."
    os.makedirs(dir_name, exist_ok=True)
    df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE][CLASSDIST] {save_csv_path}")
    return df
