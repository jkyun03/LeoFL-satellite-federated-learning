# scenarios.py
# FL 시나리오 구현: FedAvg(항상 연결), 간헐 연결, LH-FL(리더 기반 계층형+오프로딩), Decentralized, FedLEO, FedISL.
import copy, os
from typing import List, Optional, Set
import torch
import numpy as np
import pandas as pd

from LeoFL.Simulator.fl_core import (
    build_model,
    train_one_epoch,
    evaluate,
    AGGREGATORS,
    stop_never,
    build_fixed_constellation,
    compute_orbits_for_rounds,
    connected_clients_at_round,
    minutes_to_rounds,
    save_events_only_csv,
    save_time_accuracy_csv,
    SimConfig,
    precompute_meetings_and_select_leaders,
    offload_balance_datasets,
    save_class_distribution_csv,
    reindex_satellites_by_orbital_phase,
    build_transfer_plan_within_plane,
    sample_train_rounds_with_exp_jitter,
    filter_by_link_failure,
    link_success,
    jitter_hop_count,
    LINK_FAIL_P_DEFAULT,
)

from LeoFL.Simulator.comm_Logger import CommLogger, calc_model_bytes, calc_sample_bytes


class FLScenario:
    """시나리오 공통 베이스 클래스"""

    def __init__(
        self,
        client_datasets,
        test_dataset,
        config: SimConfig,
        link_fail_rng: Optional[np.random.Generator] = None,
        link_fail_p: float = LINK_FAIL_P_DEFAULT,
    ):
        assert len(client_datasets) > 0
        self.client_datasets = client_datasets
        self.test_dataset = test_dataset
        self.cfg = config
        self.stop_fn = stop_never()
        self.aggregate_fn = AGGREGATORS[self.cfg.aggregate_name]
        # --- comm logging ---
        self.comm_logger = CommLogger(seconds_per_round=self.cfg.seconds_per_round)
        self.rng = np.random.default_rng(0)
        self.train_time_log = []

        # --- 통신 링크 랜덤성 (옵트인; None이면 기존 결정적 동작 그대로) ---
        self.link_rng = link_fail_rng
        self.link_fail_p = link_fail_p

        input_size = getattr(test_dataset, "n_features", None)
        num_classes = test_dataset.n_classes

        self.global_model = build_model(
            model_name=self.cfg.model_name,
            num_classes=num_classes,
            input_size=input_size,
            hidden_size=self.cfg.hidden_size,
        ).to(self.cfg.device)

        self.client_models = [
            copy.deepcopy(self.global_model).to(self.cfg.device)
            for _ in client_datasets
        ]

        self.model_bytes = calc_model_bytes(self.global_model)

        if getattr(test_dataset, "n_features", None) is not None:
            self.sample_bytes = calc_sample_bytes(n_features=test_dataset.n_features)
        else:
            sample_shape = test_dataset.X[0].shape
            self.sample_bytes = calc_sample_bytes(sample_shape=sample_shape)

        self.n_clients = len(client_datasets)
        self.acc_log = [(0, 0.0)]
        self.best_acc = -1.0

    def after_aggregate(self, round_idx: int):
        acc = evaluate(self.global_model, self.test_dataset, device=self.cfg.device)
        self.acc_log.append((round_idx, acc))
        self.best_acc = max(self.best_acc, acc)

    def finalize_logs(self, prefix: str):
        os.makedirs(self.cfg.log_dir or ".", exist_ok=True)

        acc_csv = os.path.join(self.cfg.log_dir or ".", f"{prefix}_accuracy.csv")
        save_time_accuracy_csv(self.acc_log, acc_csv)

        comm_csv = os.path.join(self.cfg.log_dir or ".", f"{prefix}_comm.csv")
        self.comm_logger.save_csv(comm_csv)

        if getattr(self, "train_time_log", None) and len(self.train_time_log) > 0:
            tt_csv = os.path.join(self.cfg.log_dir or ".", f"{prefix}_train_time.csv")
            df = pd.DataFrame(
                self.train_time_log, columns=["round", "sat_id", "train_seconds"]
            )
            df.to_csv(tt_csv, index=False, encoding="utf-8-sig")
            print(f"[SAVE][TRAIN_TIME] {tt_csv}")

    def run(self):
        raise NotImplementedError


class AlwaysConnectedScenario(FLScenario):
    def __init__(
        self,
        client_datasets,
        test_dataset,
        config: SimConfig,
        train_minutes: int,
        per_sat_minutes: Optional[List[int]] = None,
        log_prefix: str = "sat_events_always",
    ):
        super().__init__(client_datasets, test_dataset, config)

        self.log_prefix = log_prefix
        self.train_minutes = train_minutes
        self.per_sat_minutes = per_sat_minutes  # <-- 이 줄이 없어서 터진 것

        # (선택) 기존 로직 유지해도 되지만, run()에서 jitter로 remaining_train을 새로 뽑으면
        # 아래 train_rounds_needed는 사실상 참고용이 됨
        if per_sat_minutes is not None:
            assert len(per_sat_minutes) == len(client_datasets)
            self.train_rounds_needed = [
                minutes_to_rounds(m, self.cfg.seconds_per_round)
                for m in per_sat_minutes
            ]
        else:
            fixed = minutes_to_rounds(train_minutes, self.cfg.seconds_per_round)
            self.train_rounds_needed = [fixed] * self.n_clients

    def run(self):
        global_version = 0
        downloaded_version = [-1] * self.n_clients
        remaining_train = [0] * self.n_clients
        trained_ready = [False] * self.n_clients
        uploaded: Set[int] = set()

        for round_idx in range(self.cfg.num_rounds):
            # 다운링크: always는 항상 가능
            for i in range(self.n_clients):
                if downloaded_version[i] < global_version:
                    self.client_models[i].load_state_dict(
                        copy.deepcopy(self.global_model.state_dict())
                    )
                    downloaded_version[i] = global_version
                    sampled_rounds = sample_train_rounds_with_exp_jitter(
                        mean_minutes=(
                            self.per_sat_minutes[i]
                            if self.per_sat_minutes is not None
                            else self.train_minutes
                        ),
                        seconds_per_round=self.cfg.seconds_per_round,
                        jitter_sec=0.5,
                        clamp=True,
                        rng=self.rng,
                    )

                    remaining_train[i] = sampled_rounds

                    # 학습 시간 로그 (초 단위)
                    self.train_time_log.append(
                        (
                            round_idx,
                            i,
                            float(sampled_rounds * self.cfg.seconds_per_round),
                        )
                    )

                    trained_ready[i] = False

            # 로컬 학습 카운트다운 + 1 epoch
            for i in range(self.n_clients):
                if downloaded_version[i] == global_version and remaining_train[i] > 0:
                    remaining_train[i] -= 1
                    if remaining_train[i] == 0:
                        _ = train_one_epoch(
                            self.client_models[i],
                            self.client_datasets[i],
                            device=self.cfg.device,
                            batch_size=32,
                            lr=self.cfg.lr,
                            class_weight=self.client_datasets[i].class_weight,
                        )
                        trained_ready[i] = True
                        uploaded.add(i)

            # 전역 집계 (동기 FedAvg)
            if len(uploaded) == self.n_clients:
                self.aggregate_fn(self.global_model, self.client_models, list(uploaded))
                K = self.n_clients
                model_hops = 2 * K
                self.comm_logger.log(
                    round_idx,
                    model_bytes=2 * K * self.model_bytes,
                    model_hops=model_hops,
                    note="fedavg_ideal",
                )
                self.after_aggregate(round_idx)

                if self.acc_log[-1][1] >= self.cfg.target_acc:
                    print(
                        f"[INFO] Target accuracy {self.cfg.target_acc}% reached at round {round_idx}. (continue)"
                    )

                global_version += 1
                uploaded.clear()
                for i in range(self.n_clients):
                    remaining_train[i] = 0
                    trained_ready[i] = False
                    downloaded_version[i] = -1

        self.finalize_logs(prefix=self.log_prefix)  # <-- prefix 통일
        return self.global_model, self.best_acc


class ConnectivityScenario(FLScenario):
    """연결성 반영 + 고정 학습시간 (기존 base 함수)"""

    def __init__(
        self,
        client_datasets,
        test_dataset,
        config: SimConfig,
        train_minutes: int,
        log_prefix="sat_events_base",
    ):
        super().__init__(client_datasets, test_dataset, config)
        self.log_prefix = log_prefix
        self.train_minutes = train_minutes
        self.per_sat_minutes = None
        self.train_rounds_needed = minutes_to_rounds(
            train_minutes, self.cfg.seconds_per_round
        )
        self.sats = build_fixed_constellation(self.cfg.constellation_configs)
        compute_orbits_for_rounds(
            self.sats,
            num_rounds=self.cfg.num_rounds,
            seconds_per_round=self.cfg.seconds_per_round,
        )
        self.events_log = []
        self.prev_connected = [False] * self.n_clients

    def _log(self, i, t, label):
        self.events_log.append((t, f"sat_{i + 1}", label))

    def run(self):
        global_version = 0
        downloaded_version = [-1] * self.n_clients
        uploaded: Set[int] = set()
        remaining_train = [0] * self.n_clients
        trained_ready = [False] * self.n_clients

        for round_idx in range(self.cfg.num_rounds):
            connected = connected_clients_at_round(self.sats, round_idx)
            # 연결 이벤트 로깅
            for i in range(self.n_clients):
                is_conn = i in connected
                if is_conn and not self.prev_connected[i]:
                    self._log(i, round_idx, "GS connect")
                if (not is_conn) and self.prev_connected[i]:
                    self._log(i, round_idx, "GS disconnect")

                self.prev_connected[i] = is_conn

            # 다운링크
            for i in connected:
                if downloaded_version[i] < global_version:
                    self.client_models[i].load_state_dict(
                        copy.deepcopy(self.global_model.state_dict())
                    )
                    downloaded_version[i] = global_version
                    sampled_rounds = sample_train_rounds_with_exp_jitter(
                        mean_minutes=(
                            self.per_sat_minutes[i]
                            if self.per_sat_minutes is not None
                            else self.train_minutes
                        ),
                        seconds_per_round=self.cfg.seconds_per_round,
                        jitter_sec=0.5,
                        clamp=True,
                        rng=self.rng,
                    )
                    remaining_train[i] = sampled_rounds
                    self.train_time_log.append(
                        (
                            round_idx,
                            i,
                            float(sampled_rounds * self.cfg.seconds_per_round),
                        )
                    )

                    trained_ready[i] = False
                    self._log(i, round_idx, "get_global")

            # 로컬 학습 카운트다운 + 1 epoch
            for i in range(self.n_clients):
                if downloaded_version[i] == global_version and remaining_train[i] > 0:
                    remaining_train[i] -= 1
                    if remaining_train[i] == 0:
                        _ = train_one_epoch(
                            self.client_models[i],
                            self.client_datasets[i],
                            device=self.cfg.device,
                            batch_size=32,
                            lr=self.cfg.lr,
                            class_weight=self.client_datasets[i].class_weight,
                        )
                        trained_ready[i] = True
                        self._log(i, round_idx, "local")

            # 업링크(연결된 상태여야 참여로 인정)
            for i in connected:
                if (
                    downloaded_version[i] == global_version
                    and trained_ready[i]
                    and (i not in uploaded)
                ):
                    uploaded.add(i)

            if len(uploaded) == self.n_clients:
                self.aggregate_fn(self.global_model, self.client_models, list(uploaded))
                K = self.n_clients
                model_hops = 2 * K
                self.comm_logger.log(
                    round_idx,
                    model_bytes=2 * K * self.model_bytes,
                    model_hops=model_hops,
                    note="fedavg_gs",
                )
                self.after_aggregate(round_idx)
                if self.acc_log[-1][1] >= self.cfg.target_acc:
                    print(
                        f"[INFO] Target accuracy {self.cfg.target_acc}% reached at round {round_idx}. (continue)"
                    )
                global_version += 1
                uploaded.clear()
                for i in range(self.n_clients):
                    remaining_train[i] = 0
                    trained_ready[i] = False

        save_events_only_csv(self.events_log, self.cfg.log_dir, self.log_prefix)
        self.finalize_logs(prefix=self.log_prefix)
        return self.global_model, self.best_acc


class ConnectivityVarTrainScenario(FLScenario):
    """연결성 반영 + 위성별 학습시간 가변 (기존 vartrain 함수)"""

    def __init__(
        self,
        client_datasets,
        test_dataset,
        config: SimConfig,
        per_sat_minutes: List[int],
        log_prefix="sat_events_vartrain",
    ):
        super().__init__(client_datasets, test_dataset, config)
        assert len(per_sat_minutes) == len(client_datasets), (
            "per_sat_minutes 길이가 클라이언트 수와 동일해야 합니다."
        )
        self.log_prefix = log_prefix
        self.train_rounds_needed = [
            minutes_to_rounds(m, self.cfg.seconds_per_round) for m in per_sat_minutes
        ]
        self.sats = build_fixed_constellation(self.cfg.constellation_configs)
        compute_orbits_for_rounds(
            self.sats,
            num_rounds=self.cfg.num_rounds,
            seconds_per_round=self.cfg.seconds_per_round,
        )
        self.events_log = []
        self.prev_connected = [False] * self.n_clients

    def _log(self, i, t, label):
        self.events_log.append((t, f"sat_{i + 1}", label))

    def run(self):
        global_version = 0
        downloaded_version = [-1] * self.n_clients
        uploaded: Set[int] = set()
        remaining_train = [0] * self.n_clients
        trained_ready = [False] * self.n_clients

        for round_idx in range(self.cfg.num_rounds):
            connected = [
                ci
                for ci in connected_clients_at_round(self.sats, round_idx)
                if 0 <= int(ci) < self.n_clients
            ]

            # 연결 이벤트
            for i in range(self.n_clients):
                is_conn = i in connected
                if is_conn and not self.prev_connected[i]:
                    self._log(i, round_idx, "GS connect")

                if (not is_conn) and self.prev_connected[i]:
                    self._log(i, round_idx, "GS disconnect")

                self.prev_connected[i] = is_conn

            # 다운링크
            for i in connected:
                if downloaded_version[i] < global_version:
                    self.client_models[i].load_state_dict(
                        copy.deepcopy(self.global_model.state_dict())
                    )
                    downloaded_version[i] = global_version
                    sampled_rounds = sample_train_rounds_with_exp_jitter(
                        mean_minutes=(
                            self.train_rounds_needed[i] * self.cfg.seconds_per_round
                        )
                        / 60.0,
                        seconds_per_round=self.cfg.seconds_per_round,
                        jitter_sec=0.5,
                        clamp=True,
                        rng=self.rng,
                    )
                    remaining_train[i] = sampled_rounds
                    self.train_time_log.append(
                        (
                            round_idx,
                            i,
                            float(sampled_rounds * self.cfg.seconds_per_round),
                        )
                    )

                    trained_ready[i] = False

                    self._log(i, round_idx, "get_global")

            # 로컬 학습
            for i in range(self.n_clients):
                if downloaded_version[i] == global_version and remaining_train[i] > 0:
                    remaining_train[i] -= 1
                    if remaining_train[i] == 0:
                        _ = train_one_epoch(
                            self.client_models[i],
                            self.client_datasets[i],
                            device=self.cfg.device,
                            batch_size=32,
                            lr=self.cfg.lr,
                            class_weight=self.client_datasets[i].class_weight,
                        )
                        trained_ready[i] = True
                        self._log(i, round_idx, "local")

            # 업링크
            for i in connected:
                if (
                    (downloaded_version[i] == global_version)
                    and trained_ready[i]
                    and (i not in uploaded)
                ):
                    uploaded.add(i)

            # 전역 업데이트
            if len(uploaded) == self.n_clients:
                self.aggregate_fn(self.global_model, self.client_models, list(uploaded))

                # accuracy 기록 (round 기준)
                self.after_aggregate(round_idx)

                # 목표 도달 메시지만 출력하고 계속 진행
                if self.acc_log[-1][1] >= self.cfg.target_acc:
                    print(
                        f"[INFO] Target accuracy {self.cfg.target_acc}% reached "
                        f"at round {round_idx}. (continue)"
                    )

                # 다음 global version 준비
                global_version += 1
                uploaded.clear()
                for i in range(self.n_clients):
                    remaining_train[i] = 0
                    trained_ready[i] = False

        save_events_only_csv(self.events_log, self.cfg.log_dir, self.log_prefix)
        self.finalize_logs(prefix=self.log_prefix)
        return self.global_model, self.best_acc


def ring_hop_distance(local_a: int, local_b: int, m: int) -> int:
    """plane 내부 ring에서 local index a->b 최단 hop 수"""
    d = abs(local_a - local_b)
    return min(d, m - d)


def _plane_maps_from_configs(configs, n_clients: int):
    plane_ids = [configs[i].get("plane_id", 0) for i in range(n_clients)]
    plane_to_sats = {}
    for i, p in enumerate(plane_ids):
        plane_to_sats.setdefault(p, []).append(i)
    return plane_ids, plane_to_sats


def _ring_distance_in_plane(sat_idx: int, ref_idx: int, sat_indices: list) -> int:
    """
    plane 내부 ring에서 sat_idx와 ref_idx 사이의 최단 hop 거리.
    sat_indices는 plane 내 satellite index 순서.
    """
    m = len(sat_indices)
    if m <= 1:
        return 0

    pos = {sid: k for k, sid in enumerate(sat_indices)}
    a = pos[sat_idx]
    b = pos[ref_idx]
    d = abs(a - b)
    return min(d, m - d)


def _connected_duration_from(
    sats, sat_idx: int, start_round: int, max_rounds: int
) -> int:
    """
    start_round부터 해당 satellite가 GS와 연결 상태를 유지하는 길이.
    현재 simulator의 GS visibility 함수를 그대로 사용.
    """
    duration = 0
    for t in range(start_round, max_rounds):
        conn = connected_clients_at_round(sats, t)
        if sat_idx in conn:
            duration += 1
        else:
            break
    return duration


def _choose_fedisl_sink(sats, sat_indices: list, future_round: int, max_rounds: int):
    """
    FedISL 논문 설명에 맞춰, aggregation 완료 예상 시점 future_round에서
    PS와 연결 가능한 satellite 중 남은 contact duration이 가장 긴 satellite를 sink로 선택.
    후보가 없으면 future_round 이후 가장 먼저 연결되는 satellite를 선택.
    """
    best_sat = None
    best_dur = -1

    if future_round >= max_rounds:
        return sat_indices[0]

    connected_future = set(connected_clients_at_round(sats, future_round))

    for sid in sat_indices:
        if sid in connected_future:
            dur = _connected_duration_from(sats, sid, future_round, max_rounds)
            if dur > best_dur:
                best_dur = dur
                best_sat = sid

    if best_sat is not None:
        return best_sat

    # fallback: future_round 이후 가장 먼저 GS에 연결되는 위성
    for t in range(future_round, max_rounds):
        connected = set(connected_clients_at_round(sats, t))
        candidates = [sid for sid in sat_indices if sid in connected]
        if len(candidates) > 0:
            return candidates[0]

    # 끝까지 없으면 plane 첫 위성
    return sat_indices[0]


def _add_undirected_edge(neighbors, a: int, b: int):
    """Undirected graph에 edge 추가."""
    if a == b:
        return
    neighbors[a].add(b)
    neighbors[b].add(a)


def _build_ring_neighbors(n_nodes: int):
    """
    fallback용 ring topology.
    각 node는 양옆 node와 연결됨.
    """
    neighbors = {i: set() for i in range(n_nodes)}

    if n_nodes <= 1:
        return neighbors

    for i in range(n_nodes):
        _add_undirected_edge(neighbors, i, (i + 1) % n_nodes)

    return neighbors


def _to_numpy_meeting_counts(meeting_counts):
    """
    meeting_counts가 numpy array든 pandas DataFrame이든 안전하게 numpy matrix로 변환.
    """
    if isinstance(meeting_counts, pd.DataFrame):
        numeric = meeting_counts.select_dtypes(include=[np.number])
        arr = numeric.to_numpy()

        # sat_id 같은 index column이 포함된 경우 제거
        if arr.ndim == 2 and arr.shape[1] == arr.shape[0] + 1:
            arr = arr[:, 1:]

        return arr

    return np.asarray(meeting_counts)


def _build_topk_peer_neighbors_from_meetings(
    meeting_counts,
    n_nodes: int,
    top_k: int = 2,
):
    """
    Decentralized FL baseline용 peer graph 생성.

    각 satellite가 cumulative contact count 기준으로 자주 만나는 top-k satellite와 연결된다.
    이 graph 위에서 peer-to-peer model mixing을 수행한다.

    주의:
    - 여기서는 round마다 변하는 dynamic graph가 아니라,
      cumulative meeting frequency 기반의 static neighbor graph를 사용한다.
    - 고립 node가 생기면 ring topology로 fallback한다.
    """
    neighbors = {i: set() for i in range(n_nodes)}

    try:
        arr = _to_numpy_meeting_counts(meeting_counts)

        if arr.ndim == 2 and arr.shape[0] >= n_nodes and arr.shape[1] >= n_nodes:
            for i in range(n_nodes):
                scores = []

                for j in range(n_nodes):
                    if i == j:
                        continue

                    score = float(arr[i, j]) + float(arr[j, i])

                    if score > 0:
                        scores.append((score, j))

                scores.sort(key=lambda x: x[0], reverse=True)

                for _, j in scores[:top_k]:
                    _add_undirected_edge(neighbors, i, j)

    except Exception as e:
        print(f"[WARN] Failed to build peer graph from meeting_counts: {e}")

    # fallback: isolated node 방지
    ring_neighbors = _build_ring_neighbors(n_nodes)
    for i in range(n_nodes):
        if len(neighbors[i]) == 0:
            for j in ring_neighbors[i]:
                _add_undirected_edge(neighbors, i, j)

    return neighbors


def _build_topk_leader_plane_neighbors(
    leaders_per_plane,
    meeting_counts,
    top_k: int = 2,
):
    """
    LH-FL의 inter-orbit leader decentralized mixing용 graph 생성.

    각 orbital leader는 cumulative contact count 기준으로
    자주 만나는 다른 plane의 leader top-k와 model mixing을 수행한다.
    """
    planes = sorted(leaders_per_plane.keys())
    neighbors = {p: set() for p in planes}

    try:
        arr = _to_numpy_meeting_counts(meeting_counts)

        for p in planes:
            leader_p = leaders_per_plane[p]
            scores = []

            for q in planes:
                if p == q:
                    continue

                leader_q = leaders_per_plane[q]
                score = float(arr[leader_p, leader_q]) + float(arr[leader_q, leader_p])

                if score > 0:
                    scores.append((score, q))

            scores.sort(key=lambda x: x[0], reverse=True)

            for _, q in scores[:top_k]:
                neighbors[p].add(q)
                neighbors[q].add(p)

    except Exception as e:
        print(f"[WARN] Failed to build leader graph from meeting_counts: {e}")

    # fallback: leader graph 고립 방지
    if len(planes) > 1:
        for idx, p in enumerate(planes):
            if len(neighbors[p]) == 0:
                prev_p = planes[(idx - 1) % len(planes)]
                next_p = planes[(idx + 1) % len(planes)]

                neighbors[p].add(prev_p)
                neighbors[p].add(next_p)
                neighbors[prev_p].add(p)
                neighbors[next_p].add(p)

    return neighbors


def _weighted_average_state_dict_from_models(models, weights=None):
    """
    여러 model의 state_dict를 weighted average.
    """
    if len(models) == 0:
        raise ValueError("models must not be empty.")

    if weights is None:
        weights = [1.0] * len(models)

    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.clip(weights.sum(), 1e-12, None)

    states = [m.state_dict() for m in models]
    avg_state = {}

    for key in states[0].keys():
        ref = states[0][key]

        if torch.is_floating_point(ref):
            acc = torch.zeros_like(ref, dtype=torch.float32)

            for st, w in zip(states, weights):
                acc += st[key].detach().float() * float(w)

            avg_state[key] = acc.to(dtype=ref.dtype)
        else:
            # non-floating buffer는 첫 번째 model 값을 유지
            avg_state[key] = ref.clone()

    return avg_state


def _mix_self_with_neighbor_average(
    self_model,
    neighbor_models,
    beta: float = 0.5,
    neighbor_weights=None,
):
    """
    Approach 식과 같은 leader-level mixing 구현.

    \tilde{w}_p = (1 - beta) w_p + beta * sum_q pi_pq w_q

    neighbor가 없으면 자기 model 유지.
    """
    if len(neighbor_models) == 0 or beta <= 0:
        return copy.deepcopy(self_model.state_dict())

    beta = float(np.clip(beta, 0.0, 1.0))

    own_state = self_model.state_dict()
    neighbor_avg = _weighted_average_state_dict_from_models(
        neighbor_models,
        weights=neighbor_weights,
    )

    mixed_state = {}

    for key in own_state.keys():
        own_val = own_state[key]

        if torch.is_floating_point(own_val):
            mixed = (1.0 - beta) * own_val.detach().float() + beta * neighbor_avg[
                key
            ].detach().float()
            mixed_state[key] = mixed.to(dtype=own_val.dtype)
        else:
            mixed_state[key] = own_val.clone()

    return mixed_state


def _evaluate_weighted_model_accuracy(models, weights, test_dataset, device):
    """
    단일 global model이 없는 decentralized / LH-FL 평가용.

    여러 local model 또는 plane-level model의 test accuracy를 계산한 뒤
    weighted average로 대표 accuracy를 산출한다.
    """
    if len(models) == 0:
        return 0.0

    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.clip(weights.sum(), 1e-12, None)

    acc = 0.0
    for model, w in zip(models, weights):
        acc += float(w) * evaluate(model, test_dataset, device=device)

    return acc


class OffloadScenario(FLScenario):
    """
    LH-FL:
      - 궤도 plane별 leader 선정
      - capacity-aware one-shot data offloading
      - intra-orbit: plane 내부 leader-centered FedAvg
      - inter-orbit: leader들 간 decentralized model mixing
      - 단일 global model을 training에 사용하지 않고, plane-level mixed model을 유지
    """

    def __init__(
        self,
        client_datasets,
        test_dataset,
        config: SimConfig,
        max_dist_km: float = 2000.0,
        offload_ratio: float = 0.1,
        top_pairs_per_sat: int = 1,
        log_prefix: str = "sat_events_offload",
        do_offload: bool = True,
        per_sat_minutes: Optional[List[int]] = None,
        train_minutes: int = 10,
        leader_top_k: int = 2,
        leader_mixing_beta: float = 0.5,
        link_fail_rng: Optional[np.random.Generator] = None,
        link_fail_p: float = LINK_FAIL_P_DEFAULT,
    ):
        super().__init__(
            client_datasets,
            test_dataset,
            config,
            link_fail_rng=link_fail_rng,
            link_fail_p=link_fail_p,
        )
        self.max_dist_km = max_dist_km
        self.offload_ratio = offload_ratio
        self.top_pairs_per_sat = top_pairs_per_sat
        self.log_prefix = log_prefix
        self.do_offload = do_offload
        self.leader_top_k = leader_top_k
        self.leader_mixing_beta = leader_mixing_beta

        # --- 위성별 학습 주기(빈도) ---
        if per_sat_minutes is not None:
            assert len(per_sat_minutes) == len(client_datasets), (
                "per_sat_minutes 길이가 클라이언트 수와 동일해야 합니다."
            )
            mins = np.asarray(per_sat_minutes, dtype=float)
        else:
            mins = np.full(self.n_clients, float(train_minutes))

        # --- 핵심: capacity는 학습시간(min)과 역비례 ---
        # capacity_i ∝ 1 / minutes_i  (빠를수록 큼)
        cap = 1.0 / np.clip(mins, 1e-6, None)
        cap = cap / cap.mean()  # 평균 1로 정규화
        self.train_capacities = cap
        print("[SIM][offload] per_sat_minutes(min) =", mins.tolist())
        print("[SIM][offload] train_capacities(norm) =", self.train_capacities.tolist())

        # baseline time model: T_i = C * (N_i / capacity_i)
        # C를 현재 초기 상태에서 기존 per_sat_minutes와 대략 맞도록 설정
        Ni0 = np.array([len(ds) for ds in client_datasets], dtype=float)
        ratio0 = Ni0 / np.clip(self.train_capacities, 1e-6, None)

        # 기존 mins를 최대한 유지하도록 least-squares 비슷하게 상수 맞춤
        self.time_scale_const = float(np.mean(mins / np.clip(ratio0, 1e-6, None)))

        print("[SIM][offload] initial Ni =", Ni0.tolist())
        print("[SIM][offload] time_scale_const =", self.time_scale_const)

    def _current_mean_minutes(self, sat_idx: int) -> float:
        Ni = float(len(self.client_datasets[sat_idx]))
        Si = float(self.train_capacities[sat_idx])

        mean_min = self.time_scale_const * (Ni / max(Si, 1e-6))

        mean_min = max(mean_min, 0.5)
        return mean_min

    def run(self):

        n_clients = self.n_clients
        print(
            f"[SIM][offload] start: num_rounds={self.cfg.num_rounds}, n_clients={n_clients}"
        )

        # --- offloading 전 클래스 분포 저장 ---
        before_path = os.path.join(
            self.cfg.log_dir or ".",
            f"{self.log_prefix}_classdist_before.csv",
        )
        save_class_distribution_csv(self.client_datasets, before_path)

        # 1) 위성/궤도 구성 및 궤도 계산
        configs = self.cfg.constellation_configs[:n_clients]
        sats = build_fixed_constellation(configs)
        assert len(sats) == n_clients, (
            "위성 개수와 client_datasets 개수는 같아야 합니다."
        )

        compute_orbits_for_rounds(
            sats,
            num_rounds=self.cfg.num_rounds,
            seconds_per_round=self.cfg.seconds_per_round,
        )

        # plane_id는 constellation_configs 안에 있다고 가정, 없으면 0으로 fallback
        plane_ids = [c.get("plane_id", 0) for c in configs]

        sats, plane_ids = reindex_satellites_by_orbital_phase(
            sats, plane_ids, round_idx=0
        )

        # 2) meeting_counts 및 plane leader 선정
        meeting_counts, leaders_per_plane, plane_to_sats = (
            precompute_meetings_and_select_leaders(
                sats,
                plane_ids=plane_ids,
                num_rounds=self.cfg.num_rounds,
                seconds_per_round=self.cfg.seconds_per_round,
                max_dist_km=self.max_dist_km,
                log_dir=self.cfg.log_dir,
                filename=f"{self.log_prefix}_meeting_counts.csv",
            )
        )

        print("[LEADER][offload] per plane:")
        for p, leader_idx in leaders_per_plane.items():
            print(
                f"  plane={p}, leader_sat=sat_{leader_idx + 1}, members={plane_to_sats[p]}"
            )
            # 2-1) leader 간 decentralized mixing graph 구성
        leader_neighbors = _build_topk_leader_plane_neighbors(
            leaders_per_plane=leaders_per_plane,
            meeting_counts=meeting_counts,
            top_k=self.leader_top_k,
        )

        print("[LEADER-GRAPH][offload] inter-orbit leader neighbors:")
        for p in sorted(leader_neighbors.keys()):
            neigh = sorted(leader_neighbors[p])
            print(f"  plane={p} -> neighbor_planes={neigh}")

            # -------------------------------------------------
        # Save LH-FL overview logs for 3D visualization
        # These files are used by plot_lhfl_overview_3d_from_logs.py
        # -------------------------------------------------
        overview_dir = self.cfg.log_dir or "."
        os.makedirs(overview_dir, exist_ok=True)

        # 1) plane members
        plane_member_rows = []
        for p, sat_indices in plane_to_sats.items():
            for local_idx, sat_id in enumerate(sat_indices):
                plane_member_rows.append(
                    {
                        "plane_id": int(p),
                        "local_idx": int(local_idx),
                        "sat_id": int(sat_id),
                        "sat_name": f"sat_{int(sat_id) + 1}",
                    }
                )

        pd.DataFrame(plane_member_rows).to_csv(
            os.path.join(overview_dir, f"{self.log_prefix}_plane_members.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        # 2) leaders
        leader_rows = []
        for p, leader_idx in leaders_per_plane.items():
            leader_rows.append(
                {
                    "plane_id": int(p),
                    "leader_sat_id": int(leader_idx),
                    "leader_name": f"sat_{int(leader_idx) + 1}",
                }
            )

        pd.DataFrame(leader_rows).to_csv(
            os.path.join(overview_dir, f"{self.log_prefix}_leaders.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        # 3) inter-plane leader exchange edges
        # p < q 조건으로 중복 edge 제거
        leader_exchange_rows = []
        for p in sorted(leader_neighbors.keys()):
            for q in sorted(leader_neighbors[p]):
                if int(p) < int(q):
                    src_leader = int(leaders_per_plane[p])
                    dst_leader = int(leaders_per_plane[q])

                    leader_exchange_rows.append(
                        {
                            "src_plane": int(p),
                            "dst_plane": int(q),
                            "src_leader": src_leader,
                            "dst_leader": dst_leader,
                            "src_name": f"sat_{src_leader + 1}",
                            "dst_name": f"sat_{dst_leader + 1}",
                        }
                    )

        pd.DataFrame(leader_exchange_rows).to_csv(
            os.path.join(overview_dir, f"{self.log_prefix}_leader_exchange.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        print(f"[SAVE][OVERVIEW] {self.log_prefix}_plane_members.csv")
        print(f"[SAVE][OVERVIEW] {self.log_prefix}_leaders.csv")
        print(f"[SAVE][OVERVIEW] {self.log_prefix}_leader_exchange.csv")

        cap_path = os.path.join(
            self.cfg.log_dir or ".",
            f"{self.log_prefix}_capacity.csv",
        )
        rows = []
        for i, cap in enumerate(self.train_capacities):
            rows.append(
                {
                    "sat_id": i,  # 0-based index
                    "sat_name": f"sat_{i + 1}",  # 사람이 보기 편한 이름
                    "plane_id": plane_ids[i],
                    "train_capacity": float(cap),
                }
            )
        df = pd.DataFrame(rows)
        os.makedirs(self.cfg.log_dir or ".", exist_ok=True)
        df.to_csv(cap_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE][CAPACITY] {cap_path}")

        # 3) offloading으로 non-IID 완화
        if self.do_offload:
            offload_stats = offload_balance_datasets(
                client_datasets=self.client_datasets,
                meeting_counts=meeting_counts,
                plane_ids=plane_ids,
                offload_ratio=self.offload_ratio,
                top_pairs_per_sat=self.top_pairs_per_sat,
                seed=0,
                train_capacities=self.train_capacities,
            )
        else:
            offload_stats = None

        offload_hops_total = 0
        offload_bytes_total = 0
        offload_samples_total = 0
        offload_rows = []

        if self.do_offload:
            for p, sat_indices in plane_to_sats.items():
                m = len(sat_indices)
                if m <= 1:
                    continue

                Ni_plane_before = np.array(
                    [offload_stats["Ni_before"][i] for i in sat_indices],
                    dtype=int,
                )
                Ni_plane_target = np.array(
                    [offload_stats["Ni_target"][i] for i in sat_indices],
                    dtype=int,
                )

                plan = build_transfer_plan_within_plane(
                    Ni_plane_before,
                    Ni_plane_target,
                )

                for s_local, r_local, delta in plan:
                    hop = ring_hop_distance(s_local, r_local, m)
                    moved_samples = int(abs(delta))

                    if moved_samples <= 0:
                        continue

                    # local index within plane -> global satellite id
                    src_sat_id = int(sat_indices[s_local])
                    dst_sat_id = int(sat_indices[r_local])

                    # hop 기준 비용
                    offload_hops_total += int(hop)

                    # byte 기준 비용
                    offload_bytes = int(moved_samples * self.sample_bytes * hop)
                    offload_bytes_total += offload_bytes

                    offload_samples_total += moved_samples

                    # visualization용 실제 offloading pair 기록
                    offload_rows.append(
                        {
                            "plane_id": int(p),
                            "src_sat_id": src_sat_id,
                            "dst_sat_id": dst_sat_id,
                            "src_name": f"sat_{src_sat_id + 1}",
                            "dst_name": f"sat_{dst_sat_id + 1}",
                            "moved_samples": int(moved_samples),
                            "hop": int(hop),
                            "sample_bytes": int(self.sample_bytes),
                            "offload_bytes": int(offload_bytes),
                        }
                    )

        self.comm_logger.log(
            0,
            model_bytes=0,
            offload_bytes=int(offload_bytes_total),
            model_hops=0,
            offload_hops=int(offload_hops_total),
            note=(
                f"one_shot_offload,"
                f"samples={offload_samples_total},"
                f"sample_bytes={self.sample_bytes},"
                f"hops={offload_hops_total}"
            ),
        )

        pd.DataFrame(offload_rows).to_csv(
            os.path.join(
                self.cfg.log_dir or ".",
                f"{self.log_prefix}_offload_pairs.csv",
            ),
            index=False,
            encoding="utf-8-sig",
        )

        print(f"[SAVE][OVERVIEW] {self.log_prefix}_offload_pairs.csv")

        print(f"[SAVE][OVERVIEW] {self.log_prefix}_offload_pairs.csv")

        after_path = os.path.join(
            self.cfg.log_dir or ".",
            f"{self.log_prefix}_classdist_after.csv",
        )
        save_class_distribution_csv(self.client_datasets, after_path)

        if offload_stats is not None:
            cap_csv = os.path.join(
                self.cfg.log_dir or ".",
                f"{self.log_prefix}_capacity.csv",
            )
            rows = []
            for i in range(n_clients):
                rows.append(
                    {
                        "sat_id": i,
                        "plane_id": plane_ids[i],
                        "train_capacity": float(self.train_capacities[i]),
                        "Ni_before": float(offload_stats["Ni_before"][i]),
                        "Ni_target": float(offload_stats["Ni_target"][i]),
                        "Ni_after": float(offload_stats["Ni_after"][i]),
                    }
                )
            df = pd.DataFrame(rows)
            df.to_csv(cap_csv, index=False, encoding="utf-8-sig")
            print(f"[SAVE][CAPACITY] {cap_csv}")

        # 4) 모델 초기화 (FLScenario에서 이미 global_model/client_models 만들어둠)
        #    plane별 글로벌 모델 추가
        plane_global_models = {
            p: copy.deepcopy(self.global_model).to(self.cfg.device)
            for p in plane_to_sats.keys()
        }

        # 로그 초기화
        self.acc_log = [(0, 0.0)]
        self.best_acc = -1.0

        # -------------------------------------------------
        # offloading 이후의 dataset size를 기준으로
        # 첫 communication cycle의 training time을 샘플링
        # -------------------------------------------------
        remaining_train = []
        for i in range(n_clients):
            mean_minutes_i = self._current_mean_minutes(i)
            sampled_rounds = sample_train_rounds_with_exp_jitter(
                mean_minutes=mean_minutes_i,
                seconds_per_round=self.cfg.seconds_per_round,
                jitter_sec=0.5,
                clamp=True,
                rng=self.rng,
            )
            remaining_train.append(sampled_rounds)

            self.train_time_log.append(
                (0, i, float(sampled_rounds * self.cfg.seconds_per_round))
            )

        completed_this_cycle: Set[int] = set()

        for round_idx in range(1, self.cfg.num_rounds + 1):
            # (1) 로컬 학습 (위성별 주기 반영)
            for i in range(n_clients):
                if self.client_datasets[i] is None or len(self.client_datasets[i]) == 0:
                    continue

                if remaining_train[i] > 0:
                    remaining_train[i] -= 1

                if remaining_train[i] == 0 and i not in completed_this_cycle:
                    _ = train_one_epoch(
                        self.client_models[i],
                        self.client_datasets[i],
                        device=self.cfg.device,
                        batch_size=32,
                        lr=self.cfg.lr,
                        class_weight=self.client_datasets[i].class_weight,
                    )

                    completed_this_cycle.add(i)

                    mean_minutes_i = self._current_mean_minutes(i)

                    sampled_rounds = sample_train_rounds_with_exp_jitter(
                        mean_minutes=mean_minutes_i,
                        seconds_per_round=self.cfg.seconds_per_round,
                        jitter_sec=0.5,
                        clamp=True,
                        rng=self.rng,
                    )

                    print(
                        f"[OFFLOAD][TIME MODEL] round={round_idx}, sat={i}, "
                        f"Ni={len(self.client_datasets[i])}, "
                        f"cap={self.train_capacities[i]:.4f}, "
                        f"mean_min={mean_minutes_i:.2f}, sampled_rounds={sampled_rounds}"
                    )

                    remaining_train[i] = sampled_rounds

                    self.train_time_log.append(
                        (
                            round_idx,
                            i,
                            float(sampled_rounds * self.cfg.seconds_per_round),
                        )
                    )

                    print(
                        f"[OFFLOAD][TRAIN DONE] round={round_idx}, sat={i}, completed={len(completed_this_cycle)}/{n_clients}"
                    )

            print(
                f"[OFFLOAD][CHECK] round={round_idx}, completed={len(completed_this_cycle)}/{n_clients}"
            )

            # (2) 모든 위성이 이번 cycle에서 local update를 완료했을 때만 aggregation
            if len(completed_this_cycle) == n_clients:
                print(f"[OFFLOAD][AGGREGATE] round={round_idx}")

                # -------------------------------------------------
                # (2-1) intra-orbit aggregation
                # 각 orbital plane 내부에서는 leader가 plane-level aggregator 역할을 수행
                # -------------------------------------------------
                for p, sat_indices in plane_to_sats.items():
                    self.aggregate_fn(
                        plane_global_models[p],
                        self.client_models,
                        sat_indices,
                    )

                # -------------------------------------------------
                # (2-2) inter-orbit decentralized leader mixing
                # 기존 코드처럼 하나의 global_model로 합치지 않음.
                # 각 leader는 연결된 neighbor leader들과만 model mixing.
                # -------------------------------------------------
                old_plane_models = {
                    p: copy.deepcopy(plane_global_models[p]).to(self.cfg.device)
                    for p in plane_to_sats.keys()
                }

                mixed_plane_states = {}

                for p in plane_to_sats.keys():
                    neigh_planes = sorted(leader_neighbors.get(p, set()))

                    neighbor_models = [old_plane_models[q] for q in neigh_planes]

                    mixed_plane_states[p] = _mix_self_with_neighbor_average(
                        self_model=old_plane_models[p],
                        neighbor_models=neighbor_models,
                        beta=self.leader_mixing_beta,
                        neighbor_weights=None,  # pi_pq는 uniform weight로 구현
                    )

                # 각 plane leader는 자기 mixed model을 유지
                for p in plane_to_sats.keys():
                    plane_global_models[p].load_state_dict(mixed_plane_states[p])

                # -------------------------------------------------
                # (2-3) plane 내부 broadcast
                # 각 plane의 satellite들은 해당 plane leader의 mixed model만 받음.
                # 전체 위성에 동일 global model을 broadcast하지 않음.
                # -------------------------------------------------
                for p, sat_indices in plane_to_sats.items():
                    for i in sat_indices:
                        self.client_models[i].load_state_dict(
                            copy.deepcopy(plane_global_models[p].state_dict())
                        )

                # -------------------------------------------------
                # (2-4) representative global_model
                # training에는 사용하지 않고, return/save convenience 용도로만 생성
                # -------------------------------------------------
                representative_models = [
                    plane_global_models[p] for p in sorted(plane_to_sats.keys())
                ]

                representative_weights = [
                    sum(len(self.client_datasets[i]) for i in plane_to_sats[p])
                    for p in sorted(plane_to_sats.keys())
                ]

                self.global_model.load_state_dict(
                    _weighted_average_state_dict_from_models(
                        representative_models,
                        weights=representative_weights,
                    )
                )

                # -------------------------------------------------
                # communication log
                # intra-plane: non-leader satellite upload + downlink
                # inter-plane: leader-neighbor directed transmissions
                # -------------------------------------------------
                plane_hops = 0
                for p, sat_indices in plane_to_sats.items():
                    # leader 자신의 local model은 통신으로 세지 않음
                    plane_hops += 2 * max(len(sat_indices) - 1, 0)

                leader_hops = sum(
                    len(leader_neighbors.get(p, set())) for p in leader_neighbors.keys()
                )

                model_hops = plane_hops + leader_hops

                self.comm_logger.log(
                    round_idx,
                    model_bytes=model_hops * self.model_bytes,
                    model_hops=model_hops,
                    offload_bytes=0,
                    offload_hops=0,
                    note="lhfl_intra_leader_agg+inter_leader_mixing",
                )

                # -------------------------------------------------
                # accuracy log
                # LH-FL에는 완전한 단일 global model이 없으므로
                # plane-level mixed models의 weighted average accuracy로 평가
                # -------------------------------------------------
                acc = _evaluate_weighted_model_accuracy(
                    models=representative_models,
                    weights=representative_weights,
                    test_dataset=self.test_dataset,
                    device=self.cfg.device,
                )

                self.acc_log.append((round_idx, acc))
                self.best_acc = max(self.best_acc, acc)

                print(
                    f"[ACC][offload] round={round_idx}, acc={acc:.2f}%, best={self.best_acc:.2f}%"
                )

                if acc >= self.cfg.target_acc:
                    print(
                        f"[INFO] Target accuracy {self.cfg.target_acc}% reached "
                        f"at round {round_idx}. (continue)"
                    )

                # 다음 cycle 준비
                completed_this_cycle.clear()

        print("[END][offload] finished all rounds.")
        self.finalize_logs(prefix=self.log_prefix)
        return self.global_model, self.best_acc


def _compute_train_capacities_and_time_scale(
    client_datasets,
    per_sat_minutes: Optional[List[int]] = None,
    train_minutes: int = 10,
):
    """
    OffloadScenario/ContactAwareOffloadScenario(LH-FL)와 동일한 capacity-aware
    n_i/C_i 학습시간 모델의 capacity 및 스케일 상수를 계산한다.
    (수식은 OffloadScenario.__init__과 동일 — 사다리 실험에서 offload가 없는
    rung에도 동일한 시간 모델을 적용하기 위해 별도 함수로 추출했을 뿐, 로직은
    OffloadScenario와 100% 동일하다.)
    """
    n_clients = len(client_datasets)
    if per_sat_minutes is not None:
        assert len(per_sat_minutes) == n_clients, (
            "per_sat_minutes 길이가 클라이언트 수와 동일해야 합니다."
        )
        mins = np.asarray(per_sat_minutes, dtype=float)
    else:
        mins = np.full(n_clients, float(train_minutes))

    # capacity_i ∝ 1 / minutes_i (빠를수록 큼), 평균 1로 정규화
    cap = 1.0 / np.clip(mins, 1e-6, None)
    cap = cap / cap.mean()

    # baseline time model: T_i = C * (N_i / capacity_i), 초기 상태에서
    # per_sat_minutes와 대략 맞도록 상수 C(time_scale_const)를 맞춘다.
    Ni0 = np.array([len(ds) for ds in client_datasets], dtype=float)
    ratio0 = Ni0 / np.clip(cap, 1e-6, None)
    time_scale_const = float(np.mean(mins / np.clip(ratio0, 1e-6, None)))

    return cap, time_scale_const


class DecentralizedScenario(FLScenario):
    """
    True decentralized FL baseline.

    Round definition:
      all satellites complete local training
      -> peer-to-peer model mixing with neighbor satellites
      -> each satellite keeps its own local model

    No central server.
    No global aggregation.
    No global broadcast.

    Ablation-ladder extras (off by default; do_offload=False,
    use_capacity_time_model=False reproduces the original class exactly):
        do_offload             : capacity-aware one-shot offload before training,
                                  identical mechanics to OffloadScenario/LH-FL
                                  (offload_balance_datasets + ring-hop transfer plan).
                                  Implies use_capacity_time_model=True.
        use_capacity_time_model: rebases mean_minutes_i via the n_i/C_i model
                                  (OffloadScenario._current_mean_minutes) instead
                                  of the fixed per_sat_minutes[i]. Lets the ladder
                                  isolate the effect of offload from the effect of
                                  the leader hierarchy while keeping the timing
                                  model fair/uniform across rungs.
    """

    def __init__(
        self,
        client_datasets,
        test_dataset,
        config: SimConfig,
        log_prefix="sat_events_decentral",
        per_sat_minutes: Optional[List[int]] = None,
        train_minutes: int = 10,
        max_dist_km: float = 2000.0,
        peer_top_k: int = 2,
        do_offload: bool = False,
        offload_ratio: float = 0.1,
        top_pairs_per_sat: int = 1,
        use_capacity_time_model: bool = False,
        link_fail_rng: Optional[np.random.Generator] = None,
        link_fail_p: float = LINK_FAIL_P_DEFAULT,
    ):
        super().__init__(
            client_datasets,
            test_dataset,
            config,
            link_fail_rng=link_fail_rng,
            link_fail_p=link_fail_p,
        )

        self.log_prefix = log_prefix
        self.participating = list(range(self.n_clients))
        self.per_sat_minutes = per_sat_minutes
        self.train_minutes = train_minutes
        self.max_dist_km = max_dist_km
        self.peer_top_k = peer_top_k

        self.do_offload = do_offload
        self.offload_ratio = offload_ratio
        self.top_pairs_per_sat = top_pairs_per_sat
        self.use_capacity_time_model = bool(use_capacity_time_model or do_offload)

        self.train_capacities, self.time_scale_const = (
            _compute_train_capacities_and_time_scale(
                client_datasets=client_datasets,
                per_sat_minutes=per_sat_minutes,
                train_minutes=train_minutes,
            )
        )
        if self.do_offload or self.use_capacity_time_model:
            print(
                "[SIM][decentral] per_sat_minutes(min) =",
                (
                    list(per_sat_minutes)
                    if per_sat_minutes is not None
                    else [train_minutes] * self.n_clients
                ),
            )
            print(
                "[SIM][decentral] train_capacities(norm) =",
                self.train_capacities.tolist(),
            )

        if per_sat_minutes is not None:
            assert len(per_sat_minutes) == len(client_datasets), (
                "per_sat_minutes 길이가 클라이언트 수와 동일해야 합니다."
            )
            self.train_rounds_needed = [
                minutes_to_rounds(m, self.cfg.seconds_per_round)
                for m in per_sat_minutes
            ]
        else:
            fixed = minutes_to_rounds(train_minutes, self.cfg.seconds_per_round)
            self.train_rounds_needed = [fixed] * self.n_clients

        self.remaining = self.train_rounds_needed.copy()

        # -------------------------------------------------
        # Decentralized FL용 peer graph 구성
        # -------------------------------------------------
        if self.cfg.constellation_configs is not None:
            configs = self.cfg.constellation_configs[: self.n_clients]
            sats = build_fixed_constellation(configs)

            compute_orbits_for_rounds(
                sats,
                num_rounds=self.cfg.num_rounds,
                seconds_per_round=self.cfg.seconds_per_round,
            )

            plane_ids = [c.get("plane_id", 0) for c in configs]

            meeting_counts, _leaders_per_plane, plane_to_sats = (
                precompute_meetings_and_select_leaders(
                    sats,
                    plane_ids=plane_ids,
                    num_rounds=self.cfg.num_rounds,
                    seconds_per_round=self.cfg.seconds_per_round,
                    max_dist_km=self.max_dist_km,
                    log_dir=self.cfg.log_dir,
                    filename=f"{self.log_prefix}_meeting_counts.csv",
                )
            )
            self.meeting_counts = meeting_counts
            self.plane_ids = plane_ids
            self.plane_to_sats = plane_to_sats

            self.peer_neighbors = _build_topk_peer_neighbors_from_meetings(
                meeting_counts=meeting_counts,
                n_nodes=self.n_clients,
                top_k=self.peer_top_k,
            )
        else:
            print(
                "[WARN][DECENTRAL] constellation_configs is None. Use ring peer graph."
            )
            self.peer_neighbors = _build_ring_neighbors(self.n_clients)
            self.meeting_counts = None
            self.plane_ids = None
            self.plane_to_sats = None

        print("[DECENTRAL][PEER GRAPH]")
        for i in range(min(10, self.n_clients)):
            neigh = sorted(self.peer_neighbors[i])
            print(f"  sat_{i + 1}: {[j + 1 for j in neigh]}")

    def _current_mean_minutes(self, sat_idx: int) -> float:
        """
        use_capacity_time_model=False(기본값)면 기존과 동일하게 고정
        per_sat_minutes[i](또는 train_minutes)를 반환한다 (회귀 없음).
        True면 OffloadScenario와 동일한 n_i/C_i rebase를 적용한다.
        """
        if not self.use_capacity_time_model:
            return (
                self.per_sat_minutes[sat_idx]
                if self.per_sat_minutes is not None
                else self.train_minutes
            )

        Ni = float(len(self.client_datasets[sat_idx]))
        Ci = float(self.train_capacities[sat_idx])
        mean_min = self.time_scale_const * (Ni / max(Ci, 1e-6))
        return max(mean_min, 0.5)

    def run(self):
        print("[INFO] True Decentralized FL 시작")

        if self.do_offload or self.use_capacity_time_model:
            print(
                f"[DECENTRAL][MODE] do_offload={self.do_offload}, "
                f"use_capacity_time_model={self.use_capacity_time_model}"
            )

        # -------------------------------------------------
        # (0) capacity-aware one-shot offload (optional, off by default).
        # Identical mechanics to OffloadScenario/ContactAwareOffloadScenario
        # (LH-FL): redistribute samples within each plane so that N_i / C_i
        # is balanced, then rebase the n_i/C_i time model on the post-offload
        # dataset sizes.
        # -------------------------------------------------
        if self.do_offload:
            assert self.plane_to_sats is not None, (
                "do_offload=True requires constellation_configs (plane structure)."
            )

            before_path = os.path.join(
                self.cfg.log_dir or ".", f"{self.log_prefix}_classdist_before.csv"
            )
            save_class_distribution_csv(self.client_datasets, before_path)

            offload_stats = offload_balance_datasets(
                client_datasets=self.client_datasets,
                meeting_counts=self.meeting_counts,
                plane_ids=self.plane_ids,
                offload_ratio=self.offload_ratio,
                top_pairs_per_sat=self.top_pairs_per_sat,
                seed=0,
                train_capacities=self.train_capacities,
            )

            offload_hops_total = offload_bytes_total = offload_samples_total = 0
            offload_rows = []
            for p, sat_indices in self.plane_to_sats.items():
                m = len(sat_indices)
                if m <= 1:
                    continue
                Ni_before = np.array(
                    [offload_stats["Ni_before"][i] for i in sat_indices], dtype=int
                )
                Ni_target = np.array(
                    [offload_stats["Ni_target"][i] for i in sat_indices], dtype=int
                )
                for s_local, r_local, delta in build_transfer_plan_within_plane(
                    Ni_before, Ni_target
                ):
                    hop = ring_hop_distance(s_local, r_local, m)
                    hop = jitter_hop_count(
                        self.link_rng, hop
                    )  # 궤도 섭동 근사 ±10% (link_rng=None이면 무변화)
                    moved = int(abs(delta))
                    if moved <= 0:
                        continue
                    src, dst = int(sat_indices[s_local]), int(sat_indices[r_local])
                    offload_hops_total += int(hop)
                    ob = int(moved * self.sample_bytes * hop)
                    offload_bytes_total += ob
                    offload_samples_total += moved
                    offload_rows.append(
                        {
                            "plane_id": int(p),
                            "src_sat_id": src,
                            "dst_sat_id": dst,
                            "src_name": f"sat_{src + 1}",
                            "dst_name": f"sat_{dst + 1}",
                            "moved_samples": moved,
                            "hop": int(hop),
                            "sample_bytes": int(self.sample_bytes),
                            "offload_bytes": ob,
                        }
                    )

            self.comm_logger.log(
                0,
                model_bytes=0,
                offload_bytes=int(offload_bytes_total),
                model_hops=0,
                offload_hops=int(offload_hops_total),
                note=(
                    f"one_shot_offload,samples={offload_samples_total},"
                    f"sample_bytes={self.sample_bytes},hops={offload_hops_total}"
                ),
            )
            pd.DataFrame(offload_rows).to_csv(
                os.path.join(
                    self.cfg.log_dir or ".", f"{self.log_prefix}_offload_pairs.csv"
                ),
                index=False,
                encoding="utf-8-sig",
            )

            after_path = os.path.join(
                self.cfg.log_dir or ".", f"{self.log_prefix}_classdist_after.csv"
            )
            save_class_distribution_csv(self.client_datasets, after_path)

        if self.do_offload or self.use_capacity_time_model:
            cap_path = os.path.join(
                self.cfg.log_dir or ".", f"{self.log_prefix}_capacity.csv"
            )
            cap_rows = []
            for i, cap in enumerate(self.train_capacities):
                row = {
                    "sat_id": i,
                    "sat_name": f"sat_{i + 1}",
                    "plane_id": (
                        self.plane_ids[i] if self.plane_ids is not None else 0
                    ),
                    "train_capacity": float(cap),
                }
                if self.do_offload:
                    row["Ni_before"] = float(offload_stats["Ni_before"][i])
                    row["Ni_target"] = float(offload_stats["Ni_target"][i])
                    row["Ni_after"] = float(offload_stats["Ni_after"][i])
                cap_rows.append(row)
            pd.DataFrame(cap_rows).to_csv(cap_path, index=False, encoding="utf-8-sig")
            print(f"[SAVE][CAPACITY] {cap_path}")

            # offload로 Ni가 바뀐 뒤의 상태로 n_i/C_i 학습시간 모델을 다시 rebase.
            self.train_rounds_needed = [
                minutes_to_rounds(
                    self._current_mean_minutes(i), self.cfg.seconds_per_round
                )
                for i in range(self.n_clients)
            ]
            self.remaining = self.train_rounds_needed.copy()

        completed_this_cycle: Set[int] = set()

        for round_idx in range(1, self.cfg.num_rounds + 1):
            # -------------------------------------------------
            # (1) local training countdown
            # -------------------------------------------------
            for i in self.participating:
                if self.remaining[i] > 0:
                    self.remaining[i] -= 1

                if self.remaining[i] == 0 and i not in completed_this_cycle:
                    train_one_epoch(
                        self.client_models[i],
                        self.client_datasets[i],
                        device=self.cfg.device,
                        batch_size=32,
                        lr=self.cfg.lr,
                        class_weight=self.client_datasets[i].class_weight,
                    )

                    completed_this_cycle.add(i)

                    mean_minutes_i = self._current_mean_minutes(i)

                    sampled_rounds = sample_train_rounds_with_exp_jitter(
                        mean_minutes=mean_minutes_i,
                        seconds_per_round=self.cfg.seconds_per_round,
                        jitter_sec=0.5,
                        clamp=True,
                        rng=self.rng,
                    )

                    self.remaining[i] = sampled_rounds

                    self.train_time_log.append(
                        (
                            round_idx,
                            i,
                            float(sampled_rounds * self.cfg.seconds_per_round),
                        )
                    )

            # -------------------------------------------------
            # (2) 모든 위성이 이번 cycle의 local training을 완료하면 peer mixing
            # -------------------------------------------------
            if len(completed_this_cycle) == self.n_clients:
                print(f"[DECENTRAL][MIX] sim_round={round_idx}")

                # simultaneous mixing을 위해 old model snapshot 생성
                old_models = [
                    copy.deepcopy(self.client_models[i]).to(self.cfg.device)
                    for i in self.participating
                ]

                new_states = {}
                directed_model_hops = 0

                for i in self.participating:
                    neigh = sorted(self.peer_neighbors.get(i, set()))

                    # 링크 실패: neigh 각각을 "이번 라운드에 실제로 수신 성공했는가"로
                    # 독립적으로 필터링한다. link_fail_rng가 None이면(옵트아웃) neigh 그대로.
                    neigh = filter_by_link_failure(
                        self.link_rng, self.link_fail_p, neigh
                    )

                    # closed neighborhood: 자기 model(항상 포함, 고립 방지) + 살아남은 neighbor models
                    mixing_models = [old_models[i]] + [old_models[j] for j in neigh]

                    new_states[i] = _weighted_average_state_dict_from_models(
                        mixing_models,
                        weights=None,
                    )

                    directed_model_hops += len(neigh)

                # 각 위성은 자기 neighbor와 mixing한 local model을 유지
                for i in self.participating:
                    self.client_models[i].load_state_dict(new_states[i])

                # -------------------------------------------------
                # communication log
                # undirected edge의 양방향 model exchange를 directed transmissions로 계산.
                # link failure가 켜져 있으면 "성공한" 전송만 카운트되므로 seed마다 변동이 생긴다.
                # -------------------------------------------------

                self.comm_logger.log(
                    round_idx,
                    model_bytes=directed_model_hops * self.model_bytes,
                    model_hops=directed_model_hops,
                    note="decentral_peer_mixing",
                )

                # -------------------------------------------------
                # accuracy log
                # decentralized FL에는 single global model이 없으므로
                # satellite-local models의 average accuracy를 기록
                # -------------------------------------------------
                acc = _evaluate_weighted_model_accuracy(
                    models=[self.client_models[i] for i in self.participating],
                    weights=[1.0] * self.n_clients,
                    test_dataset=self.test_dataset,
                    device=self.cfg.device,
                )

                self.acc_log.append((round_idx, acc))
                self.best_acc = max(self.best_acc, acc)

                print(
                    f"[ACC][decentral] sim_round={round_idx}, "
                    f"avg_local_acc={acc:.2f}%, best={self.best_acc:.2f}%"
                )

                if acc >= self.cfg.target_acc:
                    print(
                        f"[INFO] Target accuracy {self.cfg.target_acc}% reached "
                        f"at simulation round {round_idx}. (continue)"
                    )

                completed_this_cycle.clear()

            if round_idx % 50 == 0:
                print(
                    f"[DECENTRAL] sim_round={round_idx}, acc={self.acc_log[-1][1]:.2f}%"
                )

        self.finalize_logs(prefix=self.log_prefix)
        return self.client_models, self.best_acc


class FedLEOScenario(FLScenario):
    """
    FedLEO-style offloading-assisted decentralized FL.

    핵심 수정:
      - 모든 peer와 full decentralized mixing을 하지 않음
      - greedy offloading decision으로 선택된 collaboration pair만 사용
      - 각 pair 내부에서만 bidirectional model exchange + pairwise averaging 수행
      - 통신량은 선택된 pair 수에 비례하도록 계산
    """

    def __init__(
        self,
        client_datasets,
        test_dataset,
        config: SimConfig,
        log_prefix="sat_events_fedleo",
        per_sat_minutes: Optional[List[int]] = None,
        train_minutes: int = 10,
        max_dist_km: float = 2000.0,
        peer_top_k: int = 2,
        offload_ratio: float = 0.1,
        overload_threshold: float = 1.10,
        max_offload_iter: int = 200,
    ):
        super().__init__(client_datasets, test_dataset, config)

        self.log_prefix = log_prefix
        self.participating = list(range(self.n_clients))

        self.per_sat_minutes = per_sat_minutes
        self.train_minutes = train_minutes
        self.max_dist_km = max_dist_km
        self.peer_top_k = peer_top_k
        self.offload_ratio = offload_ratio
        self.overload_threshold = overload_threshold
        self.max_offload_iter = max_offload_iter

        # -------------------------------------------------
        # 위성별 계산 능력 설정
        # capacity_i ∝ 1 / train_minutes_i
        # -------------------------------------------------
        if per_sat_minutes is not None:
            assert len(per_sat_minutes) == len(client_datasets)
            mins = np.asarray(per_sat_minutes, dtype=float)
        else:
            mins = np.full(self.n_clients, float(train_minutes))

        self.base_train_minutes = mins

        cap = 1.0 / np.clip(mins, 1e-6, None)
        cap = cap / cap.mean()
        self.train_capacities = cap

        Ni0 = np.array([len(ds) for ds in client_datasets], dtype=float)
        ratio0 = Ni0 / np.clip(self.train_capacities, 1e-6, None)
        self.time_scale_const = float(np.mean(mins / np.clip(ratio0, 1e-6, None)))

        print("[SIM][FedLEO] per_sat_minutes(min) =", mins.tolist())
        print("[SIM][FedLEO] train_capacities(norm) =", self.train_capacities.tolist())
        print("[SIM][FedLEO] initial Ni =", Ni0.tolist())
        print("[SIM][FedLEO] time_scale_const =", self.time_scale_const)

        # -------------------------------------------------
        # constellation / peer graph 구성
        # -------------------------------------------------
        configs = self.cfg.constellation_configs[: self.n_clients]
        self.sats = build_fixed_constellation(configs)

        compute_orbits_for_rounds(
            self.sats,
            num_rounds=self.cfg.num_rounds,
            seconds_per_round=self.cfg.seconds_per_round,
        )

        self.plane_ids, self.plane_to_sats = _plane_maps_from_configs(
            configs,
            self.n_clients,
        )

        meeting_counts, _, _ = precompute_meetings_and_select_leaders(
            self.sats,
            plane_ids=self.plane_ids,
            num_rounds=self.cfg.num_rounds,
            seconds_per_round=self.cfg.seconds_per_round,
            max_dist_km=self.max_dist_km,
            log_dir=self.cfg.log_dir,
            filename=f"{self.log_prefix}_meeting_counts.csv",
        )

        self.peer_neighbors = _build_topk_peer_neighbors_from_meetings(
            meeting_counts=meeting_counts,
            n_nodes=self.n_clients,
            top_k=self.peer_top_k,
        )

        # offloading으로 실제 선택된 collaboration pair
        self.fedleo_active_pairs = set()

        print("[FedLEO][PEER GRAPH]")
        for i in range(min(10, self.n_clients)):
            print(
                f"  sat_{i + 1} -> "
                f"{[j + 1 for j in sorted(self.peer_neighbors.get(i, set()))]}"
            )

    def _current_mean_minutes(self, sat_idx: int) -> float:
        Ni = float(len(self.client_datasets[sat_idx]))
        Si = float(self.train_capacities[sat_idx])
        mean_min = self.time_scale_const * (Ni / max(Si, 1e-6))
        return max(mean_min, 0.5)

    def _rebuild_dataset(self, ref_ds, X, Y):
        """
        offloading 후 X, Y로 dataset 재생성.
        InMemoryClientDataset 생성자 호환 문제를 피하기 위해
        먼저 X, Y만 넣고, 주요 속성은 생성 후 attribute로 복사한다.
        """
        X_new = X.clone()
        Y_new = Y.clone()

        try:
            new_ds = ref_ds.__class__(X_new, Y_new)
        except TypeError:
            new_ds = copy.deepcopy(ref_ds)
            new_ds.X = X_new
            new_ds.Y = Y_new

        for attr in ["n_classes", "n_features", "class_weight"]:
            if hasattr(ref_ds, attr):
                value = getattr(ref_ds, attr)
                if torch.is_tensor(value):
                    value = value.clone()
                setattr(new_ds, attr, value)

        return new_ds

    def _move_samples(self, src_idx: int, dst_idx: int, n_move: int) -> int:
        """
        src_idx에서 dst_idx로 n_move개 샘플 이동.
        반환: 실제 이동한 샘플 수
        """
        src = self.client_datasets[src_idx]
        dst = self.client_datasets[dst_idx]

        n_src = len(src)
        if n_src <= 1:
            return 0

        n_move = int(max(0, min(n_move, n_src - 1)))
        if n_move <= 0:
            return 0

        move_idx = self.rng.choice(n_src, size=n_move, replace=False)

        move_mask_np = np.zeros(n_src, dtype=bool)
        move_mask_np[move_idx] = True

        move_mask = torch.as_tensor(
            move_mask_np,
            dtype=torch.bool,
            device=src.X.device,
        )
        keep_mask = ~move_mask

        X_move = src.X[move_mask]
        Y_move = src.Y[move_mask]

        X_src_new = src.X[keep_mask]
        Y_src_new = src.Y[keep_mask]

        X_dst_new = torch.cat([dst.X, X_move], dim=0)
        Y_dst_new = torch.cat([dst.Y, Y_move], dim=0)

        self.client_datasets[src_idx] = self._rebuild_dataset(
            src,
            X_src_new,
            Y_src_new,
        )

        self.client_datasets[dst_idx] = self._rebuild_dataset(
            dst,
            X_dst_new,
            Y_dst_new,
        )

        return n_move

    def _evaluate_offload_candidate(self, sender, receiver, Ni, cap):
        """
        sender -> receiver로 일부 샘플을 옮겼을 때
        max normalized load가 얼마나 줄어드는지 계산.
        score = load_reduction / communication_cost
        """
        load = Ni / np.clip(cap, 1e-6, None)
        current_max_load = float(load.max())

        Cs = float(cap[sender])
        Cr = float(cap[receiver])
        Ns = float(Ni[sender])
        Nr = float(Ni[receiver])

        # 두 위성의 normalized load를 같게 만드는 이상적 이동량
        equalize_x = (Ns * Cr - Nr * Cs) / max(Cs + Cr, 1e-6)

        max_x = max(1, int(self.offload_ratio * max(Ns, 1)))
        n_move = int(max(0, min(equalize_x, max_x, Ns - 1)))

        if n_move <= 0:
            return None

        Ni_new = Ni.copy()
        Ni_new[sender] -= n_move
        Ni_new[receiver] += n_move

        new_load = Ni_new / np.clip(cap, 1e-6, None)
        new_max_load = float(new_load.max())

        load_reduction = current_max_load - new_max_load
        if load_reduction <= 0:
            return None

        # peer graph 직접 연결 기준 1-hop
        hop = 1
        offload_bytes = int(n_move * self.sample_bytes * hop)

        # 통신비용 대비 load 개선량
        score = load_reduction / max(offload_bytes, 1)

        return {
            "sender": int(sender),
            "receiver": int(receiver),
            "n_move": int(n_move),
            "hop": int(hop),
            "offload_bytes": int(offload_bytes),
            "load_reduction": float(load_reduction),
            "score": float(score),
            "current_max_load": float(current_max_load),
            "new_max_load": float(new_max_load),
        }

    def _perform_fedleo_offloading(self):
        """
        FedLEO-style system-wide greedy offloading.

        기존 단순 방식:
          가장 과부하 위성 1개 -> 가장 여유 있는 peer

        수정 방식:
          모든 sender-receiver 후보를 평가한 뒤,
          max normalized load를 가장 효율적으로 줄이는 pair를 반복 선택.
        """
        before_path = os.path.join(
            self.cfg.log_dir or ".",
            f"{self.log_prefix}_classdist_before.csv",
        )
        save_class_distribution_csv(self.client_datasets, before_path)

        offload_hops_total = 0
        offload_bytes_total = 0
        offload_samples_total = 0
        transfer_rows = []
        active_pairs = set()

        for it in range(self.max_offload_iter):
            Ni = np.array(
                [len(ds) for ds in self.client_datasets],
                dtype=float,
            )
            cap = np.clip(self.train_capacities, 1e-6, None)
            load = Ni / cap

            mean_load = float(np.mean(load))
            threshold = mean_load * float(self.overload_threshold)

            # threshold보다 큰 sender만 후보
            overloaded = [i for i in range(self.n_clients) if load[i] > threshold]

            if len(overloaded) == 0:
                break

            best = None

            for sender in overloaded:
                candidates = sorted(self.peer_neighbors.get(sender, set()))

                # sender보다 load가 낮은 peer만 receiver 후보
                candidates = [r for r in candidates if load[r] < load[sender]]

                for receiver in candidates:
                    cand = self._evaluate_offload_candidate(
                        sender=sender,
                        receiver=receiver,
                        Ni=Ni,
                        cap=cap,
                    )

                    if cand is None:
                        continue

                    if best is None or cand["score"] > best["score"]:
                        best = cand

            if best is None:
                break

            sender = best["sender"]
            receiver = best["receiver"]
            n_move = best["n_move"]

            moved = self._move_samples(sender, receiver, n_move)
            if moved <= 0:
                break

            hop = int(best["hop"])
            offload_bytes = int(moved * self.sample_bytes * hop)

            offload_hops_total += hop
            offload_samples_total += moved
            offload_bytes_total += offload_bytes

            pair = tuple(sorted((sender, receiver)))
            active_pairs.add(pair)

            transfer_rows.append(
                {
                    "iter": it,
                    "sender": sender,
                    "receiver": receiver,
                    "moved_samples": int(moved),
                    "hop": hop,
                    "sample_bytes": int(self.sample_bytes),
                    "offload_bytes": offload_bytes,
                    "load_reduction": float(best["load_reduction"]),
                    "score": float(best["score"]),
                    "current_max_load": float(best["current_max_load"]),
                    "new_max_load": float(best["new_max_load"]),
                }
            )

        self.fedleo_active_pairs = active_pairs

        self.comm_logger.log(
            0,
            model_bytes=0,
            offload_bytes=int(offload_bytes_total),
            model_hops=0,
            offload_hops=int(offload_hops_total),
            note=(
                f"fedleo_greedy_offload,"
                f"samples={offload_samples_total},"
                f"sample_bytes={self.sample_bytes},"
                f"hops={offload_hops_total},"
                f"active_pairs={len(active_pairs)}"
            ),
        )

        after_path = os.path.join(
            self.cfg.log_dir or ".",
            f"{self.log_prefix}_classdist_after.csv",
        )
        save_class_distribution_csv(self.client_datasets, after_path)

        transfer_path = os.path.join(
            self.cfg.log_dir or ".",
            f"{self.log_prefix}_offload_transfers.csv",
        )
        pd.DataFrame(transfer_rows).to_csv(
            transfer_path,
            index=False,
            encoding="utf-8-sig",
        )

        pairs_path = os.path.join(
            self.cfg.log_dir or ".",
            f"{self.log_prefix}_active_pairs.csv",
        )
        pd.DataFrame(
            [{"sat_a": a, "sat_b": b} for a, b in sorted(active_pairs)]
        ).to_csv(pairs_path, index=False, encoding="utf-8-sig")

        print(f"[SAVE][FedLEO-OFFLOAD] {transfer_path}")
        print(f"[SAVE][FedLEO-PAIRS] {pairs_path}")
        print(
            f"[FedLEO][OFFLOAD] samples={offload_samples_total}, "
            f"hops={offload_hops_total}, bytes={offload_bytes_total}, "
            f"active_pairs={len(active_pairs)}"
        )

    def _sample_training_rounds_for_all(self, start_round: int):
        remaining = []

        for i in range(self.n_clients):
            sampled = sample_train_rounds_with_exp_jitter(
                mean_minutes=self._current_mean_minutes(i),
                seconds_per_round=self.cfg.seconds_per_round,
                jitter_sec=0.5,
                clamp=True,
                rng=self.rng,
            )

            remaining.append(sampled)

            self.train_time_log.append(
                (start_round, i, float(sampled * self.cfg.seconds_per_round))
            )

        return remaining

    def _pairwise_model_mixing(self, active_pairs):
        """
        선택된 collaboration pair 내부에서만 bidirectional model exchange.
        각 pair는 두 모델을 weighted average하여 두 위성에 동일하게 반영한다.
        """
        old_models = [
            copy.deepcopy(self.client_models[i]).to(self.cfg.device)
            for i in self.participating
        ]

        new_states = {
            i: copy.deepcopy(old_models[i].state_dict()) for i in self.participating
        }

        for a, b in sorted(active_pairs):
            weights = [
                len(self.client_datasets[a]),
                len(self.client_datasets[b]),
            ]

            avg_state = _weighted_average_state_dict_from_models(
                [old_models[a], old_models[b]],
                weights=weights,
            )

            new_states[a] = copy.deepcopy(avg_state)
            new_states[b] = copy.deepcopy(avg_state)

        for i in self.participating:
            self.client_models[i].load_state_dict(new_states[i])

    def run(self):
        print("[INFO][FedLEO] start")

        # 1) 학습 시작 전 FedLEO-style greedy offloading
        self._perform_fedleo_offloading()

        active_pairs = set(self.fedleo_active_pairs)

        # fallback: offloading pair가 하나도 없으면 top-k graph에서 최소 연결만 사용
        if len(active_pairs) == 0:
            print(
                "[WARN][FedLEO] no active offloading pairs. Use minimal peer edges fallback."
            )
            for i in range(self.n_clients):
                neigh = sorted(self.peer_neighbors.get(i, set()))
                if len(neigh) > 0:
                    active_pairs.add(tuple(sorted((i, neigh[0]))))

        print(f"[FedLEO][ACTIVE PAIRS] {len(active_pairs)} pairs")

        remaining = self._sample_training_rounds_for_all(start_round=0)
        completed_this_cycle: Set[int] = set()

        for round_idx in range(1, self.cfg.num_rounds + 1):
            # -------------------------------------------------
            # 1) local training countdown
            # -------------------------------------------------
            for i in self.participating:
                if self.client_datasets[i] is None or len(self.client_datasets[i]) == 0:
                    continue

                if remaining[i] > 0:
                    remaining[i] -= 1

                if remaining[i] == 0 and i not in completed_this_cycle:
                    train_one_epoch(
                        self.client_models[i],
                        self.client_datasets[i],
                        device=self.cfg.device,
                        batch_size=32,
                        lr=self.cfg.lr,
                        class_weight=self.client_datasets[i].class_weight,
                    )

                    completed_this_cycle.add(i)

            # -------------------------------------------------
            # 2) 모든 위성이 local training 완료 -> selected pair mixing
            # -------------------------------------------------
            if len(completed_this_cycle) == self.n_clients:
                print(f"[FedLEO][PAIR-MIX] sim_round={round_idx}")

                self._pairwise_model_mixing(active_pairs)

                # 선택된 pair만 bidirectional model exchange
                # a <-> b 이므로 pair당 2 model transmissions
                model_hops = 2 * len(active_pairs)

                self.comm_logger.log(
                    round_idx,
                    model_bytes=model_hops * self.model_bytes,
                    model_hops=model_hops,
                    offload_bytes=0,
                    offload_hops=0,
                    note=f"fedleo_selected_pair_mixing,pairs={len(active_pairs)}",
                )

                acc = _evaluate_weighted_model_accuracy(
                    models=[self.client_models[i] for i in self.participating],
                    weights=[len(self.client_datasets[i]) for i in self.participating],
                    test_dataset=self.test_dataset,
                    device=self.cfg.device,
                )

                self.acc_log.append((round_idx, acc))
                self.best_acc = max(self.best_acc, acc)

                print(
                    f"[ACC][FedLEO] sim_round={round_idx}, "
                    f"acc={acc:.2f}%, best={self.best_acc:.2f}%"
                )

                if acc >= self.cfg.target_acc:
                    print(
                        f"[INFO][FedLEO] Target accuracy {self.cfg.target_acc}% reached "
                        f"at simulation round {round_idx}. (continue)"
                    )

                completed_this_cycle.clear()
                remaining = self._sample_training_rounds_for_all(
                    start_round=round_idx,
                )

        self.finalize_logs(prefix=self.log_prefix)
        return self.client_models, self.best_acc


class FedISLScenario(FLScenario):
    """
    FedISL baseline.

    Paper-faithful structure:
      1) PS sends global model to one source satellite per orbital plane.
      2) Source distributes global model through intra-plane ISL ring.
      3) All satellites perform local training.
      4) Satellites aggregate updates toward a sink node within each plane.
      5) Each sink sends plane-level partial aggregate to PS.
      6) PS aggregates plane-level models into one global model.

    Notes:
      - Current simulator uses discrete simulation rounds.
      - Intra-plane ISL propagation delay is modeled as ring hop count.
      - Source is the first PS-visible satellite in each plane.
      - Sink is selected by predicted future PS contact duration.
    """

    def __init__(
        self,
        client_datasets,
        test_dataset,
        config: SimConfig,
        log_prefix="sat_events_fedisl",
        per_sat_minutes: Optional[List[int]] = None,
        train_minutes: int = 10,
        link_fail_rng: Optional[np.random.Generator] = None,
        link_fail_p: float = LINK_FAIL_P_DEFAULT,
    ):
        super().__init__(
            client_datasets,
            test_dataset,
            config,
            link_fail_rng=link_fail_rng,
            link_fail_p=link_fail_p,
        )

        self.log_prefix = log_prefix
        self.per_sat_minutes = per_sat_minutes
        self.train_minutes = train_minutes

        if per_sat_minutes is not None:
            assert len(per_sat_minutes) == len(client_datasets)
            self.base_train_minutes = np.asarray(per_sat_minutes, dtype=float)
        else:
            self.base_train_minutes = np.full(self.n_clients, float(train_minutes))

        configs = self.cfg.constellation_configs[: self.n_clients]
        self.sats = build_fixed_constellation(configs)
        compute_orbits_for_rounds(
            self.sats,
            num_rounds=self.cfg.num_rounds,
            seconds_per_round=self.cfg.seconds_per_round,
        )

        self.plane_ids, self.plane_to_sats = _plane_maps_from_configs(
            configs,
            self.n_clients,
        )

        # plane 내부 ID 순서를 orbital phase 기준으로 재정렬하고 싶으면 기존 함수 사용 가능
        # 단, client_datasets index와 satellite index가 바뀌면 혼동될 수 있어 여기서는 configs 순서를 유지함.

    def _find_source_for_plane(self, sat_indices, start_round: int):
        """
        start_round 이후 해당 plane에서 PS와 가장 먼저 연결되는 satellite를 source로 선택.

        link_rng가 설정되어 있으면, "PS 가시성(궤도 결정적)"과 "그 순간 다운링크 전송 성공
        여부(랜덤)"를 분리한다: 접촉이 가능한 매 라운드마다 성공 여부를 새로 굴려서, 실패하면
        접촉 윈도우가 끝날 때까지/다음 접촉까지 계속 재시도한다.
        """
        for t in range(start_round, self.cfg.num_rounds):
            connected = set(connected_clients_at_round(self.sats, t))
            candidates = [sid for sid in sat_indices if sid in connected]
            if len(candidates) > 0 and link_success(self.link_rng, self.link_fail_p):
                return candidates[0], t

        return None, None

    def _train_satellite_once(self, sat_idx: int):
        train_one_epoch(
            self.client_models[sat_idx],
            self.client_datasets[sat_idx],
            device=self.cfg.device,
            batch_size=32,
            lr=self.cfg.lr,
            class_weight=self.client_datasets[sat_idx].class_weight,
        )

    def run(self):
        print("[INFO][FedISL] start")

        current_round = 0
        global_version = 0

        while current_round < self.cfg.num_rounds:
            plane_models = []
            plane_weights = []

            plane_finish_rounds = []
            total_model_hops = 0
            ps_hops = 0
            isl_hops = 0

            # -------------------------------------------------
            # 각 plane별 source selection → distribution → local training → sink aggregation
            # -------------------------------------------------
            for p, sat_indices in self.plane_to_sats.items():
                sat_indices = list(sat_indices)
                m = len(sat_indices)

                source, source_round = self._find_source_for_plane(
                    sat_indices,
                    start_round=current_round,
                )

                if source is None:
                    print(
                        f"[WARN][FedISL] no source found for plane={p}; skip remaining."
                    )
                    continue

                # PS -> source: global model downlink
                for sid in sat_indices:
                    self.client_models[sid].load_state_dict(
                        copy.deepcopy(self.global_model.state_dict())
                    )

                ps_hops += 1

                # source가 ring으로 전체 plane에 model을 전파하는 데 필요한 최대 hop
                dist_hops = max(
                    _ring_distance_in_plane(sid, source, sat_indices)
                    for sid in sat_indices
                )
                dist_hops = jitter_hop_count(self.link_rng, dist_hops)  # 궤도 섭동 근사
                distribution_done_round = source_round + dist_hops

                isl_hops += jitter_hop_count(
                    self.link_rng, max(m - 1, 0)
                )  # distribution spanning ring/tree edge count

                # FedISL sink selection: 예상 완료 시점 기준
                max_train_rounds = 0
                for sid in sat_indices:
                    sampled = sample_train_rounds_with_exp_jitter(
                        mean_minutes=float(self.base_train_minutes[sid]),
                        seconds_per_round=self.cfg.seconds_per_round,
                        jitter_sec=0.5,
                        clamp=True,
                        rng=self.rng,
                    )
                    max_train_rounds = max(max_train_rounds, sampled)
                    self.train_time_log.append(
                        (
                            distribution_done_round,
                            sid,
                            float(sampled * self.cfg.seconds_per_round),
                        )
                    )

                # sink까지 aggregation propagation도 ring 최대 hop으로 근사
                # 논문 구조상 sink-root routing tree에 따른 partial aggregation
                pre_sink_finish = distribution_done_round + max_train_rounds + dist_hops

                sink = _choose_fedisl_sink(
                    self.sats,
                    sat_indices=sat_indices,
                    future_round=pre_sink_finish,
                    max_rounds=self.cfg.num_rounds,
                )

                # 모든 satellite local training 수행
                for sid in sat_indices:
                    self._train_satellite_once(sid)

                # plane-level partial aggregation
                tmp_plane_model = copy.deepcopy(self.global_model).to(self.cfg.device)
                self.aggregate_fn(
                    tmp_plane_model,
                    self.client_models,
                    sat_indices,
                )

                # sink가 PS와 연결될 수 있는 시점까지 기다림 (접촉 가능 + 업링크 성공)
                sink_upload_round = None
                for t in range(pre_sink_finish, self.cfg.num_rounds):
                    connected = set(connected_clients_at_round(self.sats, t))
                    if sink in connected and link_success(
                        self.link_rng, self.link_fail_p
                    ):
                        sink_upload_round = t
                        break

                if sink_upload_round is None:
                    print(f"[WARN][FedISL] sink cannot reach PS for plane={p}; skip.")
                    continue

                ps_hops += 1  # sink -> PS
                isl_hops += jitter_hop_count(
                    self.link_rng, max(m - 1, 0)
                )  # aggregation tree edge count

                plane_models.append(tmp_plane_model)
                plane_weights.append(
                    sum(len(self.client_datasets[sid]) for sid in sat_indices)
                )
                plane_finish_rounds.append(sink_upload_round)

                print(
                    f"[FedISL][plane={p}] source=sat_{source + 1}, sink=sat_{sink + 1}, "
                    f"source_round={source_round}, finish={sink_upload_round}"
                )

            if len(plane_models) == 0:
                break

            # -------------------------------------------------
            # PS global aggregation: 모든 plane partial model을 기다림
            # -------------------------------------------------
            aggregate_round = max(plane_finish_rounds)

            self.global_model.load_state_dict(
                _weighted_average_state_dict_from_models(
                    plane_models,
                    weights=plane_weights,
                )
            )

            for i in range(self.n_clients):
                self.client_models[i].load_state_dict(
                    copy.deepcopy(self.global_model.state_dict())
                )

            total_model_hops = ps_hops + isl_hops

            self.comm_logger.log(
                aggregate_round,
                model_bytes=total_model_hops * self.model_bytes,
                model_hops=total_model_hops,
                note=f"fedisl_ps_hops={ps_hops},isl_hops={isl_hops}",
            )

            self.after_aggregate(aggregate_round)

            print(
                f"[ACC][FedISL] round={aggregate_round}, "
                f"acc={self.acc_log[-1][1]:.2f}%, "
                f"ps_hops={ps_hops}, isl_hops={isl_hops}"
            )

            if self.acc_log[-1][1] >= self.cfg.target_acc:
                print(
                    f"[INFO][FedISL] Target accuracy {self.cfg.target_acc}% reached "
                    f"at round {aggregate_round}. (continue)"
                )

            global_version += 1
            current_round = aggregate_round + 1

        self.finalize_logs(prefix=self.log_prefix)
        return self.global_model, self.best_acc
