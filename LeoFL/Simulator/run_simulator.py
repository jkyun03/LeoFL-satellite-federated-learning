# run_simulator.py
# mode 문자열(always/offload/decentral/fedisl/fedleo)에 맞는 시나리오 객체를 생성해 실행하는 단일 진입 함수 run_simulation().
import numpy as np
from typing import List, Optional

from LeoFL.Simulator.fl_core import SimConfig, LINK_FAIL_P_DEFAULT
from LeoFL.Simulator.scenarios import (
    AlwaysConnectedScenario,
    OffloadScenario,
    DecentralizedScenario,
    FedLEOScenario,
    FedISLScenario,
)

# run_simulator.py


def run_simulation(
    mode: str,
    client_datasets: List,
    test_dataset,
    num_rounds: int = 100,
    per_sat_minutes: Optional[List[int]] = None,
    train_minutes: int = 10,
    agg_name: str = "fedavg",
    target_acc: float = 70.0,
    device=None,
    hidden_size: int = 100,
    lr: float = 1e-3,
    seconds_per_round: int = 60,
    log_dir: str = ".",
    model_name: str = "dnn",
    decentral_peer_top_k: int = 2,
    decentral_max_dist_km: float = 2000.0,
    decentral_do_offload: bool = False,
    decentral_offload_ratio: float = 0.1,
    decentral_top_pairs_per_sat: int = 1,
    decentral_use_capacity_time_model: bool = False,
    # FedLEO용 파라미터
    fedleo_peer_top_k: int = 2,
    fedleo_max_dist_km: float = 2000.0,
    fedleo_offload_ratio: float = 0.1,
    fedleo_overload_threshold: float = 1.10,
    fedleo_max_offload_iter: int = 200,
    leader_top_k: int = 2,
    leader_mixing_beta: float = 0.5,
    # 통신 링크 랜덤성 (옵트인; None이면 기존 결정적 동작 그대로). decentral/fedisl에만 적용,
    # FedAvg("always")는 이상적 upper bound로 그대로 유지.
    link_fail_seed: Optional[int] = None,
    link_fail_p: float = LINK_FAIL_P_DEFAULT,
    # --- 기존 호환용 prefix들 ---
    log_prefix_base: str = "sat_events_base",
    log_prefix_var: str = "sat_events_vartrain",
    log_prefix_always: str = "sat_events_always",
    log_prefix_offload: str = "sat_events_offload",
    # --- 추가: decentral prefix & 공통 prefix ---
    log_prefix_decentral: str = "sat_events_decentral",
    log_prefix: Optional[str] = None,
    constellation_configs=None,
):
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
        log_prefix=log_prefix or "sat_events",
        model_name=model_name,
    )

    if constellation_configs is not None:
        if len(constellation_configs) != len(client_datasets):
            raise ValueError(
                f"constellation_configs length ({len(constellation_configs)}) "
                f"must match number of client_datasets ({len(client_datasets)}). "
                "Set NUM_CLIENTS = len(CONSTELLATION_CONFIGS)."
            )

    if per_sat_minutes is not None:
        if len(per_sat_minutes) != len(client_datasets):
            raise ValueError(
                f"per_sat_minutes length ({len(per_sat_minutes)}) "
                f"must match number of client_datasets ({len(client_datasets)})."
            )

    # 1) prefix 결정
    if log_prefix is None:
        if mode == "always":
            log_prefix = log_prefix_always
        elif mode == "offload":
            log_prefix = log_prefix_offload
        elif mode == "decentral":
            log_prefix = log_prefix_decentral
        elif mode == "fedisl":
            log_prefix = "sat_events_fedisl"
        elif mode == "fedleo":
            log_prefix = "sat_events_fedleo"
        else:
            log_prefix = "sat_events"

    # 2) scenario 생성
    if mode == "always":
        scenario = AlwaysConnectedScenario(
            client_datasets=client_datasets,
            test_dataset=test_dataset,
            config=cfg,
            train_minutes=train_minutes,
            per_sat_minutes=per_sat_minutes,
            log_prefix=log_prefix,
        )

    elif mode == "offload":
        scenario = OffloadScenario(
            client_datasets=client_datasets,
            test_dataset=test_dataset,
            config=cfg,
            max_dist_km=2000.0,
            offload_ratio=0.1,
            top_pairs_per_sat=1,
            log_prefix=log_prefix,
            do_offload=True,
            per_sat_minutes=per_sat_minutes,
            train_minutes=train_minutes,
            leader_top_k=leader_top_k,
            leader_mixing_beta=leader_mixing_beta,
        )

    elif mode == "decentral":
        link_fail_rng = (
            np.random.default_rng(link_fail_seed)
            if link_fail_seed is not None
            else None
        )
        scenario = DecentralizedScenario(
            client_datasets=client_datasets,
            test_dataset=test_dataset,
            config=cfg,
            log_prefix=log_prefix,
            per_sat_minutes=per_sat_minutes,
            train_minutes=train_minutes,
            max_dist_km=decentral_max_dist_km,
            peer_top_k=decentral_peer_top_k,
            do_offload=decentral_do_offload,
            offload_ratio=decentral_offload_ratio,
            top_pairs_per_sat=decentral_top_pairs_per_sat,
            use_capacity_time_model=decentral_use_capacity_time_model,
            link_fail_rng=link_fail_rng,
            link_fail_p=link_fail_p,
        )

    elif mode == "fedisl":
        link_fail_rng = (
            np.random.default_rng(link_fail_seed)
            if link_fail_seed is not None
            else None
        )
        scenario = FedISLScenario(
            client_datasets=client_datasets,
            test_dataset=test_dataset,
            config=cfg,
            log_prefix=log_prefix,
            per_sat_minutes=per_sat_minutes,
            train_minutes=train_minutes,
            link_fail_rng=link_fail_rng,
            link_fail_p=link_fail_p,
        )

    elif mode == "fedleo":
        scenario = FedLEOScenario(
            client_datasets=client_datasets,
            test_dataset=test_dataset,
            config=cfg,
            log_prefix=log_prefix,
            per_sat_minutes=per_sat_minutes,
            train_minutes=train_minutes,
            max_dist_km=fedleo_max_dist_km,
            peer_top_k=fedleo_peer_top_k,
            offload_ratio=fedleo_offload_ratio,
            overload_threshold=fedleo_overload_threshold,
            max_offload_iter=fedleo_max_offload_iter,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 3) 실행
    return scenario.run()
