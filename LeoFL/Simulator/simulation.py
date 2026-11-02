# simulation.py
# UNR-IDD 데이터로 Simulator 패키지의 시나리오들을 실행·비교하던 초기 실험 진입점(결과는 logs/).
import os
import random
import numpy as np
import torch

from LeoFL.Simulator.fl_core import (
    NIDDataset,
    make_clients_from_dataset,
    FIXED_CONSTELLATION,
)
from LeoFL.Simulator.run_simulator import run_simulation
from LeoFL.Simulator.analysis_utils import (
    plot_accuracy_from_csv,
    plot_class_distribution_from_csv,
    plot_meeting_counts_heatmap,
    plot_capacity_from_csv,
    plot_normalized_load,
    plot_binned_heatmap,
    plot_capacity_summary,
    plot_class_distribution_summary,
    plot_multiple_accuracy,
    plot_multiple_communication,
    plot_train_time_candles_from_csv,
)

from LeoFL.Simulator.constellations_starlink import STARLINK_LIKE_SMALL
# --- 스타링크옵션을 끄려면 USE_STARLINK = False 로 바꾸세요 ---

N_ROUNDS = 300


# --- 재현성 고정 ---
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


GLOBAL_SEED = 20240301
set_global_seed(GLOBAL_SEED)


# --- 클라이언트 분포 디버그 출력용 ---
def print_client_class_distribution(client_datasets):
    """
    각 클라이언트(dataset)의 클래스 분포를 출력한다.
    """
    import numpy as np

    for idx, ds in enumerate(client_datasets):
        y_np = ds.Y.numpy().astype(int)
        classes, counts = np.unique(y_np, return_counts=True)

        print(f"\n=== Client {idx} ===")
        print(f"총 샘플 수: {len(ds)}")
        for c, cnt in zip(classes, counts):
            print(f"  class {c}: {cnt} samples")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 데이터 경로 ---
train_path = "LeoFL/UNR-IDD/UNR-IDD_train.feather"
test_path = "LeoFL/UNR-IDD/UNR-IDD_test.feather"
# --- 위성 궤도 설정 ---
USE_STARLINK = False  # 여기 True/False 로 모드 토글

if USE_STARLINK:
    CONSTELLATION_CONFIGS = STARLINK_LIKE_SMALL
else:
    CONSTELLATION_CONFIGS = FIXED_CONSTELLATION
# --- 클라이언트 분배 설정 ---
NUM_CLIENTS = len(CONSTELLATION_CONFIGS)  # 위성과 동일 개수 권장

print("[ENV] USE_STARLINK =", USE_STARLINK)
print("[ENV] num_sats(CONSTELLATION_CONFIGS) =", len(CONSTELLATION_CONFIGS))


rng = np.random.default_rng(GLOBAL_SEED)

# 예시: 5~30분 랜덤 (논문에서 "range [5,30]"로 명시 가능)
per_sat_minutes = rng.integers(low=5, high=31, size=NUM_CLIENTS).tolist()

# (옵션) 분포 로그 저장
print(
    "[ENV] per_sat_minutes summary:",
    f"min={min(per_sat_minutes)}, max={max(per_sat_minutes)}, mean={np.mean(per_sat_minutes):.2f}",
)

NONIID = False  # False면 IID
ALPHA = 0.5  # Non-IID 강도(작을수록 더 불균형)

# --- 원본 train 한 번만 로드 (스케일러/인코더 학습용) ---
central_train = NIDDataset(train_path)

# --- 클라이언트 자동 분할(IID/Non-IID) ---
client_datasets = make_clients_from_dataset(
    dataset=central_train,
    num_clients=NUM_CLIENTS,
    split_mode="non",
    alpha=ALPHA,
    seed=GLOBAL_SEED,
)

# 분포 확인용 출력
print_client_class_distribution(client_datasets)


# --- 테스트셋 (같은 인코더/스케일러 사용) ---
test_dataset = NIDDataset(
    test_path,
    category_encoder=central_train.category_encoder,
    scaler=central_train.scaler,
)

# --- 로그 디렉토리 ---
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

"""
# ===== 시나리오 1: BASE (연결성 + 고정 학습시간) =====
run_simulation(
    mode="base",
    client_datasets=client_datasets,
    test_dataset=test_dataset,
    num_rounds=N_ROUNDS,
    train_minutes=10,            # 분 단위
    agg_name="fedavg",
    target_acc=70.0,
    device=device,
    seconds_per_round=60,
    log_dir=LOG_DIR,
    log_prefix_base="sat_events_base",
    constellation_configs=CONSTELLATION_CONFIGS,
)

# ===== 시나리오 2: VARTRAIN (연결성 + 위성별 학습시간 가변) =====
per_sat_minutes = [10, 20, 30, 40, 50, 60, 70]   # 클라이언트 수와 동일 길이
run_simulation(
    mode="vartrain",
    client_datasets=client_datasets,
    test_dataset=test_dataset,
    per_sat_minutes=per_sat_minutes,
    num_rounds=N_ROUNDS,
    agg_name="fedavg",
    target_acc=70.0,
    device=device,
    seconds_per_round=60,
    log_dir=LOG_DIR,
    log_prefix_var="sat_events_vartrain",
    constellation_configs=CONSTELLATION_CONFIGS,
)

# ===== 시나리오 3: ALWAYS (항상 연결) =====
run_simulation(
    mode="always",
    client_datasets=client_datasets,
    test_dataset=test_dataset,
    num_rounds=N_ROUNDS,
    train_minutes=10,            # BASE와 공정 비교
    agg_name="fedavg",
    target_acc=70.0,
    device=device,
    seconds_per_round=60,
    log_dir=LOG_DIR,
    log_prefix_always="sat_events_always",
    constellation_configs=CONSTELLATION_CONFIGS,
)


# ===== 시나리오 4: OFFLOAD (지상국 없이 계층형 FL) =====
mode = "offload"
model, best_acc = run_simulation(
    mode=mode,
    client_datasets=client_datasets,
    test_dataset=test_dataset,
    num_rounds=N_ROUNDS,          # 테스트용
    agg_name="fedavg",
    target_acc=70.0,
    device=device,
    seconds_per_round=60,
    log_dir=LOG_DIR,
    constellation_configs=CONSTELLATION_CONFIGS,
)

# ===== 시나리오 5: DECENTRALIZED (Gossip/Ring FL) =====
mode = "decentral"
model, best_acc = run_simulation(
    mode=mode,
    client_datasets=client_datasets,
    test_dataset=test_dataset,
    num_rounds=N_ROUNDS,
    agg_name="decentral",
    target_acc=70.0,
    device=device,
    seconds_per_round=60,
    log_dir=LOG_DIR,
    constellation_configs=CONSTELLATION_CONFIGS,
)

print(f"\n[DONE] mode={mode}, best accuracy = {best_acc:.2f}%")

"""


def make_fresh_clients():
    return make_clients_from_dataset(
        dataset=central_train,
        num_clients=NUM_CLIENTS,
        split_mode="non",
        alpha=ALPHA,
        seed=GLOBAL_SEED,
    )


# 1) Ideal connectivity upper bound (formerly ALWAYS)
client_datasets = make_fresh_clients()
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
    log_prefix="sat_events_ideal",  # <-- 이름 변경
    constellation_configs=CONSTELLATION_CONFIGS,
)

# 2) Hybrid-Offload (ours)
client_datasets = make_fresh_clients()
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
    log_prefix="sat_events_offload",
    constellation_configs=CONSTELLATION_CONFIGS,
)

# 3) Fully decentralized
client_datasets = make_fresh_clients()
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
    log_prefix="sat_events_decentral",
    constellation_configs=CONSTELLATION_CONFIGS,
)


# --- 그래프들 ---
acc_csv = os.path.join(LOG_DIR, "sat_events_offload_accuracy.csv")
if os.path.exists(acc_csv):
    plot_accuracy_from_csv(
        acc_csv,
        title=f"Accuracy",
        save_path=acc_csv.replace(".csv", ".png"),
    )

before_csv = os.path.join(LOG_DIR, "sat_events_offload_classdist_before.csv")
after_csv = os.path.join(LOG_DIR, "sat_events_offload_classdist_after.csv")
if os.path.exists(before_csv):
    plot_class_distribution_from_csv(
        before_csv,
        title="Class Distribution Before Offload",
        save_path=before_csv.replace(".csv", ".png"),
    )
if os.path.exists(after_csv):
    plot_class_distribution_from_csv(
        after_csv,
        title="Class Distribution After Offload",
        save_path=after_csv.replace(".csv", ".png"),
    )

meeting_csv = os.path.join(LOG_DIR, "sat_events_offload_meeting_counts.csv")
if os.path.exists(meeting_csv):
    plot_meeting_counts_heatmap(
        meeting_csv,
        save_path=meeting_csv.replace(".csv", ".png"),
    )

cap_csv = os.path.join(LOG_DIR, "sat_events_offload_capacity.csv")
if os.path.exists(cap_csv):
    plot_capacity_from_csv(
        cap_csv,
        save_path=cap_csv.replace(".csv", ".png"),
    )

norm_save = os.path.join(LOG_DIR, "sat_events_offload_normalized_load.png")
plot_normalized_load(before_csv, after_csv, cap_csv, save_path=norm_save)

compare_csvs = [
    os.path.join(LOG_DIR, "sat_events_base_accuracy.csv"),
    os.path.join(LOG_DIR, "sat_events_always_accuracy.csv"),
    os.path.join(LOG_DIR, "sat_events_vartrain_accuracy.csv"),
    os.path.join(LOG_DIR, "sat_events_offload_accuracy.csv"),
    os.path.join(LOG_DIR, "sat_events_decentral_accuracy.csv"),
]

compare_labels = [
    "BASE",
    "ALWAYS",
    "VARTRAIN",
    "HYBRID-OFFLOAD",
    "DECENTRALIZED",
]

plot_multiple_accuracy(
    csv_paths=[
        "logs/sat_events_ideal_accuracy.csv",
        "logs/sat_events_offload_accuracy.csv",
        "logs/sat_events_decentral_accuracy.csv",
    ],
    labels=[
        "CENTRALIZED FL",
        "HYBRID-OFFLOAD",
        "DECENTRALIZED",
    ],
    max_x=1400,
    save_path="logs/accuracy_comparison.png",
)

comm_csvs = [
    "logs/sat_events_ideal_comm.csv",
    "logs/sat_events_offload_comm.csv",
    "logs/sat_events_decentral_comm.csv",
]

plot_multiple_communication(
    csv_paths=comm_csvs,
    labels=["CENTRALIZED FL", "HYBRID-OFFLOAD", "DECENTRALIZED FL"],
    which="model_hops",
    mode="per_round",
    max_x=N_ROUNDS,
    title="Per-Round Model Communication (Transmission Count)",
    save_path="logs/comm_model_hops_comparison.png",
)


SUMMARY_DIR = os.path.join(LOG_DIR, "summary")
os.makedirs(SUMMARY_DIR, exist_ok=True)

plot_binned_heatmap(
    "./logs/sat_events_offload_meeting_counts.csv",
    num_bins=20,
    save_path=os.path.join(SUMMARY_DIR, "meeting_20x20.png"),
)

plot_capacity_summary(
    "./logs/sat_events_offload_capacity.csv",
    save_path=os.path.join(SUMMARY_DIR, "capacity_summary.png"),
)

plot_class_distribution_summary(
    "./logs/sat_events_offload_classdist_after.csv",
    save_path=os.path.join(SUMMARY_DIR, "class_summary.png"),
)


plot_train_time_candles_from_csv(
    "logs/sat_events_ideal_train_time.csv",
    save_path="logs/train_time_candles_ideal.png",
)
