"""
전송 지연을 반영한 라운드 완료 시간 재계산 (후처리, 학습 재실행 없음).

각 방법의 comm.csv에서 라운드별 model_bytes를 읽어,
  전송 시간 = model_bytes*8 / bandwidth (초)
를 그 라운드의 완료 시점에 더한다. (model_bytes에 이미 hop 수가 반영돼 있음)

그다음 accuracy.csv의 라운드->시간 매핑에 이 지연을 누적 반영하여,
"전송 지연 포함" 라운드 완료 시간을 무시(0)/100Gbps/10Gbps 세 가지로 비교한다.

logs_eurosat/ 가 있는 위치(= 서버 프로젝트 루트)에서 실행:
    python scripts/compute_tx_time.py
"""

import os
import numpy as np
import pandas as pd

LOG_DIR = "./logs_eurosat"
SECONDS_PER_ROUND = 60
BANDWIDTHS = {"ignore(0)": None, "100 Gbps": 100e9, "10 Gbps": 10e9}

# 사다리 5개 방법 prefix (seed 자리 {idx})
METHODS = {
    "FedISL": "starlink10p_seed{idx}_fedisl_cap",
    "FedISL+decentral": "starlink10p_seed{idx}_fedisl_mix",
    "FedISL+offload": "starlink10p_seed{idx}_fedisl_cap_off",
    "FedISL+decentral+off": "starlink10p_seed{idx}_fedisl_mix_off",
    "LH-FL": "starlink10p_seed{idx}_lhfl_b5000",
}
SEEDS = list(range(1, 11))  # seed1..seed10


def round_completion_minutes(comm_path, bw):
    """comm.csv 하나에서 '평균 라운드 완료 시간(분)'을 전송지연 bw로 계산.
    - 라운드 완료 시간(무시) = 연속된 집계 라운드 간격 * SECONDS_PER_ROUND
    - 전송지연 포함 = 각 집계 라운드에서 model_bytes 전송시간을 누적으로 추가
    """
    df = pd.read_csv(comm_path)
    # 집계가 일어난 행만 (model_hops>0). round 0의 offload-only 행은 제외.
    agg = df[df["model_hops"] > 0].copy().sort_values("time").reset_index(drop=True)
    if len(agg) < 2:
        return None

    # 각 집계의 '완료 라운드(time)' -> 실제 완료 시각(초)
    # 무시(0): 완료 시각 = time * SECONDS_PER_ROUND
    # 전송지연: 그 라운드까지 누적된 전송시간을 더함
    times_round = agg["time"].to_numpy(dtype=float)
    model_bytes = agg["model_bytes"].to_numpy(dtype=float)

    if bw is None:
        tx_cumsum = np.zeros(len(agg))
    else:
        tx_per_agg = model_bytes * 8.0 / bw  # 각 집계의 전송시간(초)
        tx_cumsum = np.cumsum(tx_per_agg)  # 누적

    finish_sec = times_round * SECONDS_PER_ROUND + tx_cumsum

    # 라운드 완료 시간 = 연속 집계 간의 간격
    gaps_sec = np.diff(finish_sec)
    return float(np.mean(gaps_sec)) / 60.0  # 분


def main():
    rows = []
    for label, prefix_tpl in METHODS.items():
        per_bw = {}
        for bw_name, bw in BANDWIDTHS.items():
            vals = []
            for s in SEEDS:
                prefix = prefix_tpl.format(idx=s)
                comm_path = os.path.join(LOG_DIR, f"{prefix}_comm.csv")
                if not os.path.exists(comm_path):
                    continue
                v = round_completion_minutes(comm_path, bw)
                if v is not None:
                    vals.append(v)
            if vals:
                per_bw[bw_name] = (float(np.mean(vals)), float(np.std(vals)))
        if per_bw:
            row = {"method": label}
            for bw_name in BANDWIDTHS:
                if bw_name in per_bw:
                    m, sd = per_bw[bw_name]
                    row[bw_name] = f"{m:.5f}"
                    row[f"{bw_name}_std"] = f"{sd:.5f}"
            rows.append(row)

    if not rows:
        print("[WARN] comm.csv를 찾지 못했습니다. LOG_DIR 경로를 확인하세요:", LOG_DIR)
        return

    out = pd.DataFrame(rows)
    print("\n=== 라운드 완료 시간 (분), 전송지연 반영 비교 ===")
    # 보기 좋게 핵심 컬럼만
    show_cols = ["method"] + [
        c for c in ["ignore(0)", "100 Gbps", "10 Gbps"] if c in out.columns
    ]
    print(out[show_cols].to_string(index=False))

    # 차이(delta) 요약
    print("\n=== 전송지연으로 인한 증가량 (분) ===")
    for _, r in out.iterrows():
        base = float(r.get("ignore(0)", "nan"))
        d100 = float(r.get("100 Gbps", base)) - base
        d10 = float(r.get("10 Gbps", base)) - base
        print(
            f"  {r['method']:<22} +100Gbps: {d100:+.6f} min,  +10Gbps: {d10:+.6f} min"
        )

    save = os.path.join(LOG_DIR, "table_round_completion_with_tx_delay.csv")
    out.to_csv(save, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] {save}")


if __name__ == "__main__":
    main()
