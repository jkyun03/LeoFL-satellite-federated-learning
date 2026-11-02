# [초기 프로토타입] BASE/VAR/ALWAYS 시나리오의 시간-정확도 곡선을 확대 inset과 함께 그린다.
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import os


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
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(df_base["time"], df_base["accuracy"], label=labels[0], linewidth=2)
    ax.plot(df_var["time"], df_var["accuracy"], label=labels[1], linewidth=2)
    if df_alw is not None:
        ax.plot(df_alw["time"], df_alw["accuracy"], label=labels[2], linewidth=2)

    # ax.set_title("Accuracy over Time (BASE vs VAR vs ALWAYS)",fontsize=16, fontweight='bold')
    ax.set_xlabel("Time (minutes)", fontsize=25)
    ax.set_ylabel("Accuracy (%)", fontsize=25)
    ax.set_xlim(left=0, right=3500)
    ax.set_ylim(0, 70)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=18)  # 눈금 숫자 폰트 크기

    ax.legend(fontsize=25)
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
    axins.set_title("Zoom: 0–500 min", fontsize=11)
    axins.grid(True, linestyle="--", alpha=0.4)

    plt.subplots_adjust(left=0.08, right=0.97, top=0.97, bottom=0.12)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"그래프 저장 완료: {save_path}")
    else:
        plt.show()


plot_accuracy_over_time(
    base_csv="sat_events_base_accuracy.csv",
    var_csv="sat_events_vartrain_accuracy.csv",
    always_csv="sat_events_always_accuracy.csv",
    labels=("BASE", "VAR", "ALWAYS"),
    save_path="base_var_always_accuracy.png",
)
