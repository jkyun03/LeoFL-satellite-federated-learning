# LeoFL/Simulator/analysis_utils.py
# 실험 로그(CSV)를 읽어 정확도·통신량·라운드 완료시간·부하 분포 등 논문용 그래프와 multi-seed 요약표를 만드는 분석 함수 모음.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Optional


def apply_paper_plot_style():
    mpl.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 19,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 16,
            "figure.titlesize": 20,
            # 논문용 선/축 스타일
            "axes.linewidth": 1.4,
            "lines.linewidth": 2.8,
            "lines.markersize": 6,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


# Colorblind-friendly palette + grayscale-friendly line styles

PAPER_COLORS = [
    "#4C78A8",  # muted blue
    "#72B7B2",  # soft teal
    "#E45756",  # warm red
    "#F2A541",  # soft orange
    "#8E6C8A",  # muted purple
    "#6B6B6B",  # gray
]

PAPER_LINESTYLES = ["-", "--", "-.", ":"]
PAPER_MARKERS = ["o", "s", "^", "D", "v", "P"]


def _paper_series_style(label: str, idx: int = 0):
    """
    색상 + 선스타일 + 마커를 함께 지정.
    FedAvg / FedISL / FedLEO / LH-FL을 명시적으로 구분.
    """
    text = str(label).upper().replace(" ", "")

    # FedAvg / Centralized FL
    if "FEDAVG" in text or "CENTRALIZED" in text:
        return {
            "color": "#4C78A8",  # blue
            "linestyle": "--",
            "marker": "o",
            "linewidth": 2.8,
            "markersize": 6,
        }

    # FedISL
    if "FEDISL" in text:
        return {
            "color": "#72B7B2",  # teal
            "linestyle": "-.",
            "marker": "s",
            "linewidth": 2.8,
            "markersize": 6,
        }

    # FedLEO
    if "FEDLEO" in text:
        return {
            "color": "#8E6C8A",  # purple
            "linestyle": ":",
            "marker": "^",
            "linewidth": 2.8,
            "markersize": 6,
        }

    # LH-FL / Proposed
    if "LH-FL" in text or "HYBRID" in text or "OFFLOAD" in text:
        return {
            "color": "#E45756",  # red
            "linestyle": "-",
            "marker": "D",
            "linewidth": 3.2,
            "markersize": 6,
        }

    # Partial FedAvg rho variants
    if "RHO=1.0" in text or "\\RHO=1.0" in text:
        return {
            "color": "#4C78A8",
            "linestyle": "--",
            "marker": "o",
            "linewidth": 2.8,
            "markersize": 6,
        }

    if "RHO=0.75" in text or "\\RHO=0.75" in text:
        return {
            "color": "#72B7B2",
            "linestyle": "-.",
            "marker": "s",
            "linewidth": 2.8,
            "markersize": 6,
        }

    if "RHO=0.5" in text or "\\RHO=0.5" in text:
        return {
            "color": "#8E6C8A",
            "linestyle": ":",
            "marker": "^",
            "linewidth": 2.8,
            "markersize": 6,
        }

    # 기존 Decentralized label 호환
    if "DECENTRALIZED" in text:
        return {
            "color": "#72B7B2",
            "linestyle": "-.",
            "marker": "s",
            "linewidth": 2.8,
            "markersize": 6,
        }

    return {
        "color": PAPER_COLORS[idx % len(PAPER_COLORS)],
        "linestyle": PAPER_LINESTYLES[idx % len(PAPER_LINESTYLES)],
        "marker": PAPER_MARKERS[idx % len(PAPER_MARKERS)],
        "linewidth": 2.8,
        "markersize": 6,
    }


def _markevery(x, n_markers: int = 10):
    """
    marker가 너무 많이 찍히지 않도록 간격 조절.
    """
    try:
        return max(1, len(x) // n_markers)
    except Exception:
        return 1


apply_paper_plot_style()


def _get_elapsed_time_from_df(df, seconds_per_round=60, time_unit="min"):
    """
    DataFrame에서 x축용 elapsed time 생성.
    우선순위:
      1) time 컬럼이 있으면 그대로 사용
      2) round 컬럼이 있으면 round * seconds_per_round 로 시간 변환
      3) 둘 다 없으면 index 사용

    반환:
      x_values, x_label
    """
    if "time" in df.columns:
        x = pd.to_numeric(df["time"], errors="coerce").values

        if time_unit == "sec":
            return x, "Time (sec)"
        elif time_unit == "min":
            return x / 60.0, "Time (min)"
        elif time_unit == "hour":
            return x / 3600.0, "Time (hour)"
        else:
            return x, "Time"

    elif "round" in df.columns:
        r = pd.to_numeric(df["round"], errors="coerce").values

        if time_unit == "sec":
            return r * seconds_per_round, "Time (sec)"
        elif time_unit == "min":
            return r * seconds_per_round / 60.0, "Time (min)"
        elif time_unit == "hour":
            return r * seconds_per_round / 3600.0, "Time (hour)"
        else:
            return r, "Round"

    else:
        x = np.arange(len(df))
        return x, "Index"


def plot_accuracy_from_csv(
    csv_path: str,
    title: str = None,
    save_path: str = None,
    seconds_per_round: int = 60,
    time_unit: str = "min",
):
    df = pd.read_csv(csv_path)

    if "accuracy" not in df.columns:
        raise ValueError("CSV에 'accuracy' 컬럼이 필요합니다.")

    x, xlabel = _get_elapsed_time_from_df(
        df,
        seconds_per_round=seconds_per_round,
        time_unit=time_unit,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(x, df["accuracy"].values, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy (%)")
    plt.title(title or "Test accuracy over time")
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[GRAPH][ACCURACY] saved: {save_path}")
    else:
        plt.show()


def plot_class_distribution_from_csv(
    csv_path: str, title: str = None, save_path: str = None
):
    df = pd.read_csv(csv_path)
    if "client_id" not in df.columns:
        raise ValueError("CSV에 'client_id' 컬럼이 없습니다.")
    class_cols = [c for c in df.columns if c.startswith("class_")]
    if not class_cols:
        raise ValueError("class_로 시작하는 컬럼이 없습니다.")

    plot_df = df.set_index("client_id")[class_cols]
    plt.figure(figsize=(10, 6))
    plot_df.plot(kind="bar", stacked=True)
    plt.xlabel("Client (Satellite) ID")
    plt.ylabel("Number of samples")
    plt.title(title or "Class distribution per client")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[GRAPH][CLASSDIST] saved: {save_path}")
    else:
        plt.show()


def plot_meeting_counts_heatmap(csv_path: str, save_path: str = None):
    mc = np.loadtxt(csv_path, delimiter=",")
    plt.figure(figsize=(6, 5))
    im = plt.imshow(mc, cmap="viridis")
    plt.colorbar(im, label="# meetings")
    plt.xticks(
        range(mc.shape[0]), [f"sat_{i + 1}" for i in range(mc.shape[0])], rotation=45
    )
    plt.yticks(range(mc.shape[0]), [f"sat_{i + 1}" for i in range(mc.shape[0])])
    plt.title("Inter-satellite Meeting Counts Heatmap")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[GRAPH][MEETING] saved: {save_path}")
    else:
        plt.show()


def plot_capacity_from_csv(csv_path: str, save_path: Optional[str] = None):
    """
    sat_events_offload_capacity.csv를 읽어서
    위성별 train_capacity (그리고 있으면 Ni_after) 를 시각화.

    - train_capacity + Ni_after 둘 다 있으면: plane별 scatter
    - Ni_after가 없으면: 위성별 train_capacity 막대그래프만 그림
    """
    df = pd.read_csv(csv_path)

    if "train_capacity" not in df.columns:
        raise ValueError(
            "CSV에 'train_capacity' 컬럼이 없습니다. 저장 단계 코드를 다시 확인하세요."
        )

    has_ni_after = "Ni_after" in df.columns

    plt.figure(figsize=(8, 6))

    if has_ni_after and "plane_id" in df.columns:
        # plane별 capacity vs Ni_after scatter
        ax = plt.gca()
        for plane_id, sub in df.groupby("plane_id"):
            ax.scatter(
                sub["train_capacity"],
                sub["Ni_after"],
                label=f"plane {plane_id}",
                alpha=0.8,
            )
        ax.set_xlabel("train_capacity (relative)")
        ax.set_ylabel("Ni_after (num samples after offload)")
        ax.set_title("Satellite train capacity vs samples after offload")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
    else:
        # Ni_after가 없으면: 위성별 capacity 막대그래프
        x_labels = df["sat_name"] if "sat_name" in df.columns else df.index.astype(str)
        plt.bar(x_labels, df["train_capacity"])
        plt.xlabel("Satellite")
        plt.ylabel("train_capacity (relative)")
        plt.title("Train capacity per satellite")
        plt.xticks(rotation=45)
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[SAVE][PLOT] {save_path}")
    else:
        plt.show()


def plot_train_time_candles_from_csv(csv_path: str, save_path: str = None):
    """
    prefix_train_time.csv (columns: round, sat_id, train_seconds)를 읽어서
    위성별 학습시간 분포(min~max 범위 + mean 표시)를 그림.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)

    required = {"sat_id", "train_seconds"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain {required}. got={set(df.columns)}")

    g = df.groupby("sat_id")["train_seconds"]
    stats = pd.DataFrame(
        {
            "min": g.min(),
            "max": g.max(),
            "mean": g.mean(),
            "median": g.median(),
        }
    ).reset_index()

    x = stats["sat_id"].values
    ymin = stats["min"].values
    ymax = stats["max"].values
    ymean = stats["mean"].values
    ymed = stats["median"].values

    plt.figure(figsize=(10, 5))

    # min~max "wick"
    plt.vlines(x, ymin, ymax, linewidth=2)

    # mean / median markers
    plt.scatter(x, ymean, marker="o", label="mean")
    plt.scatter(x, ymed, marker="x", label="median")

    plt.xlabel("Satellite (sat_id)")
    plt.ylabel("Train time (seconds)")
    plt.title("Per-satellite Train Time Range (min-max) with Mean/Median")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVE][TRAIN_TIME_PLOT] {save_path}")
    else:
        plt.show()


def plot_normalized_load(
    before_csv: str,
    after_csv: str,
    capacity_csv: str,
    save_path: str = None,
    title=None,
):
    """
    Offloading 전/후의 normalized training load (N_i / train_capacity)를 시각화.
    논문용:
    - 제목 제거 가능
    - Before/After 색상 및 선스타일 구분
    - 흑백 출력 대비 linestyle/marker 구분
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    apply_paper_plot_style()

    # --- Load data ---
    cap = pd.read_csv(capacity_csv)
    before = pd.read_csv(before_csv)
    after = pd.read_csv(after_csv)

    capacities = cap["train_capacity"].values

    Ni_before = (
        before[[c for c in before.columns if c.startswith("class_")]].sum(axis=1).values
    )
    Ni_after = (
        after[[c for c in after.columns if c.startswith("class_")]].sum(axis=1).values
    )

    norm_before = Ni_before / capacities
    norm_after = Ni_after / capacities

    x = np.arange(len(capacities))

    plt.figure(figsize=(10, 6))

    plt.plot(
        x,
        norm_before,
        label="Before Offload",
        color="#1F77B4",  # blue
        linestyle="--",
        marker="o",
        linewidth=2.8,
        markersize=5,
        markevery=max(1, len(x) // 16),
    )

    plt.plot(
        x,
        norm_after,
        label="After Offload",
        color="#D62728",  # red
        linestyle="-",
        marker="D",
        linewidth=3.0,
        markersize=5,
        markevery=max(1, len(x) // 16),
    )

    plt.xlabel("Satellite ID (index)")
    plt.ylabel(r"$N_i$ / train capacity")

    # 그래프 내부 title 제거
    if title is not None and str(title).strip() != "":
        plt.title(title)

    plt.grid(True, linestyle=":", alpha=0.55)
    plt.legend(frameon=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVE][NORMALIZED_LOAD] {save_path}")
    else:
        plt.show()

    plt.close()


def plot_multiple_accuracy(
    csv_paths,
    labels,
    title="Accuracy Comparison",
    save_path=None,
    max_x=None,
    seconds_per_round: int = 60,
    time_unit: str = "min",
):
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.figure(figsize=(10, 6))
    xlabel = "Time"

    for csv, label in zip(csv_paths, labels):
        if not os.path.exists(csv):
            print(f"[WARN] skip missing file: {csv}")
            continue

        df = pd.read_csv(csv)

        if "accuracy" not in df.columns:
            print(f"[WARN] missing accuracy column: {csv}")
            continue

        x, xlabel = _get_elapsed_time_from_df(
            df,
            seconds_per_round=seconds_per_round,
            time_unit=time_unit,
        )
        y = df["accuracy"].values

        if max_x is not None:
            mask = x <= max_x
            x = x[mask]
            y = y[mask]

        plt.plot(x, y, label=label, linewidth=2)

    plt.xlabel(xlabel)
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if max_x is not None:
        plt.xlim(0, max_x)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[MULTI-ACC] saved: {save_path}")
    else:
        plt.show()


def plot_binned_heatmap(
    csv_path, num_bins=20, title="Binned Meeting Heatmap", save_path=None
):
    """
    큰 meeting_counts 행렬을 num_bins × num_bins 로 축소하여 표시.
    예: 80×80 → 20×20
    """

    mc = np.loadtxt(csv_path, delimiter=",")

    # 원본 크기
    N = mc.shape[0]
    bin_size = N // num_bins

    # binning (mean pooling)
    binned = np.zeros((num_bins, num_bins))

    for i in range(num_bins):
        for j in range(num_bins):
            block = mc[
                i * bin_size : (i + 1) * bin_size, j * bin_size : (j + 1) * bin_size
            ]
            binned[i, j] = block.mean()

    plt.figure(figsize=(6, 5))
    im = plt.imshow(binned, cmap="viridis")
    plt.colorbar(im, label="avg meetings per bin")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SUMMARY][HEATMAP] saved: {save_path}")
    else:
        plt.show()


def plot_class_distribution_summary(csv_path, save_path=None):
    """
    위성이 많아 상세 bar chart가 복잡할 때:
    - 각 class별 평균 샘플 수
    - boxplot 로 분포 요약
    """

    df = pd.read_csv(csv_path)
    class_cols = [c for c in df.columns if c.startswith("class_")]
    class_data = df[class_cols]

    # ----- 평균 bar -----
    plt.figure(figsize=(7, 4))
    class_data.mean().plot(kind="bar", color="steelblue")
    plt.ylabel("Avg samples per satellite")
    plt.title("Class Distribution Summary (Mean)")
    plt.tight_layout()

    if save_path:
        p = save_path.replace(".png", "_mean.png")
        plt.savefig(p, dpi=300, bbox_inches="tight")
        print(f"[SUMMARY][CLASS MEAN] saved: {p}")
    else:
        plt.show()

    # ----- boxplot -----
    plt.figure(figsize=(7, 4))
    plt.boxplot(class_data.values, labels=class_cols, showfliers=False)
    plt.ylabel("Samples")
    plt.title("Class Distribution Summary (Boxplot)")
    plt.tight_layout()

    if save_path:
        p = save_path.replace(".png", "_box.png")
        plt.savefig(p, dpi=300, bbox_inches="tight")
        print(f"[SUMMARY][CLASS BOX] saved: {p}")
    else:
        plt.show()


def plot_capacity_summary(csv_path, save_path=None):
    """
    너무 많은 위성 capacity 막대그래프 대신,
    정렬된 simple line plot 제공.
    """

    df = pd.read_csv(csv_path)
    caps = np.sort(df["train_capacity"].values)

    plt.figure(figsize=(8, 4))
    plt.plot(caps, marker="o", linestyle="-")
    plt.xlabel("Satellite (sorted)")
    plt.ylabel("train_capacity")
    plt.title("Train Capacity Summary (Sorted)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SUMMARY][CAPACITY] saved: {save_path}")
    else:
        plt.show()


def plot_multiple_communication(
    csv_paths,
    labels,
    title="Cumulative Model Communication (Transmission Count)",
    save_path=None,
    max_x=None,
    which="model",
    mode="cumulative",
    seconds_per_round: int = 60,
    time_unit: str = "min",
):

    apply_paper_plot_style()

    which_map = {
        "model": "model_bytes",
        "offload": "offload_bytes",
        "model_hops": "model_hops",
        "offload_hops": "offload_hops",
    }
    if which not in which_map:
        raise ValueError(f"Invalid which={which}. Use one of {list(which_map.keys())}")

    if mode not in ("cumulative", "per_round"):
        raise ValueError("mode must be one of {'cumulative','per_round'}")

    col = which_map[which]
    is_hops = col.endswith("_hops")

    plt.figure(figsize=(10, 6))
    xlabel = "Time"

    for idx, (csv, label) in enumerate(zip(csv_paths, labels)):
        if not os.path.exists(csv):
            print(f"[WARN] skip missing file: {csv}")
            continue

        df = pd.read_csv(csv)

        if col not in df.columns:
            print(
                f"[WARN] missing column '{col}' in {csv} (columns={list(df.columns)})"
            )
            continue

        x, xlabel = _get_elapsed_time_from_df(
            df,
            seconds_per_round=seconds_per_round,
            time_unit=time_unit,
        )

        y = pd.to_numeric(df[col], errors="coerce").fillna(0).values

        if not is_hops:
            y = y / (1024 * 1024)  # bytes -> MB

        if mode == "cumulative":
            y = np.cumsum(y)

        if max_x is not None:
            mask = x <= float(max_x)
            x = x[mask]
            y = y[mask]

        style = _paper_series_style(label, idx)

        plt.plot(
            x,
            y,
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            markevery=max(1, len(x) // 10),
        )

    plt.xlabel(xlabel)

    if is_hops:
        if mode == "per_round":
            ylabel = "Transmission count per time step"
        else:
            ylabel = "Cumulative transmission"
    else:
        if mode == "per_round":
            ylabel = "Communication (MB) per time step"
        else:
            ylabel = "Cumulative communication (MB)"

    plt.ylabel(f"{ylabel} (log)")
    plt.yscale("log")

    # 그래프 내부 title 제거
    if title is not None and str(title).strip() != "":
        plt.title(title)

    plt.grid(True, linestyle=":", alpha=0.55)
    plt.legend(frameon=True)
    plt.tight_layout()

    if max_x is not None:
        plt.xlim(0, float(max_x))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[COMM] saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_accuracy_vs_comm(acc_csv, comm_csv, label="run", save_path=None):
    acc = pd.read_csv(acc_csv)
    comm = pd.read_csv(comm_csv)

    key = None
    if "round" in acc.columns and "round" in comm.columns:
        key = "round"
    elif "time" in acc.columns and "time" in comm.columns:
        key = "time"
    else:
        raise ValueError(
            "acc_csv와 comm_csv에 공통 key('round' 또는 'time')가 필요합니다."
        )

    comm["cum_model_mb"] = pd.to_numeric(comm["model_bytes"], errors="coerce").fillna(
        0
    ).cumsum() / (1024 * 1024)
    m = pd.merge(acc, comm[[key, "cum_model_mb"]], on=key, how="inner")

    plt.figure(figsize=(7, 5))
    plt.plot(m["cum_model_mb"], m["accuracy"], marker="o")
    plt.xlabel("Cumulative Model Communication (MB)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy vs Communication ({label})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[ACC-vs-COMM] saved: {save_path}")
    else:
        plt.show()


def plot_global_update_intervals(
    csv_paths,
    labels,
    save_path=None,
    seconds_per_round=60,
    time_unit="min",
    title="Global Update Interval per Cycle",
):
    """
    accuracy csv의 round 컬럼을 이용해
    연속된 global aggregation 사이의 간격을 계산해서 그림.

    예:
      rounds = [0, 37, 79, 120]
      intervals = [37, 42, 41]
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    for csv, label in zip(csv_paths, labels):
        if not os.path.exists(csv):
            print(f"[WARN] skip missing file: {csv}")
            continue

        df = pd.read_csv(csv)

        if "round" not in df.columns:
            print(f"[WARN] missing 'round' column: {csv}")
            continue

        rounds = pd.to_numeric(df["round"], errors="coerce").dropna().values

        if len(rounds) < 2:
            print(f"[WARN] not enough global updates in {csv}")
            continue

        intervals = np.diff(rounds)

        if time_unit == "sec":
            intervals = intervals * seconds_per_round
            ylabel = "Cycle completion time (sec)"
        elif time_unit == "min":
            intervals = intervals * seconds_per_round / 60.0
            ylabel = "Cycle completion time (min)"
        elif time_unit == "hour":
            intervals = intervals * seconds_per_round / 3600.0
            ylabel = "Cycle completion time (hour)"
        else:
            ylabel = "Cycle completion time (rounds)"

        x = np.arange(1, len(intervals) + 1)  # global update index
        plt.plot(x, intervals, marker="o", linewidth=2, label=label)

    plt.xlabel("Global update index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[GRAPH][UPDATE_INTERVAL] saved: {save_path}")
    else:
        plt.show()


def _get_update_completion_intervals_from_accuracy(
    acc_csv_path,
    seconds_per_round=60,
    time_unit="min",
):
    """
    accuracy.csv의 round 기록 간격으로 update completion time 계산.

    accuracy.csv:
      round, accuracy

    반환:
      consecutive update events 사이의 completion time array
    """
    if not os.path.exists(acc_csv_path):
        raise FileNotFoundError(f"accuracy csv not found: {acc_csv_path}")

    df = pd.read_csv(acc_csv_path)

    if "round" not in df.columns:
        raise ValueError(f"accuracy csv must contain 'round': {acc_csv_path}")

    rounds = (
        pd.to_numeric(df["round"], errors="coerce")
        .dropna()
        .sort_values()
        .drop_duplicates()
        .values
    )

    if len(rounds) < 2:
        return np.array([], dtype=float)

    intervals = np.diff(rounds)

    # 혹시 같은 round가 중복 기록되었거나 비정상 interval이 있으면 제거
    intervals = intervals[intervals > 0]

    if time_unit == "sec":
        return intervals * seconds_per_round
    elif time_unit == "hour":
        return intervals * seconds_per_round / 3600.0
    else:
        return intervals * seconds_per_round / 60.0


def plot_update_completion_summary_from_accuracy(
    acc_csv_paths,
    labels,
    title=None,
    save_path=None,
    save_csv_path=None,
    seconds_per_round=60,
    time_unit="min",
    plot_type="bar",  # "bar" or "box"
    ylim=None,
):
    """
    Update completion time을 summary plot으로 시각화.

    계산 기준:
      - accuracy.csv의 연속된 update 발생 round 간격
      - 즉, np.diff(accuracy_rounds)

    plot_type:
      - "bar": mean ± std bar plot
      - "box": distribution boxplot
    """
    apply_paper_plot_style()

    interval_data = []
    rows = []

    for acc_csv, label in zip(acc_csv_paths, labels):
        if not os.path.exists(acc_csv):
            print(f"[WARN] missing accuracy csv: {acc_csv}")
            interval_data.append(np.array([], dtype=float))
            rows.append(
                {
                    "Method": label,
                    "Num intervals": 0,
                    "Mean": np.nan,
                    "Std": np.nan,
                    "Median": np.nan,
                    "Min": np.nan,
                    "Max": np.nan,
                }
            )
            continue

        intervals = _get_update_completion_intervals_from_accuracy(
            acc_csv_path=acc_csv,
            seconds_per_round=seconds_per_round,
            time_unit=time_unit,
        )

        interval_data.append(intervals)

        if len(intervals) == 0:
            rows.append(
                {
                    "Method": label,
                    "Num intervals": 0,
                    "Mean": np.nan,
                    "Std": np.nan,
                    "Median": np.nan,
                    "Min": np.nan,
                    "Max": np.nan,
                }
            )
        else:
            rows.append(
                {
                    "Method": label,
                    "Num intervals": int(len(intervals)),
                    "Mean": float(np.mean(intervals)),
                    "Std": float(np.std(intervals, ddof=1))
                    if len(intervals) > 1
                    else 0.0,
                    "Median": float(np.median(intervals)),
                    "Min": float(np.min(intervals)),
                    "Max": float(np.max(intervals)),
                }
            )

    summary_df = pd.DataFrame(rows)

    if save_csv_path:
        summary_df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE][UPDATE_COMPLETION_SUMMARY] {save_csv_path}")

    valid_rows = summary_df.dropna(subset=["Mean"]).copy()

    if len(valid_rows) == 0:
        print("[WARN] No valid update completion intervals. Skip plot.")
        return summary_df

    if time_unit == "sec":
        ylabel = "Round completion time (sec)"
    elif time_unit == "hour":
        ylabel = "Round completion time (hour)"
    else:
        ylabel = "Round completion time (min)"

    fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor="white")
    ax.set_facecolor("white")

    methods = valid_rows["Method"].values
    colors = [
        _paper_series_style(method, idx)["color"] for idx, method in enumerate(methods)
    ]

    if plot_type == "box":
        valid_interval_data = [
            interval_data[labels.index(method)] for method in methods
        ]

        bp = ax.boxplot(
            valid_interval_data,
            labels=methods,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
            widths=0.55,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 5,
            },
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)

        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.2)

        ax.set_xticklabels(methods, rotation=0, fontsize=12)

    else:
        x = np.arange(len(valid_rows))
        means = valid_rows["Mean"].values
        stds = valid_rows["Std"].values

        ax.bar(
            x,
            means,
            yerr=stds,
            capsize=5,
            color=colors,
            edgecolor="black",
            linewidth=0.7,
            width=0.62,
            error_kw={
                "elinewidth": 1.2,
                "capthick": 1.2,
            },
        )

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=0, fontsize=12)

        # value label: mean ± std
        y_offset = max(means) * 0.015 if max(means) > 0 else 0.2
        for xi, mean, std in zip(x, means, stds):
            ax.text(
                xi,
                mean + std + y_offset,
                f"{mean:.1f}±{std:.1f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )

    ax.set_ylabel(ylabel, fontsize=14)

    if title is not None and str(title).strip() != "":
        ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.grid(False, axis="x")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[GRAPH][UPDATE_COMPLETION_SUMMARY] saved: {save_path}")
    else:
        plt.show()

    plt.close()

    return summary_df


def plot_multiple_accuracy_vs_time(
    acc_csv_paths,
    train_time_csv_paths,
    labels,
    title="Accuracy Convergence Over Time",
    save_path=None,
    time_unit="min",
    max_x=None,
):
    """
    accuracy csv + train_time csv를 결합해서 실제 시간 기준 accuracy 비교 그래프 생성.
    """
    import os
    import matplotlib.pyplot as plt

    if not (len(acc_csv_paths) == len(train_time_csv_paths) == len(labels)):
        raise ValueError("acc_csv_paths, train_time_csv_paths, labels length mismatch")

    plt.figure(figsize=(10, 6))

    for acc_csv, tt_csv, label in zip(acc_csv_paths, train_time_csv_paths, labels):
        if not os.path.exists(acc_csv):
            print(f"[WARN] missing accuracy csv: {acc_csv}")
            continue
        if not os.path.exists(tt_csv):
            print(f"[WARN] missing train_time csv: {tt_csv}")
            continue

        df = build_accuracy_time_dataframe(acc_csv, tt_csv)

        if time_unit == "sec":
            x = df["time_sec"]
            xlabel = "Time (sec)"
        elif time_unit == "hour":
            x = df["time_hour"]
            xlabel = "Time (hour)"
        else:
            x = df["time_min"]
            xlabel = "Time (min)"

        y = df["accuracy"]

        if max_x is not None:
            mask = x <= max_x
            x = x[mask]
            y = y[mask]

        plt.plot(x, y, label=label, linewidth=2, marker="o", markersize=3)

    plt.xlabel(xlabel)
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if max_x is not None:
        plt.xlim(0, max_x)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[GRAPH][ACC_TIME] saved: {save_path}")
    else:
        plt.show()


def build_accuracy_time_dataframe(
    acc_csv_path: str,
    train_time_csv_path: str,
    save_csv_path: str = None,
):
    """
    accuracy csv(round, accuracy) + train_time csv(round, sat_id, train_seconds)를 이용해
    실제 누적 시간 기준 accuracy dataframe 생성.

    중요:
    - accuracy round와 train_time round는 정확히 같은 값이 아닐 수 있음
    - 따라서 'global update 순서' 기준으로 맞춘다
    - 첫 accuracy row(round=0, acc=0)는 초기 placeholder로 시간 0
    """
    import os
    import pandas as pd

    if not os.path.exists(acc_csv_path):
        raise FileNotFoundError(f"accuracy csv not found: {acc_csv_path}")
    if not os.path.exists(train_time_csv_path):
        raise FileNotFoundError(f"train_time csv not found: {train_time_csv_path}")

    acc_df = pd.read_csv(acc_csv_path).copy()
    tt_df = pd.read_csv(train_time_csv_path).copy()

    # 숫자형 변환
    acc_df["round"] = pd.to_numeric(acc_df["round"], errors="coerce")
    acc_df["accuracy"] = pd.to_numeric(acc_df["accuracy"], errors="coerce")
    tt_df["round"] = pd.to_numeric(tt_df["round"], errors="coerce")
    tt_df["train_seconds"] = pd.to_numeric(tt_df["train_seconds"], errors="coerce")

    acc_df = (
        acc_df.dropna(subset=["round", "accuracy"])
        .sort_values("round")
        .reset_index(drop=True)
    )
    tt_df = (
        tt_df.dropna(subset=["round", "train_seconds"])
        .sort_values("round")
        .reset_index(drop=True)
    )

    # round별 cycle completion time = max(train_seconds)
    cycle_df = (
        tt_df.groupby("round", as_index=False)["train_seconds"]
        .max()
        .rename(columns={"train_seconds": "cycle_time_sec"})
        .sort_values("round")
        .reset_index(drop=True)
    )

    # accuracy의 첫 행은 초기 placeholder (round=0, acc=0)
    # 그 다음 accuracy 값들부터 실제 cycle time과 1:1 순서 매칭
    if len(acc_df) - 1 > len(cycle_df):
        raise ValueError(
            f"Not enough cycle times: accuracy updates={len(acc_df) - 1}, cycle rows={len(cycle_df)}"
        )

    # 초기 row
    first_row = acc_df.iloc[[0]].copy()
    first_row["cycle_time_sec"] = 0.0
    first_row["time_sec"] = 0.0
    first_row["time_min"] = 0.0
    first_row["time_hour"] = 0.0

    # 실제 업데이트 rows
    acc_updates = acc_df.iloc[1:].copy().reset_index(drop=True)
    cycle_used = cycle_df.iloc[: len(acc_updates)].copy().reset_index(drop=True)

    acc_updates["cycle_time_sec"] = cycle_used["cycle_time_sec"].values
    acc_updates["time_sec"] = acc_updates["cycle_time_sec"].cumsum()
    acc_updates["time_min"] = acc_updates["time_sec"] / 60.0
    acc_updates["time_hour"] = acc_updates["time_sec"] / 3600.0

    out_df = pd.concat([first_row, acc_updates], ignore_index=True)

    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path) or ".", exist_ok=True)
        out_df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE][ACC_TIME] {save_csv_path}")

    return out_df


"""
def plot_multiple_accuracy_round(
    csv_paths,
    labels,
    title="Accuracy vs Global Update Round",
    save_path=None,
    use_step=True,
    max_x=None,
):
    
    apply_paper_plot_style()
    plt.figure(figsize=(10, 6))

    for idx, (csv, label) in enumerate(zip(csv_paths, labels)):
        if not os.path.exists(csv):
            print(f"[WARN] skip missing file: {csv}")
            continue

        df = pd.read_csv(csv)
        if "round" not in df.columns or "accuracy" not in df.columns:
            print(f"[WARN] invalid accuracy csv: {csv}")
            continue

        x = pd.to_numeric(df["round"], errors="coerce")
        y = pd.to_numeric(df["accuracy"], errors="coerce")

        mask = x.notna() & y.notna()

        if max_x is not None:
            mask = mask & (x <= max_x)

        x = x[mask]
        y = y[mask]

        style = _paper_series_style(label, idx)

        if use_step:
            plt.step(
                x, y,
                where="post",
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
            )
        else:
            plt.plot(
                x, y,
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=style["markersize"],
                markevery=_markevery(x, n_markers=10),
            )

    plt.xlabel("Global update round")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.55)
    plt.legend(frameon=True)

    if max_x is not None:
        plt.xlim(0, max_x)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[GRAPH][ACC_ROUND] saved: {save_path}")
    else:
        plt.show()

    plt.close()
"""


def plot_multiple_accuracy_time(
    acc_csv_paths,
    train_time_csv_paths,
    labels,
    title="Accuracy vs Time",
    save_path=None,
    time_unit="min",
    use_step=True,
    max_x=None,
):
    """
    accuracy csv + train_time csv를 결합해서 실제 시간 기준 accuracy 그래프 생성.
    """
    apply_paper_plot_style()
    plt.figure(figsize=(10, 6))

    for idx, (acc_csv, tt_csv, label) in enumerate(
        zip(acc_csv_paths, train_time_csv_paths, labels)
    ):
        if not os.path.exists(acc_csv):
            print(f"[WARN] missing accuracy csv: {acc_csv}")
            continue
        if not os.path.exists(tt_csv):
            print(f"[WARN] missing train_time csv: {tt_csv}")
            continue

        df = build_accuracy_time_dataframe(acc_csv, tt_csv)

        if time_unit == "sec":
            x = df["time_sec"]
            xlabel = "Time (sec)"
        elif time_unit == "hour":
            x = df["time_hour"]
            xlabel = "Time (hour)"
        else:
            x = df["time_min"]
            xlabel = "Time (min)"

        y = df["accuracy"]

        if max_x is not None:
            mask = x <= max_x
            x = x[mask]
            y = y[mask]

        style = _paper_series_style(label, idx)

        if use_step:
            plt.step(
                x,
                y,
                where="post",
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
            )
        else:
            plt.plot(
                x,
                y,
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=style["markersize"],
                markevery=_markevery(x, n_markers=10),
            )

    plt.xlabel(xlabel)
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.55)
    plt.legend(frameon=True)
    plt.tight_layout()

    if max_x is not None:
        plt.xlim(0, max_x)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[GRAPH][ACC_TIME] saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_cycle_completion_time(
    train_time_csv_paths,
    labels,
    title="Cycle Completion Time",
    save_path=None,
    time_unit="min",
    max_x=None,
    smoothing_window=5,
    x_axis="simulation_round",
):
    """
    train_time csv에서 round별 max(train_seconds)를 구해 cycle completion time 그래프 생성.

    x_axis:
      - "simulation_round": 실제 simulation round를 x축으로 사용
      - "update_index": 각 방법별 global update index를 x축으로 사용
    """
    apply_paper_plot_style()
    plt.figure(figsize=(10, 6))

    for idx, (csv, label) in enumerate(zip(train_time_csv_paths, labels)):
        if not os.path.exists(csv):
            print(f"[WARN] missing train_time csv: {csv}")
            continue

        df = pd.read_csv(csv)
        if not {"round", "train_seconds"}.issubset(df.columns):
            print(f"[WARN] invalid train_time csv: {csv}")
            continue

        cycle_df = (
            df.groupby("round", as_index=False)["train_seconds"]
            .max()
            .sort_values("round")
            .reset_index(drop=True)
        )

        if x_axis == "simulation_round":
            x = cycle_df["round"].values
            xlabel = "Simulation round"
        elif x_axis == "update_index":
            x = np.arange(1, len(cycle_df) + 1)
            xlabel = "Global update index"
        else:
            raise ValueError("x_axis must be 'simulation_round' or 'update_index'")

        if time_unit == "sec":
            y = cycle_df["train_seconds"].values
            ylabel = "Cycle completion time (sec)"
        elif time_unit == "hour":
            y = (cycle_df["train_seconds"] / 3600.0).values
            ylabel = "Cycle completion time (hour)"
        else:
            y = (cycle_df["train_seconds"] / 60.0).values
            ylabel = "Cycle completion time (min)"

        if smoothing_window is not None and smoothing_window > 1:
            y = pd.Series(y).rolling(smoothing_window, min_periods=1).mean().values

        if max_x is not None:
            mask = x <= max_x
            x = x[mask]
            y = y[mask]

        style = _paper_series_style(label, idx)

        plt.plot(
            x,
            y,
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=style["markersize"],
            markevery=_markevery(x, n_markers=10),
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.55)
    plt.legend(frameon=True)
    plt.tight_layout()

    if max_x is not None:
        plt.xlim(0, max_x)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[GRAPH][CYCLE_TIME] saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_multiple_accuracy_sim_time(
    acc_csv_paths,
    labels,
    title="Accuracy vs Simulation Time",
    save_path=None,
    seconds_per_round=60,
    time_unit="min",
    use_step=True,
    max_x=None,
):
    apply_paper_plot_style()
    plt.figure(figsize=(10, 6))

    if time_unit == "sec":
        factor = seconds_per_round
        xlabel = "Time (sec)"
    elif time_unit == "hour":
        factor = seconds_per_round / 3600.0
        xlabel = "Time (hour)"
    else:
        factor = seconds_per_round / 60.0
        xlabel = "Time (min)"

    for idx, (csv, label) in enumerate(zip(acc_csv_paths, labels)):
        if not os.path.exists(csv):
            print(f"[WARN] missing accuracy csv: {csv}")
            continue

        df = pd.read_csv(csv)

        if "round" not in df.columns or "accuracy" not in df.columns:
            print(f"[WARN] invalid accuracy csv: {csv}")
            continue

        x = pd.to_numeric(df["round"], errors="coerce") * factor
        y = pd.to_numeric(df["accuracy"], errors="coerce")

        mask = x.notna() & y.notna()
        if max_x is not None:
            mask = mask & (x <= max_x)

        x = x[mask]
        y = y[mask]

        # max_x까지 마지막 accuracy를 수평 연장
        if max_x is not None and len(x) > 0:
            last_x = float(x.iloc[-1] if hasattr(x, "iloc") else x[-1])
            last_y = float(y.iloc[-1] if hasattr(y, "iloc") else y[-1])

            if last_x < max_x:
                x = np.append(np.asarray(x), max_x)
                y = np.append(np.asarray(y), last_y)

        style = _paper_series_style(label, idx)

        if use_step:
            plt.step(
                x,
                y,
                where="post",
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
            )
        else:
            plt.plot(
                x,
                y,
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=style["markersize"],
                markevery=_markevery(x, n_markers=10),
            )

    plt.xlabel(xlabel)
    plt.ylabel("Accuracy (%)")

    if title is not None and str(title).strip() != "":
        plt.title(title)

    plt.grid(True, linestyle=":", alpha=0.55)
    plt.legend(frameon=True)

    if max_x is not None:
        plt.xlim(0, max_x)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[GRAPH][ACC_SIM_TIME] saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_multiple_accuracy_update_round(
    csv_paths,
    labels,
    title=None,
    save_path=None,
    use_step=False,
    max_x=None,
):
    """
    Accuracy vs actual communication / hierarchical update round.

    주의:
    - accuracy.csv의 'round' 컬럼은 simulation tick 또는 elapsed-time index에 가까움.
    - 여기서는 accuracy 기록 순서를 기준으로 실제 update index를 만든다.
      0 = initial model
      1 = first aggregation/mixing update
      2 = second aggregation/mixing update
      ...
    """
    apply_paper_plot_style()
    plt.figure(figsize=(10, 6))

    for idx, (csv, label) in enumerate(zip(csv_paths, labels)):
        if not os.path.exists(csv):
            print(f"[WARN] missing accuracy csv: {csv}")
            continue

        df = pd.read_csv(csv)

        if "accuracy" not in df.columns:
            print(f"[WARN] invalid accuracy csv: {csv}")
            continue

        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
        df = df.dropna(subset=["accuracy"]).reset_index(drop=True)

        # 실제 communication/hierarchical update index
        x = np.arange(len(df))
        y = df["accuracy"].values

        if max_x is not None:
            mask = x <= max_x
            x = x[mask]
            y = y[mask]

        style = _paper_series_style(label, idx)

        if use_step:
            plt.step(
                x,
                y,
                where="post",
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
            )
        else:
            plt.plot(
                x,
                y,
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=style["markersize"],
                markevery=_markevery(x, n_markers=10),
            )

    plt.xlabel("Communication update round")
    plt.ylabel("Accuracy (%)")

    if title is not None and str(title).strip() != "":
        plt.title(title)

    plt.grid(True, linestyle=":", alpha=0.55)
    plt.legend(frameon=True)

    if max_x is not None:
        plt.xlim(0, max_x)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[GRAPH][ACC_UPDATE_ROUND] saved: {save_path}")
    else:
        plt.show()

    plt.close()


def _derive_comm_total_column(comm: pd.DataFrame, comm_col: str) -> pd.DataFrame:
    """
    comm.csv에 comm_col로 요청된 파생 통신 지표(total_hops/total_bytes/total_mb)를
    계산해 컬럼으로 추가한다. model_hops/offload_hops/model_bytes/offload_bytes
    기준이며, 누락된 컬럼은 0으로 처리한다.

    _build_accuracy_comm_dataframe과 make_comm_cost_*_multiseed(final_round=True)가
    동일한 파생 로직을 공유하기 위한 헬퍼.
    """
    if comm_col == "total_hops":
        if "model_hops" not in comm.columns:
            comm["model_hops"] = 0
        if "offload_hops" not in comm.columns:
            comm["offload_hops"] = 0

        comm["model_hops"] = pd.to_numeric(comm["model_hops"], errors="coerce").fillna(
            0
        )
        comm["offload_hops"] = pd.to_numeric(
            comm["offload_hops"], errors="coerce"
        ).fillna(0)
        comm["total_hops"] = comm["model_hops"] + comm["offload_hops"]

    if comm_col == "total_bytes":
        if "model_bytes" not in comm.columns:
            comm["model_bytes"] = 0
        if "offload_bytes" not in comm.columns:
            comm["offload_bytes"] = 0

        comm["model_bytes"] = pd.to_numeric(
            comm["model_bytes"], errors="coerce"
        ).fillna(0)
        comm["offload_bytes"] = pd.to_numeric(
            comm["offload_bytes"], errors="coerce"
        ).fillna(0)
        comm["total_bytes"] = comm["model_bytes"] + comm["offload_bytes"]

    if comm_col == "total_mb":
        if "model_bytes" not in comm.columns:
            comm["model_bytes"] = 0
        if "offload_bytes" not in comm.columns:
            comm["offload_bytes"] = 0

        comm["model_bytes"] = pd.to_numeric(
            comm["model_bytes"], errors="coerce"
        ).fillna(0)
        comm["offload_bytes"] = pd.to_numeric(
            comm["offload_bytes"], errors="coerce"
        ).fillna(0)
        comm["total_mb"] = (comm["model_bytes"] + comm["offload_bytes"]) / (1024 * 1024)

    if comm_col not in comm.columns:
        raise ValueError(
            f"comm dataframe must contain '{comm_col}' after derivation, "
            f"columns={list(comm.columns)}"
        )

    return comm


def _build_accuracy_comm_dataframe(
    acc_csv_path,
    comm_csv_path,
    comm_col="model_hops",
):
    """
    accuracy.csv와 comm.csv를 결합.

    반환 컬럼:
      - round: accuracy가 기록된 simulation round
      - accuracy
      - update_round: accuracy 기록 순서 기준 update index
      - cumulative_comm: 해당 accuracy 시점까지 누적 communication cost

    comm.csv의 시간 컬럼명이 round가 아닐 수 있으므로
    round/time/t/step/round_idx 등을 자동 탐색한다.
    """
    acc = pd.read_csv(acc_csv_path).copy()
    comm = pd.read_csv(comm_csv_path).copy()

    if "round" not in acc.columns or "accuracy" not in acc.columns:
        raise ValueError(
            f"Invalid accuracy csv: {acc_csv_path}, columns={list(acc.columns)}"
        )

    comm = _derive_comm_total_column(comm, comm_col)

    # -----------------------------
    # 1) accuracy csv 정리
    # -----------------------------
    acc["round"] = pd.to_numeric(acc["round"], errors="coerce")
    acc["accuracy"] = pd.to_numeric(acc["accuracy"], errors="coerce")

    acc = (
        acc.dropna(subset=["round", "accuracy"])
        .sort_values("round")
        .reset_index(drop=True)
    )

    # -----------------------------
    # 2) comm csv의 시간/round 컬럼 자동 탐색
    # -----------------------------
    candidate_cols = [
        "round",
        "round_idx",
        "sim_round",
        "simulation_round",
        "time",
        "t",
        "step",
    ]

    comm_round_col = None
    for c in candidate_cols:
        if c in comm.columns:
            comm_round_col = c
            break

    if comm_round_col is None:
        print(
            f"[WARN] No round/time-like column found in {comm_csv_path}. "
            f"Use row index as communication event index. "
            f"columns={list(comm.columns)}"
        )
        comm["__round_for_merge__"] = np.arange(len(comm), dtype=float)
        comm_round_col = "__round_for_merge__"

    comm[comm_round_col] = pd.to_numeric(comm[comm_round_col], errors="coerce")
    comm[comm_col] = pd.to_numeric(comm[comm_col], errors="coerce").fillna(0)

    comm = (
        comm.dropna(subset=[comm_round_col])
        .sort_values(comm_round_col)
        .reset_index(drop=True)
    )

    # -----------------------------
    # 3) round별 communication 합산
    # -----------------------------
    comm_by_round = (
        comm.groupby(comm_round_col, as_index=False)[comm_col]
        .sum()
        .sort_values(comm_round_col)
        .reset_index(drop=True)
    )

    comm_by_round = comm_by_round.rename(columns={comm_round_col: "round"})
    comm_by_round["round"] = pd.to_numeric(comm_by_round["round"], errors="coerce")
    comm_by_round["cumulative_comm"] = comm_by_round[comm_col].cumsum()

    # round=0 cumulative=0 보장
    zero = pd.DataFrame(
        {
            "round": [0.0],
            "cumulative_comm": [0.0],
        }
    )

    comm_cum = pd.concat(
        [zero, comm_by_round[["round", "cumulative_comm"]]],
        ignore_index=True,
    )

    comm_cum = (
        comm_cum.groupby("round", as_index=False)["cumulative_comm"]
        .max()
        .sort_values("round")
        .reset_index(drop=True)
    )

    # -----------------------------
    # dtype 통일: merge_asof는 key dtype이 같아야 함
    # -----------------------------
    acc["round"] = pd.to_numeric(acc["round"], errors="coerce").astype(float)
    comm_cum["round"] = pd.to_numeric(comm_cum["round"], errors="coerce").astype(float)

    acc = (
        acc.dropna(subset=["round", "accuracy"])
        .sort_values("round")
        .reset_index(drop=True)
    )

    comm_cum = (
        comm_cum.dropna(subset=["round"]).sort_values("round").reset_index(drop=True)
    )

    # accuracy 시점까지 발생한 cumulative communication을 붙임
    out = pd.merge_asof(
        acc,
        comm_cum,
        on="round",
        direction="backward",
    )

    out["cumulative_comm"] = out["cumulative_comm"].fillna(0.0)
    out["update_round"] = np.arange(len(out))

    return out


def plot_multiple_accuracy_vs_cumulative_comm(
    acc_csv_paths,
    comm_csv_paths,
    labels,
    title=None,
    save_path=None,
    comm_col="model_hops",
    xscale_log=False,
    max_x=None,
):
    """
    Accuracy vs cumulative communication cost.

    권장 용도:
      - communication efficiency 비교
      - 같은 transmission budget에서 accuracy 비교
    """
    apply_paper_plot_style()
    plt.figure(figsize=(10, 6))

    for idx, (acc_csv, comm_csv, label) in enumerate(
        zip(acc_csv_paths, comm_csv_paths, labels)
    ):
        if not os.path.exists(acc_csv):
            print(f"[WARN] missing accuracy csv: {acc_csv}")
            continue
        if not os.path.exists(comm_csv):
            print(f"[WARN] missing communication csv: {comm_csv}")
            continue

        df = _build_accuracy_comm_dataframe(
            acc_csv_path=acc_csv,
            comm_csv_path=comm_csv,
            comm_col=comm_col,
        )

        x = df["cumulative_comm"].values
        y = df["accuracy"].values

        if xscale_log:
            mask = x > 0
            x = x[mask]
            y = y[mask]

        if max_x is not None:
            mask = x <= max_x
            x = x[mask]
            y = y[mask]

        style = _paper_series_style(label, idx)

        plt.plot(
            x,
            y,
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=style["markersize"],
            markevery=_markevery(x, n_markers=10),
        )

    plt.xlabel("Cumulative model transmissions")
    plt.ylabel("Accuracy (%)")

    if xscale_log:
        plt.xscale("log")

    if title is not None and str(title).strip() != "":
        plt.title(title)

    plt.grid(True, linestyle=":", alpha=0.55)
    plt.legend(frameon=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[GRAPH][ACC_COMM] saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_comm_cost_to_target_bar(
    acc_csv_paths,
    comm_csv_paths,
    labels,
    target_acc=85.0,
    save_path=None,
    save_csv_path=None,
    seconds_per_round=60,
    yscale_log=False,
):
    """
    Target accuracy에 도달할 때까지 필요한 communication cost bar plot.

    출력 table 컬럼:
      - Method
      - Target accuracy (%)
      - Update round to target
      - Simulation time to target (min)
      - Communication cost

    Figure:
      - horizontal bar plot
      - x-axis: model transmissions required to reach target accuracy
    """
    rows = []

    for acc_csv, comm_csv, label in zip(acc_csv_paths, comm_csv_paths, labels):
        if not os.path.exists(acc_csv) or not os.path.exists(comm_csv):
            rows.append(
                {
                    "Method": label,
                    "Target accuracy (%)": target_acc,
                    "Update round to target": None,
                    "Simulation time to target (min)": None,
                    "Communication cost": None,
                }
            )
            continue

        df = _build_accuracy_comm_dataframe(
            acc_csv_path=acc_csv,
            comm_csv_path=comm_csv,
            comm_col="model_hops",
        )

        hit = df[df["accuracy"] >= target_acc]

        if hit.empty:
            rows.append(
                {
                    "Method": label,
                    "Target accuracy (%)": target_acc,
                    "Update round to target": None,
                    "Simulation time to target (min)": None,
                    "Communication cost": None,
                }
            )
            continue

        first = hit.iloc[0]

        rows.append(
            {
                "Method": label,
                "Target accuracy (%)": target_acc,
                "Update round to target": int(first["update_round"]),
                "Simulation time to target (min)": float(
                    first["round"] * seconds_per_round / 60.0
                ),
                "Communication cost": int(first["cumulative_comm"]),
            }
        )

    out = pd.DataFrame(rows)

    if save_csv_path:
        out.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE][COMM_TARGET_TABLE] {save_csv_path}")

    plot_df = out.dropna(subset=["Communication cost"]).copy()

    if len(plot_df) == 0:
        print(
            "[WARN] No method reached target accuracy. Skip target communication bar plot."
        )
        return out

    # -------------------------------------------------
    # Paper-style horizontal bar plot
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor="white")
    ax.set_facecolor("white")

    methods = plot_df["Method"].values
    values = plot_df["Communication cost"].astype(float).values

    y = np.arange(len(methods))

    colors = [
        _paper_series_style(method, idx)["color"] for idx, method in enumerate(methods)
    ]

    ax.barh(
        y,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.65,
        height=0.58,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=14)

    ax.set_xlabel(
        f"Communication cost to reach {target_acc:.0f}% accuracy",
        fontsize=14,
    )

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=14)

    if yscale_log:
        ax.set_xscale("log")
        label_offset = max(values) * 0.04
    else:
        xmax = max(values) * 1.16
        ax.set_xlim(0, xmax)
        label_offset = max(values) * 0.012

    # value labels
    for yi, val in zip(y, values):
        ax.text(
            val + label_offset,
            yi,
            f"{int(val):,}",
            va="center",
            ha="left",
            fontsize=12,
        )

    ax.grid(True, axis="x", linestyle=":", alpha=0.35)
    ax.grid(False, axis="y")

    # remove unnecessary top/right visual weight
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # keep order: Centralized, Decentralized, LH-FL from top to bottom
    ax.invert_yaxis()

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[GRAPH][COMM_TARGET_BAR] saved: {save_path}")
    else:
        plt.show()

    plt.close()

    return out


def make_comm_cost_to_target_table(
    acc_csv_paths,
    comm_csv_paths,
    labels,
    target_acc=85.0,
    save_csv_path=None,
):
    rows = []

    for acc_csv, comm_csv, label in zip(acc_csv_paths, comm_csv_paths, labels):
        acc = pd.read_csv(acc_csv)
        comm = pd.read_csv(comm_csv)

        acc["round"] = pd.to_numeric(acc["round"], errors="coerce")
        acc["accuracy"] = pd.to_numeric(acc["accuracy"], errors="coerce")
        if "round" in comm.columns:
            comm_round_col = "round"
        elif "time" in comm.columns:
            comm_round_col = "time"
        else:
            raise ValueError(f"comm csv has no round/time column: {comm.columns}")

        comm[comm_round_col] = pd.to_numeric(comm[comm_round_col], errors="coerce")
        comm["model_hops"] = pd.to_numeric(comm["model_hops"], errors="coerce").fillna(
            0
        )

        hit = acc[acc["accuracy"] >= target_acc]

        if hit.empty:
            rows.append(
                {
                    "Method": label,
                    "Target accuracy (%)": target_acc,
                    "Reached round": None,
                    "Communication cost": None,
                }
            )
            continue

        target_round = float(hit.iloc[0]["round"])

        comm_until = comm[comm[comm_round_col] <= target_round]
        total_comm = comm_until["model_hops"].sum()

        rows.append(
            {
                "Method": label,
                "Target accuracy (%)": target_acc,
                "Reached round": int(target_round),
                "Communication cost": int(total_comm),
            }
        )

    out = pd.DataFrame(rows)

    if save_csv_path:
        out.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE][COMM_TARGET_TABLE] {save_csv_path}")

    return out


def plot_update_completion_time_from_accuracy(
    acc_csv_paths,
    labels,
    title=None,
    save_path=None,
    seconds_per_round=60,
    time_unit="min",
    max_x=None,
    smoothing_window=None,
):
    """
    accuracy.csv의 update 발생 round 간격으로 cycle/update completion time을 계산.

    의미:
      - x축: communication / hierarchical update index
      - y축: consecutive update events 사이의 elapsed time
    """
    apply_paper_plot_style()
    plt.figure(figsize=(10, 6))

    for idx, (csv, label) in enumerate(zip(acc_csv_paths, labels)):
        if not os.path.exists(csv):
            print(f"[WARN] missing accuracy csv: {csv}")
            continue

        df = pd.read_csv(csv)

        if "round" not in df.columns:
            print(f"[WARN] missing round column: {csv}")
            continue

        rounds = pd.to_numeric(df["round"], errors="coerce").dropna().values

        # 첫 row가 initial placeholder round=0이면 포함해도 diff 계산에는 문제 없음
        if len(rounds) < 2:
            print(f"[WARN] not enough update events in {csv}")
            continue

        intervals = np.diff(rounds)

        if time_unit == "sec":
            y = intervals * seconds_per_round
            ylabel = "Round completion time (sec)"
        elif time_unit == "hour":
            y = intervals * seconds_per_round / 3600.0
            ylabel = "Round completion time (hour)"
        else:
            y = intervals * seconds_per_round / 60.0
            ylabel = "Round completion time (min)"

        x = np.arange(1, len(y) + 1)

        if smoothing_window is not None and smoothing_window > 1:
            y = pd.Series(y).rolling(smoothing_window, min_periods=1).mean().values

        if max_x is not None:
            mask = x <= max_x
            x = x[mask]
            y = y[mask]

        style = _paper_series_style(label, idx)

        plt.plot(
            x,
            y,
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=style["markersize"],
            markevery=_markevery(x, n_markers=10),
        )

    plt.xlabel("Communication update index")
    plt.ylabel(ylabel)

    if title is not None and str(title).strip() != "":
        plt.title(title)

    plt.grid(True, linestyle=":", alpha=0.55)
    plt.legend(frameon=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[GRAPH][UPDATE_COMPLETION_TIME] saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_update_completion_reduction_bar(
    summary_csv_path,
    baseline_methods=("CENTRALIZED FL", "DECENTRALIZED FL"),
    proposed_method="LH-FL",
    save_path=None,
):
    """
    table_update_completion_time_summary.csv를 읽어서
    LH-FL의 update completion time reduction (%)을 그림.

    reduction (%) = (baseline_mean - proposed_mean) / baseline_mean * 100
    """
    apply_paper_plot_style()

    df = pd.read_csv(summary_csv_path)

    if "Method" not in df.columns or "Mean" not in df.columns:
        raise ValueError("summary csv must contain 'Method' and 'Mean' columns.")

    df["Method_norm"] = df["Method"].astype(str).str.upper().str.strip()
    df["Mean"] = pd.to_numeric(df["Mean"], errors="coerce")

    proposed_key = proposed_method.upper().strip()
    proposed_row = df[df["Method_norm"] == proposed_key]

    if proposed_row.empty:
        raise ValueError(f"Proposed method not found: {proposed_method}")

    proposed_mean = float(proposed_row.iloc[0]["Mean"])

    rows = []
    for base in baseline_methods:
        base_key = base.upper().strip()
        base_row = df[df["Method_norm"] == base_key]

        if base_row.empty:
            print(f"[WARN] baseline not found: {base}")
            continue

        base_mean = float(base_row.iloc[0]["Mean"])
        reduction = (base_mean - proposed_mean) / base_mean * 100.0
        speedup = base_mean / proposed_mean

        rows.append(
            {
                "Baseline": base,
                "Reduction (%)": reduction,
                "Speedup": speedup,
            }
        )

    plot_df = pd.DataFrame(rows)

    if plot_df.empty:
        print("[WARN] No valid baseline rows. Skip reduction plot.")
        return plot_df

    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor="white")
    ax.set_facecolor("white")

    x = np.arange(len(plot_df))
    y = plot_df["Reduction (%)"].values

    colors = ["#4C78A8", "#72B7B2"]

    ax.bar(
        x,
        y,
        color=colors[: len(x)],
        edgecolor="black",
        linewidth=0.7,
        width=0.6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"vs. {b.replace(' FL', '')}" for b in plot_df["Baseline"]],
        fontsize=12,
    )

    ax.set_ylabel("Reduction in update completion time (%)", fontsize=13)

    ymax = max(y) * 1.25
    ax.set_ylim(0, ymax)

    for xi, val, sp in zip(x, y, plot_df["Speedup"].values):
        ax.text(
            xi,
            val + ymax * 0.03,
            f"{val:.1f}%\n({sp:.2f}×)",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.grid(False, axis="x")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[GRAPH][UPDATE_COMPLETION_REDUCTION] saved: {save_path}")
    else:
        plt.show()

    plt.close()

    return plot_df


def plot_normalized_update_completion_time(
    summary_csv_path,
    reference_method="CENTRALIZED FL",
    save_path=None,
):
    """
    update completion time mean을 reference method 기준으로 정규화해서 그림.
    normalized time = method_mean / reference_mean
    """
    apply_paper_plot_style()

    df = pd.read_csv(summary_csv_path)

    if "Method" not in df.columns or "Mean" not in df.columns:
        raise ValueError("summary csv must contain 'Method' and 'Mean' columns.")

    df = df.copy()
    df["Mean"] = pd.to_numeric(df["Mean"], errors="coerce")
    df["Method_norm"] = df["Method"].astype(str).str.upper().str.strip()

    ref_key = reference_method.upper().strip()
    ref_row = df[df["Method_norm"] == ref_key]

    if ref_row.empty:
        raise ValueError(f"reference method not found: {reference_method}")

    ref_mean = float(ref_row.iloc[0]["Mean"])

    df["Normalized time"] = df["Mean"] / ref_mean

    fig, ax = plt.subplots(figsize=(8.0, 4.8), facecolor="white")
    ax.set_facecolor("white")

    methods = df["Method"].values
    values = df["Normalized time"].values

    colors = [
        _paper_series_style(method, idx)["color"] for idx, method in enumerate(methods)
    ]

    x = np.arange(len(methods))

    ax.bar(
        x,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        width=0.62,
    )

    ax.axhline(
        1.0,
        color="black",
        linestyle=":",
        linewidth=1.2,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)

    ax.set_ylabel("Normalized update completion time", fontsize=13)
    ax.set_ylim(0, max(values) * 1.18)

    for xi, val in zip(x, values):
        ax.text(
            xi,
            val + 0.025,
            f"{val:.2f}×",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.grid(False, axis="x")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[GRAPH][NORMALIZED_UPDATE_COMPLETION] saved: {save_path}")
    else:
        plt.show()

    plt.close()

    return df


def plot_update_completion_dumbbell(
    summary_csv_path,
    baseline_methods=("CENTRALIZED FL", "DECENTRALIZED FL"),
    proposed_method="LH-FL",
    save_path=None,
):
    """
    Baseline update completion time과 LH-FL update completion time을
    dumbbell plot으로 비교.

    입력:
      table_update_completion_time_summary.csv
      columns: Method, Mean, Std, ...
    """
    apply_paper_plot_style()

    df = pd.read_csv(summary_csv_path)

    if "Method" not in df.columns or "Mean" not in df.columns:
        raise ValueError("summary csv must contain 'Method' and 'Mean' columns.")

    df = df.copy()
    df["Method_norm"] = df["Method"].astype(str).str.upper().str.strip()
    df["Mean"] = pd.to_numeric(df["Mean"], errors="coerce")

    proposed_key = proposed_method.upper().strip()
    proposed_row = df[df["Method_norm"] == proposed_key]

    if proposed_row.empty:
        raise ValueError(f"Proposed method not found: {proposed_method}")

    proposed_mean = float(proposed_row.iloc[0]["Mean"])

    rows = []
    for base in baseline_methods:
        base_key = base.upper().strip()
        base_row = df[df["Method_norm"] == base_key]

        if base_row.empty:
            print(f"[WARN] baseline not found: {base}")
            continue

        baseline_mean = float(base_row.iloc[0]["Mean"])
        saved = baseline_mean - proposed_mean
        reduction = saved / baseline_mean * 100.0
        speedup = baseline_mean / proposed_mean

        rows.append(
            {
                "Baseline": base,
                "Baseline time": baseline_mean,
                "LH-FL time": proposed_mean,
                "Saved time": saved,
                "Reduction (%)": reduction,
                "Speedup": speedup,
            }
        )

    plot_df = pd.DataFrame(rows)

    if plot_df.empty:
        print("[WARN] No valid baseline rows. Skip dumbbell plot.")
        return plot_df

    # -----------------------------
    # Cleaner paper-style plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8.0, 3.6), facecolor="white")
    ax.set_facecolor("white")

    y = np.arange(len(plot_df))[::-1]  # top to bottom
    baseline_color = "#4C78A8"
    lhfl_color = "#E45756"
    line_color = "#6B6B6B"

    for row_idx, (_, row) in enumerate(plot_df.iterrows()):
        yi = y[row_idx]

        base_time = row["Baseline time"]
        lhfl_time = row["LH-FL time"]
        saved = row["Saved time"]
        reduction = row["Reduction (%)"]
        speedup = row["Speedup"]

        # connecting line
        ax.plot(
            [lhfl_time, base_time],
            [yi, yi],
            color=line_color,
            linewidth=2.2,
            alpha=0.75,
            zorder=1,
        )

        # baseline point
        ax.scatter(
            base_time,
            yi,
            s=120,
            color=baseline_color,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )

        # LH-FL point
        ax.scatter(
            lhfl_time,
            yi,
            s=130,
            color=lhfl_color,
            marker="D",
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        )

        # point value labels
        ax.text(
            lhfl_time - 0.18,
            yi,
            f"{lhfl_time:.1f}",
            ha="right",
            va="center",
            fontsize=11,
            color=lhfl_color,
        )

        ax.text(
            base_time + 0.18,
            yi,
            f"{base_time:.1f}",
            ha="left",
            va="center",
            fontsize=11,
            color=baseline_color,
        )

        # saved annotation above line
        mid_x = (lhfl_time + base_time) / 2.0
        ax.text(
            mid_x,
            yi + 0.18,
            f"saved {saved:.1f} min  ({reduction:.1f}%, {speedup:.2f}×)",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#333333",
        )

    ylabels = [base.replace(" FL", "") for base in plot_df["Baseline"].values]

    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=12)

    ax.set_xlabel("Average update completion time (min)", fontsize=13)

    xmin = min(plot_df["LH-FL time"].min(), plot_df["Baseline time"].min()) - 0.8
    xmax = max(plot_df["LH-FL time"].max(), plot_df["Baseline time"].max()) + 0.8
    ax.set_xlim(xmin, xmax)

    # direct mini legend outside plot area
    ax.scatter([], [], s=100, color=baseline_color, edgecolor="black", label="Baseline")
    ax.scatter(
        [], [], s=110, color=lhfl_color, marker="D", edgecolor="black", label="LH-FL"
    )
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        fontsize=12,
        handletextpad=0.5,
        columnspacing=1.5,
    )

    ax.grid(True, axis="x", linestyle=":", alpha=0.35)
    ax.grid(False, axis="y")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[GRAPH][UPDATE_COMPLETION_DUMBBELL] saved: {save_path}")
    else:
        plt.show()

    plt.close()

    return plot_df


def plot_update_completion_saved_segment_bar(
    summary_csv_path,
    baseline_methods=("CENTRALIZED FL", "DECENTRALIZED FL"),
    proposed_method="LH-FL",
    save_path=None,
):
    """
    Baseline update completion time을
    [LH-FL time] + [saved time] 형태의 stacked horizontal bar로 표시.

    예:
      Centralized:   25.0 min + saved 6.0 min = 31.0 min
      Decentralized: 25.0 min + saved 5.6 min = 30.6 min
    """
    apply_paper_plot_style()

    df = pd.read_csv(summary_csv_path)

    if "Method" not in df.columns or "Mean" not in df.columns:
        raise ValueError("summary csv must contain 'Method' and 'Mean' columns.")

    df = df.copy()
    df["Method_norm"] = df["Method"].astype(str).str.upper().str.strip()
    df["Mean"] = pd.to_numeric(df["Mean"], errors="coerce")

    proposed_key = proposed_method.upper().strip()
    proposed_row = df[df["Method_norm"] == proposed_key]

    if proposed_row.empty:
        raise ValueError(f"Proposed method not found: {proposed_method}")

    proposed_mean = float(proposed_row.iloc[0]["Mean"])

    rows = []
    for base in baseline_methods:
        base_key = base.upper().strip()
        base_row = df[df["Method_norm"] == base_key]

        if base_row.empty:
            print(f"[WARN] baseline not found: {base}")
            continue

        baseline_mean = float(base_row.iloc[0]["Mean"])
        saved = baseline_mean - proposed_mean
        reduction = saved / baseline_mean * 100.0
        speedup = baseline_mean / proposed_mean

        rows.append(
            {
                "Baseline": base,
                "Baseline time": baseline_mean,
                "LH-FL time": proposed_mean,
                "Saved time": saved,
                "Reduction (%)": reduction,
                "Speedup": speedup,
            }
        )

    plot_df = pd.DataFrame(rows)

    if plot_df.empty:
        print("[WARN] No valid baseline rows. Skip saved-segment plot.")
        return plot_df

    # -----------------------------
    # Stacked horizontal bar plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8.4, 3.9), facecolor="white")
    ax.set_facecolor("white")

    y = np.arange(len(plot_df))

    labels = [base.replace(" FL", "") for base in plot_df["Baseline"].values]

    lhfl_times = plot_df["LH-FL time"].values
    saved_times = plot_df["Saved time"].values
    baseline_times = plot_df["Baseline time"].values
    reductions = plot_df["Reduction (%)"].values

    lhfl_color = "#E45756"
    saved_color = "#F2A541"
    edge_color = "black"

    # LH-FL 부분
    ax.barh(
        y,
        lhfl_times,
        color=lhfl_color,
        edgecolor=edge_color,
        linewidth=0.7,
        height=0.52,
        label="LH-FL time",
    )

    # 절약된 시간 부분
    ax.barh(
        y,
        saved_times,
        left=lhfl_times,
        color=saved_color,
        edgecolor=edge_color,
        linewidth=0.7,
        height=0.52,
        label="Time saved",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)

    ax.set_xlabel("Average update completion time (min)", fontsize=13)

    xmax = max(baseline_times) * 1.12
    ax.set_xlim(0, xmax)

    # 라벨 표시
    for yi, lhfl, saved, total, red in zip(
        y, lhfl_times, saved_times, baseline_times, reductions
    ):
        # LH-FL 내부 라벨
        ax.text(
            lhfl / 2,
            yi,
            f"LH-FL\n{lhfl:.1f} min",
            ha="center",
            va="center",
            fontsize=10.5,
            color="white",
            fontweight="bold",
        )

        # saved segment 라벨
        ax.text(
            lhfl + saved / 2,
            yi,
            f"saved\n{saved:.1f} min",
            ha="center",
            va="center",
            fontsize=10.5,
            color="black",
            fontweight="bold",
        )

        # 전체 baseline 시간
        ax.text(
            total + xmax * 0.015,
            yi,
            f"{total:.1f} min\n({red:.1f}%↓)",
            ha="left",
            va="center",
            fontsize=10.5,
            color="#222222",
        )

    ax.grid(True, axis="x", linestyle=":", alpha=0.35)
    ax.grid(False, axis="y")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        fontsize=11,
    )

    ax.invert_yaxis()

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[GRAPH][UPDATE_COMPLETION_SAVED_SEGMENT] saved: {save_path}")
    else:
        plt.show()

    plt.close()

    return plot_df


def make_comm_cost_to_target_multiseed(
    log_dir,
    seeds,
    method_prefix_map,
    labels,
    target_acc=85.0,
    comm_col="model_hops",
    save_csv_path=None,
    final_round=False,
):
    """
    여러 seed에 대해 communication cost를 계산.

    final_round=False (기본, 기존 동작):
      accuracy.csv에서 처음 target_acc 이상이 되는 시점을 찾고,
      comm.csv에서 해당 시점까지 cumulative communication cost를 계산.
      target_acc에 도달하지 못한 run은 Reached=False로 표시되고 집계에서 제외된다.

    final_round=True:
      target_acc 도달 여부와 무관하게, comm.csv 전체(시뮬레이션이 끝난 시점,
      즉 num_rounds 시점)까지의 누적 communication cost를 계산한다.
      target에 도달하지 못하는 방법도 비교할 수 있다.

    반환:
      raw_df: seed별 결과
      summary_df: method별 mean/std/min/max/count
    """
    import os
    import numpy as np
    import pandas as pd

    rows = []

    for label in labels:
        prefix_template = method_prefix_map[label]

        for idx, seed in enumerate(seeds, start=1):
            prefix = prefix_template.format(idx=idx, seed=seed)

            acc_csv = os.path.join(log_dir, f"{prefix}_accuracy.csv")
            comm_csv = os.path.join(log_dir, f"{prefix}_comm.csv")

            if final_round:
                if not os.path.exists(comm_csv):
                    print(
                        f"[WARN][MULTI-SEED-FINAL] missing comm csv: method={label}, seed={seed}, prefix={prefix}"
                    )
                    rows.append(
                        {
                            "Method": label,
                            "Seed": seed,
                            "Reached": False,
                            "Communication cost": np.nan,
                            "Update round to target": np.nan,
                            "Simulation round to target": np.nan,
                            "Reached accuracy": np.nan,
                        }
                    )
                    continue

                comm = _derive_comm_total_column(pd.read_csv(comm_csv).copy(), comm_col)
                total = float(
                    pd.to_numeric(comm[comm_col], errors="coerce").fillna(0).sum()
                )

                final_acc = np.nan
                if os.path.exists(acc_csv):
                    acc_df = pd.read_csv(acc_csv)
                    acc_df["round"] = pd.to_numeric(acc_df["round"], errors="coerce")
                    acc_df["accuracy"] = pd.to_numeric(
                        acc_df["accuracy"], errors="coerce"
                    )
                    acc_df = acc_df.dropna(subset=["round", "accuracy"]).sort_values(
                        "round"
                    )
                    if not acc_df.empty:
                        final_acc = float(acc_df["accuracy"].iloc[-1])

                rows.append(
                    {
                        "Method": label,
                        "Seed": seed,
                        "Reached": True,
                        "Communication cost": total,
                        "Update round to target": np.nan,
                        "Simulation round to target": np.nan,
                        "Reached accuracy": final_acc,
                    }
                )
                continue

            if not os.path.exists(acc_csv) or not os.path.exists(comm_csv):
                print(
                    f"[WARN][MULTI-SEED] missing files: method={label}, seed={seed}, prefix={prefix}"
                )
                rows.append(
                    {
                        "Method": label,
                        "Seed": seed,
                        "Reached": False,
                        "Communication cost": np.nan,
                        "Update round to target": np.nan,
                        "Simulation round to target": np.nan,
                        "Reached accuracy": np.nan,
                    }
                )
                continue

            df = _build_accuracy_comm_dataframe(
                acc_csv_path=acc_csv,
                comm_csv_path=comm_csv,
                comm_col=comm_col,
            )

            hit = df[df["accuracy"] >= target_acc]

            if hit.empty:
                print(
                    f"[WARN][MULTI-SEED] {label}, seed={seed} did not reach {target_acc}%"
                )
                rows.append(
                    {
                        "Method": label,
                        "Seed": seed,
                        "Reached": False,
                        "Communication cost": np.nan,
                        "Update round to target": np.nan,
                        "Simulation round to target": np.nan,
                        "Reached accuracy": np.nan,
                    }
                )
                continue

            first = hit.iloc[0]

            rows.append(
                {
                    "Method": label,
                    "Seed": seed,
                    "Reached": True,
                    "Communication cost": float(first["cumulative_comm"]),
                    "Update round to target": int(first["update_round"]),
                    "Simulation round to target": float(first["round"]),
                    "Reached accuracy": float(first["accuracy"]),
                }
            )

    raw_df = pd.DataFrame(rows)

    reached_df = raw_df[raw_df["Reached"] == True].copy()

    summary_df = reached_df.groupby("Method", as_index=False).agg(
        Mean=("Communication cost", "mean"),
        Std=("Communication cost", "std"),
        Min=("Communication cost", "min"),
        Max=("Communication cost", "max"),
        Num_runs=("Communication cost", "count"),
    )

    # label 순서 유지
    summary_df["Method"] = pd.Categorical(
        summary_df["Method"],
        categories=labels,
        ordered=True,
    )
    summary_df = summary_df.sort_values("Method").reset_index(drop=True)

    # std NaN 방지
    # summary_df["Std"] = summary_df["Std"].fillna(0.0)

    if save_csv_path is not None:
        raw_path = save_csv_path.replace(".csv", "_raw.csv")
        raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
        summary_df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE][MULTI-SEED RAW] {raw_path}")
        print(f"[SAVE][MULTI-SEED SUMMARY] {save_csv_path}")

    return raw_df, summary_df


def plot_comm_cost_to_target_multiseed(
    summary_df,
    target_acc=85.0,
    save_path=None,
    metric_label="Communication cost",
    xlabel=None,
):
    """
    multi-seed communication cost mean ± std bar plot.

    xlabel을 명시하면 기본 "{metric_label} to reach {target_acc}% accuracy" 문구
    대신 그 문자열을 그대로 x축 라벨로 사용한다(예: final_round 누적 통신비용 표시용).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    apply_paper_plot_style()

    plot_df = summary_df.copy()
    plot_df = plot_df.dropna(subset=["Mean"])

    if plot_df.empty:
        print("[WARN][MULTI-SEED PLOT] empty summary_df")
        return

    methods = plot_df["Method"].astype(str).values
    means = plot_df["Mean"].astype(float).values
    stds = plot_df["Std"].fillna(0.0).astype(float).values

    y = np.arange(len(methods))

    colors = [
        _paper_series_style(method, idx)["color"] for idx, method in enumerate(methods)
    ]

    fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor="white")
    ax.set_facecolor("white")

    ax.barh(
        y,
        means,
        xerr=stds,
        color=colors,
        edgecolor="black",
        linewidth=0.75,
        height=0.58,
        capsize=5,
        error_kw={
            "elinewidth": 1.2,
            "capthick": 1.2,
        },
    )

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=14)

    ax.set_xlabel(
        xlabel
        if xlabel is not None
        else f"{metric_label} to reach {target_acc:.0f}% accuracy",
        fontsize=14,
    )

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=14)

    xmax = max(means + stds) * 1.18
    ax.set_xlim(0, xmax)

    label_offset = xmax * 0.015

    for yi, mean, std in zip(y, means, stds):
        ax.text(
            mean + std + label_offset,
            yi,
            f"{mean:.0f}±{std:.1f}",
            va="center",
            ha="left",
            fontsize=12,
        )

    ax.grid(True, axis="x", linestyle=":", alpha=0.35)
    ax.grid(False, axis="y")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.invert_yaxis()

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[GRAPH][MULTI-SEED COMM TARGET] saved: {save_path}")
    else:
        plt.show()

    plt.close()


def make_accuracy_round_multiseed(
    log_dir,
    seeds,
    method_prefix_map,
    labels,
    num_rounds,
    save_csv_path=None,
    save_summary_csv_path=None,
):
    """
    여러 seed에 대해 라운드 기준 accuracy 곡선과 round-averaged accuracy를 계산.

    (a) per-round mean accuracy: 각 라운드 r(0..num_rounds)에서 seed별 accuracy.csv를
        라운드 그리드에 선형보간(np.interp, 마지막 관측치 이후는 flat 연장)한 뒤
        seed 축으로 평균/표준편차를 낸 값. accuracy-round 곡선을 그리는 데 쓰이는
        값과 동일하며, curve_df로 반환/저장된다.

    (b) round-averaged accuracy: 각 seed의 보간된 곡선을 라운드 축으로 평균낸 값
        (= area-under-curve / num_rounds). 방법별로 seed에 대해 다시
        mean/std/min/max를 낸 summary_df로 반환/저장된다.

    반환:
      curve_df:  [Method, round, mean_accuracy, std_accuracy] (round당 1행)
      summary_df: [Method, Mean, Std, Min, Max, Num_runs] (method당 1행,
                  round-averaged accuracy 기준)
    """
    import os
    import numpy as np
    import pandas as pd

    grid = np.arange(0, int(num_rounds) + 1, 1, dtype=float)

    curve_rows = []
    summary_rows = []

    for label in labels:
        prefix_template = method_prefix_map[label]
        seed_curves = []

        for idx, seed in enumerate(seeds, start=1):
            prefix = prefix_template.format(idx=idx, seed=seed)
            acc_path = os.path.join(log_dir, f"{prefix}_accuracy.csv")

            if not os.path.exists(acc_path):
                print(f"[WARN][ACC-ROUND] missing: {acc_path}")
                continue

            df = pd.read_csv(acc_path)
            if "round" not in df.columns or "accuracy" not in df.columns:
                print(f"[WARN][ACC-ROUND] invalid columns: {acc_path}")
                continue

            df = df.sort_values("round")
            r = df["round"].to_numpy(dtype=float)
            acc = df["accuracy"].to_numpy(dtype=float)

            uniq_r, uniq_idx = np.unique(r, return_index=True)
            uniq_acc = acc[uniq_idx]

            acc_interp = np.interp(
                grid,
                uniq_r,
                uniq_acc,
                left=uniq_acc[0],
                right=uniq_acc[-1],
            )
            seed_curves.append(acc_interp)

        if len(seed_curves) == 0:
            print(f"[WARN][ACC-ROUND] no runs found for method={label}")
            continue

        seed_curves = np.vstack(seed_curves)  # [n_seeds, n_rounds+1]

        mean_curve = seed_curves.mean(axis=0)
        std_curve = seed_curves.std(axis=0)

        for r_val, m_val, s_val in zip(grid, mean_curve, std_curve):
            curve_rows.append(
                {
                    "Method": label,
                    "round": float(r_val),
                    "mean_accuracy": float(m_val),
                    "std_accuracy": float(s_val),
                }
            )

        per_seed_round_avg = seed_curves.mean(
            axis=1
        )  # [n_seeds] round-averaged accuracy

        summary_rows.append(
            {
                "Method": label,
                "Mean": float(np.mean(per_seed_round_avg)),
                "Std": float(np.std(per_seed_round_avg, ddof=1))
                if len(per_seed_round_avg) > 1
                else 0.0,
                "Min": float(np.min(per_seed_round_avg)),
                "Max": float(np.max(per_seed_round_avg)),
                "Num_runs": int(len(per_seed_round_avg)),
            }
        )

    curve_df = pd.DataFrame(curve_rows)
    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:
        summary_df["Method"] = pd.Categorical(
            summary_df["Method"], categories=labels, ordered=True
        )
        summary_df = summary_df.sort_values("Method").reset_index(drop=True)

    if save_csv_path is not None and not curve_df.empty:
        curve_df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE][ACC-ROUND CURVE] {save_csv_path}")

    if save_summary_csv_path is not None and not summary_df.empty:
        summary_df.to_csv(save_summary_csv_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE][ACC-ROUND SUMMARY] {save_summary_csv_path}")

    return curve_df, summary_df


def plot_accuracy_round_multiseed(
    curve_df,
    labels,
    num_rounds,
    save_path=None,
):
    """
    make_accuracy_round_multiseed()가 만든 curve_df로 accuracy-round 곡선(mean ± std band)을 그린다.
    """
    import matplotlib.pyplot as plt

    apply_paper_plot_style()

    plt.figure(figsize=(8.5, 5.2))

    for idx, label in enumerate(labels):
        sub = curve_df[curve_df["Method"] == label].sort_values("round")
        if sub.empty:
            continue

        r = sub["round"].to_numpy(dtype=float)
        mean_acc = sub["mean_accuracy"].to_numpy(dtype=float)
        std_acc = sub["std_accuracy"].to_numpy(dtype=float)

        style = _paper_series_style(label, idx)

        (line,) = plt.plot(
            r,
            mean_acc,
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=3.0,
            marker=style["marker"],
            markevery=max(int(len(r) / 40), 1),
            markersize=6.5,
        )

        plt.fill_between(
            r,
            mean_acc - std_acc,
            mean_acc + std_acc,
            alpha=0.12,
            color=line.get_color(),
            linewidth=0,
        )

    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.xlim(0, num_rounds)
    plt.ylim(0, 100)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVE][ACC-ROUND] {save_path}")
    else:
        plt.show()

    plt.close()


def make_round_completion_multiseed(
    log_dir,
    seeds,
    method_prefix_map,
    labels,
    seconds_per_round=60,
    time_unit="min",
    save_csv_path=None,
):
    """
    Fig.1용 multi-seed round completion time 계산.

    각 run마다:
      accuracy.csv의 update 발생 round 간격을 계산
      그 run의 평균 round completion time을 1개 값으로 저장

    이후:
      method별 10개 run mean의 평균±표준편차 계산
    """
    import os
    import numpy as np
    import pandas as pd

    rows = []

    for label in labels:
        prefix_template = method_prefix_map[label]

        for idx, seed in enumerate(seeds, start=1):
            prefix = prefix_template.format(idx=idx, seed=seed)
            acc_csv = os.path.join(log_dir, f"{prefix}_accuracy.csv")

            if not os.path.exists(acc_csv):
                print(f"[WARN][ROUND-MULTI] missing accuracy csv: {acc_csv}")
                rows.append(
                    {
                        "Method": label,
                        "Seed": seed,
                        "Valid": False,
                        "Run mean": np.nan,
                        "Run std": np.nan,
                        "Num intervals": 0,
                    }
                )
                continue

            intervals = _get_update_completion_intervals_from_accuracy(
                acc_csv_path=acc_csv,
                seconds_per_round=seconds_per_round,
                time_unit=time_unit,
            )

            if len(intervals) == 0:
                print(f"[WARN][ROUND-MULTI] no intervals: {acc_csv}")
                rows.append(
                    {
                        "Method": label,
                        "Seed": seed,
                        "Valid": False,
                        "Run mean": np.nan,
                        "Run std": np.nan,
                        "Num intervals": 0,
                    }
                )
                continue

            rows.append(
                {
                    "Method": label,
                    "Seed": seed,
                    "Valid": True,
                    "Run mean": float(np.mean(intervals)),
                    "Run std": float(np.std(intervals, ddof=1))
                    if len(intervals) > 1
                    else 0.0,
                    "Num intervals": int(len(intervals)),
                }
            )

    raw_df = pd.DataFrame(rows)
    valid_df = raw_df[raw_df["Valid"] == True].copy()

    summary_df = valid_df.groupby("Method", as_index=False).agg(
        Mean=("Run mean", "mean"),
        Std=("Run mean", "std"),
        Min=("Run mean", "min"),
        Max=("Run mean", "max"),
        Num_runs=("Run mean", "count"),
    )

    summary_df["Method"] = pd.Categorical(
        summary_df["Method"],
        categories=labels,
        ordered=True,
    )
    summary_df = summary_df.sort_values("Method").reset_index(drop=True)

    expected_runs = len(seeds)
    for _, row in summary_df.iterrows():
        if int(row["Num_runs"]) < expected_runs:
            print(
                f"[WARN][ROUND-MULTI] {row['Method']} has "
                f"{int(row['Num_runs'])}/{expected_runs} valid runs only."
            )

    if save_csv_path is not None:
        raw_path = save_csv_path.replace(".csv", "_raw.csv")
        raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
        summary_df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE][ROUND-MULTI RAW] {raw_path}")
        print(f"[SAVE][ROUND-MULTI SUMMARY] {save_csv_path}")

    return raw_df, summary_df


def plot_round_completion_multiseed(
    summary_df,
    save_path=None,
    time_unit="min",
):
    """
    Fig.1용 multi-seed average round completion time plot.
    Error bars = std across run-level means.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    apply_paper_plot_style()

    plot_df = summary_df.dropna(subset=["Mean"]).copy()

    if plot_df.empty:
        print("[WARN][ROUND-MULTI PLOT] empty summary")
        return

    methods = plot_df["Method"].astype(str).values
    means = plot_df["Mean"].astype(float).values
    stds = plot_df["Std"].fillna(0.0).astype(float).values

    x = np.arange(len(methods))
    colors = [
        _paper_series_style(method, idx)["color"] for idx, method in enumerate(methods)
    ]

    fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor="white")
    ax.set_facecolor("white")

    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.75,
        width=0.62,
        error_kw={
            "elinewidth": 1.2,
            "capthick": 1.2,
        },
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=13)

    if time_unit == "sec":
        ylabel = "Average round completion time (sec)"
    elif time_unit == "hour":
        ylabel = "Average round completion time (hour)"
    else:
        ylabel = "Average round completion time (min)"

    ax.set_ylabel(ylabel, fontsize=14)

    ymax = max(means + stds) * 1.18
    ax.set_ylim(0, ymax)

    for xi, mean, std in zip(x, means, stds):
        ax.text(
            xi,
            mean + std + ymax * 0.02,
            f"{mean:.1f}±{std:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.grid(False, axis="x")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[GRAPH][ROUND-MULTI] saved: {save_path}")
    else:
        plt.show()

    plt.close()


def _build_accuracy_comm_components_dataframe(
    acc_csv_path,
    comm_csv_path,
):
    """
    accuracy.csv와 comm.csv를 결합하여
    target accuracy 도달 시점의 model_hops/offload_hops/total_hops를 분리 계산한다.

    반환 컬럼:
      - round
      - accuracy
      - update_round
      - cumulative_model_hops
      - cumulative_offload_hops
      - cumulative_total_hops
    """
    import os
    import numpy as np
    import pandas as pd

    if not os.path.exists(acc_csv_path):
        raise FileNotFoundError(f"accuracy csv not found: {acc_csv_path}")
    if not os.path.exists(comm_csv_path):
        raise FileNotFoundError(f"comm csv not found: {comm_csv_path}")

    acc = pd.read_csv(acc_csv_path).copy()
    comm = pd.read_csv(comm_csv_path).copy()

    if "round" not in acc.columns or "accuracy" not in acc.columns:
        raise ValueError(
            f"Invalid accuracy csv: {acc_csv_path}, columns={list(acc.columns)}"
        )

    # -----------------------------
    # accuracy 정리
    # -----------------------------
    acc["round"] = pd.to_numeric(acc["round"], errors="coerce")
    acc["accuracy"] = pd.to_numeric(acc["accuracy"], errors="coerce")

    acc = (
        acc.dropna(subset=["round", "accuracy"])
        .sort_values("round")
        .reset_index(drop=True)
    )

    # -----------------------------
    # comm의 round/time 컬럼 탐색
    # -----------------------------
    candidate_cols = [
        "round",
        "round_idx",
        "sim_round",
        "simulation_round",
        "time",
        "t",
        "step",
    ]

    comm_round_col = None
    for c in candidate_cols:
        if c in comm.columns:
            comm_round_col = c
            break

    if comm_round_col is None:
        print(
            f"[WARN] No round/time-like column found in {comm_csv_path}. "
            f"Use row index as communication event index. "
            f"columns={list(comm.columns)}"
        )
        comm["__round_for_merge__"] = np.arange(len(comm), dtype=float)
        comm_round_col = "__round_for_merge__"

    comm[comm_round_col] = pd.to_numeric(comm[comm_round_col], errors="coerce")

    # 없는 컬럼은 0으로 처리
    if "model_hops" not in comm.columns:
        comm["model_hops"] = 0

    if "offload_hops" not in comm.columns:
        comm["offload_hops"] = 0

    comm["model_hops"] = pd.to_numeric(
        comm["model_hops"],
        errors="coerce",
    ).fillna(0)

    comm["offload_hops"] = pd.to_numeric(
        comm["offload_hops"],
        errors="coerce",
    ).fillna(0)

    comm = (
        comm.dropna(subset=[comm_round_col])
        .sort_values(comm_round_col)
        .reset_index(drop=True)
    )

    # -----------------------------
    # round별 model/offload communication 분리 합산
    # -----------------------------
    comm_by_round = (
        comm.groupby(comm_round_col, as_index=False)[["model_hops", "offload_hops"]]
        .sum()
        .sort_values(comm_round_col)
        .reset_index(drop=True)
    )

    comm_by_round = comm_by_round.rename(columns={comm_round_col: "round"})
    comm_by_round["round"] = pd.to_numeric(comm_by_round["round"], errors="coerce")

    comm_by_round["cumulative_model_hops"] = comm_by_round["model_hops"].cumsum()
    comm_by_round["cumulative_offload_hops"] = comm_by_round["offload_hops"].cumsum()
    comm_by_round["cumulative_total_hops"] = (
        comm_by_round["cumulative_model_hops"]
        + comm_by_round["cumulative_offload_hops"]
    )

    # round=0 기록이 없는 baseline을 위해 0 row 보장
    zero = pd.DataFrame(
        {
            "round": [0.0],
            "cumulative_model_hops": [0.0],
            "cumulative_offload_hops": [0.0],
            "cumulative_total_hops": [0.0],
        }
    )

    comm_cum = pd.concat(
        [
            zero,
            comm_by_round[
                [
                    "round",
                    "cumulative_model_hops",
                    "cumulative_offload_hops",
                    "cumulative_total_hops",
                ]
            ],
        ],
        ignore_index=True,
    )

    # round=0에 실제 offload가 있으면 0 row와 충돌하므로 max로 보존
    comm_cum = (
        comm_cum.groupby("round", as_index=False)
        .max()
        .sort_values("round")
        .reset_index(drop=True)
    )

    acc["round"] = acc["round"].astype(float)
    comm_cum["round"] = comm_cum["round"].astype(float)

    out = pd.merge_asof(
        acc.sort_values("round"),
        comm_cum.sort_values("round"),
        on="round",
        direction="backward",
    )

    for c in [
        "cumulative_model_hops",
        "cumulative_offload_hops",
        "cumulative_total_hops",
    ]:
        out[c] = out[c].fillna(0.0)

    out["update_round"] = np.arange(len(out))

    return out


def make_comm_cost_components_to_target_multiseed(
    log_dir,
    seeds,
    method_prefix_map,
    labels,
    target_acc=85.0,
    save_csv_path=None,
    final_round=False,
):
    """
    Fig.4용 multi-seed communication cost component 계산.

    final_round=False (기본, 기존 동작):
      - target accuracy에 처음 도달한 시점 탐색
      - 그 시점까지 누적된 model_hops, offload_hops, total_hops를 분리 저장

    final_round=True:
      - target accuracy 도달 여부와 무관하게, comm.csv 전체(시뮬레이션 종료 시점,
        즉 num_rounds 시점)의 model_hops/offload_hops/total_hops 합계를 분리 저장

    summary:
      - Method별 model/offload/total mean
      - total std
    """
    import os
    import numpy as np
    import pandas as pd

    rows = []

    for label in labels:
        prefix_template = method_prefix_map[label]

        for idx, seed in enumerate(seeds, start=1):
            prefix = prefix_template.format(idx=idx, seed=seed)

            acc_csv = os.path.join(log_dir, f"{prefix}_accuracy.csv")
            comm_csv = os.path.join(log_dir, f"{prefix}_comm.csv")

            if final_round:
                if not os.path.exists(comm_csv):
                    print(f"[WARN][COMM-COMP-FINAL] missing comm csv: {prefix}")
                    rows.append(
                        {
                            "Method": label,
                            "Seed": seed,
                            "Reached": False,
                            "Model communication": np.nan,
                            "Offloading overhead": np.nan,
                            "Total communication": np.nan,
                            "Update round to target": np.nan,
                            "Simulation round to target": np.nan,
                            "Reached accuracy": np.nan,
                        }
                    )
                    continue

                comm = pd.read_csv(comm_csv).copy()
                if "model_hops" not in comm.columns:
                    comm["model_hops"] = 0
                if "offload_hops" not in comm.columns:
                    comm["offload_hops"] = 0
                model_comm = float(
                    pd.to_numeric(comm["model_hops"], errors="coerce").fillna(0).sum()
                )
                offload_comm = float(
                    pd.to_numeric(comm["offload_hops"], errors="coerce").fillna(0).sum()
                )
                total_comm = model_comm + offload_comm

                final_acc = np.nan
                if os.path.exists(acc_csv):
                    acc_df = pd.read_csv(acc_csv)
                    acc_df["round"] = pd.to_numeric(acc_df["round"], errors="coerce")
                    acc_df["accuracy"] = pd.to_numeric(
                        acc_df["accuracy"], errors="coerce"
                    )
                    acc_df = acc_df.dropna(subset=["round", "accuracy"]).sort_values(
                        "round"
                    )
                    if not acc_df.empty:
                        final_acc = float(acc_df["accuracy"].iloc[-1])

                rows.append(
                    {
                        "Method": label,
                        "Seed": seed,
                        "Reached": True,
                        "Model communication": model_comm,
                        "Offloading overhead": offload_comm,
                        "Total communication": total_comm,
                        "Update round to target": np.nan,
                        "Simulation round to target": np.nan,
                        "Reached accuracy": final_acc,
                    }
                )
                continue

            if not os.path.exists(acc_csv) or not os.path.exists(comm_csv):
                print(f"[WARN][COMM-COMP] missing files: {prefix}")
                rows.append(
                    {
                        "Method": label,
                        "Seed": seed,
                        "Reached": False,
                        "Model communication": np.nan,
                        "Offloading overhead": np.nan,
                        "Total communication": np.nan,
                        "Update round to target": np.nan,
                        "Simulation round to target": np.nan,
                        "Reached accuracy": np.nan,
                    }
                )
                continue

            df = _build_accuracy_comm_components_dataframe(
                acc_csv_path=acc_csv,
                comm_csv_path=comm_csv,
            )

            hit = df[df["accuracy"] >= target_acc]

            if hit.empty:
                print(
                    f"[WARN][COMM-COMP] {label}, seed={seed} did not reach {target_acc}%"
                )
                rows.append(
                    {
                        "Method": label,
                        "Seed": seed,
                        "Reached": False,
                        "Model communication": np.nan,
                        "Offloading overhead": np.nan,
                        "Total communication": np.nan,
                        "Update round to target": np.nan,
                        "Simulation round to target": np.nan,
                        "Reached accuracy": np.nan,
                    }
                )
                continue

            first = hit.iloc[0]

            model_comm = float(first["cumulative_model_hops"])
            offload_comm = float(first["cumulative_offload_hops"])
            total_comm = float(first["cumulative_total_hops"])

            rows.append(
                {
                    "Method": label,
                    "Seed": seed,
                    "Reached": True,
                    "Model communication": model_comm,
                    "Offloading overhead": offload_comm,
                    "Total communication": total_comm,
                    "Update round to target": int(first["update_round"]),
                    "Simulation round to target": float(first["round"]),
                    "Reached accuracy": float(first["accuracy"]),
                }
            )

    raw_df = pd.DataFrame(rows)
    reached_df = raw_df[raw_df["Reached"] == True].copy()

    summary_df = reached_df.groupby("Method", as_index=False).agg(
        Model_Mean=("Model communication", "mean"),
        Model_Std=("Model communication", "std"),
        Offload_Mean=("Offloading overhead", "mean"),
        Offload_Std=("Offloading overhead", "std"),
        Total_Mean=("Total communication", "mean"),
        Total_Std=("Total communication", "std"),
        Total_Min=("Total communication", "min"),
        Total_Max=("Total communication", "max"),
        Num_runs=("Total communication", "count"),
    )

    summary_df["Method"] = pd.Categorical(
        summary_df["Method"],
        categories=labels,
        ordered=True,
    )
    summary_df = summary_df.sort_values("Method").reset_index(drop=True)

    expected_runs = len(seeds)
    for _, row in summary_df.iterrows():
        if int(row["Num_runs"]) < expected_runs:
            print(
                f"[WARN][COMM-COMP] {row['Method']} reached target in "
                f"{int(row['Num_runs'])}/{expected_runs} runs only."
            )

    if save_csv_path is not None:
        raw_path = save_csv_path.replace(".csv", "_raw.csv")
        raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
        summary_df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE][COMM-COMP RAW] {raw_path}")
        print(f"[SAVE][COMM-COMP SUMMARY] {save_csv_path}")

    return raw_df, summary_df


def plot_comm_cost_components_to_target_multiseed(
    summary_df,
    target_acc=85.0,
    save_path=None,
    xlabel=None,
):
    """
    Fig.4용 stacked horizontal bar plot.

    Bar:
      model communication + one-shot offloading overhead

    Error bar:
      total communication cost의 std across runs

    xlabel을 명시하면 기본 "Communication cost to reach {target_acc}% accuracy"
    문구 대신 그 문자열을 그대로 x축 라벨로 사용한다(예: final_round 누적 통신비용 표시용).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    apply_paper_plot_style()

    plot_df = summary_df.dropna(subset=["Total_Mean"]).copy()

    if plot_df.empty:
        print("[WARN][COMM-COMP PLOT] empty summary_df")
        return

    methods = plot_df["Method"].astype(str).values
    model_means = plot_df["Model_Mean"].astype(float).values
    offload_means = plot_df["Offload_Mean"].fillna(0.0).astype(float).values
    total_means = plot_df["Total_Mean"].astype(float).values
    total_stds = plot_df["Total_Std"].fillna(0.0).astype(float).values

    y = np.arange(len(methods))

    method_colors = [
        _paper_series_style(method, idx)["color"] for idx, method in enumerate(methods)
    ]

    offload_color = "#F2A541"  # orange

    fig, ax = plt.subplots(figsize=(9.8, 5.4), facecolor="white")
    ax.set_facecolor("white")

    # 1) model communication
    ax.barh(
        y,
        model_means,
        color=method_colors,
        edgecolor="black",
        linewidth=0.75,
        height=0.58,
        label="Model communication",
    )

    # 2) offloading overhead
    ax.barh(
        y,
        offload_means,
        left=model_means,
        color=offload_color,
        edgecolor="black",
        linewidth=0.75,
        height=0.58,
        hatch="//",
        label="One-shot offloading overhead",
    )

    # 3) total std error bar
    ax.errorbar(
        total_means,
        y,
        xerr=total_stds,
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=5,
        capthick=1.2,
        zorder=5,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=14)

    ax.set_xlabel(
        xlabel
        if xlabel is not None
        else f"Communication cost to reach {target_acc:.0f}% accuracy",
        fontsize=14,
    )

    xmax = max(total_means + total_stds) * 1.30
    ax.set_xlim(0, xmax)

    label_offset = xmax * 0.022

    for yi, model_val, offload_val, total_val, std_val in zip(
        y,
        model_means,
        offload_means,
        total_means,
        total_stds,
    ):
        # total label
        ax.text(
            total_val + std_val + label_offset,
            yi,
            f"{total_val:.0f}±{std_val:.1f}",
            va="center",
            ha="left",
            fontsize=12,
        )

        if offload_val > 0:
            ax.text(
                model_val + offload_val / 2.0,
                yi - 0.28,  # y축 invert 전 기준. 그래프에서는 bar 위쪽으로 이동
                f"+{offload_val:.0f}",
                va="center",
                ha="center",
                fontsize=10.5,
                color="black",
                fontweight="bold",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                    pad=1.0,
                ),
                zorder=10,
            )

    ax.grid(True, axis="x", linestyle=":", alpha=0.35)
    ax.grid(False, axis="y")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.invert_yaxis()

    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
        fontsize=11,
    )

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[GRAPH][COMM-COMP PLOT] saved: {save_path}")
    else:
        plt.show()

    plt.close()
