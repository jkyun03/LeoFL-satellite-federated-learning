# LeoFL/followup/make_report.py
"""
Generates the 10-seed comparison report (figures + tables) from EXISTING logs
under logs_followup/ -- no retraining, reads only. Output: LeoFL/followup/report/.

Colors follow the dataviz-skill categorical palette (references/palette.md),
validated for this 3-hue + neutral-gray set via scripts/validate_palette.js
(PASS, with a CVD WARN on red<->aqua at the 6-8 floor and a contrast WARN on
aqua -- both mitigated here with secondary encoding: distinct markers/line
styles and direct value labels, never color alone):
  fixed            -> #595955 (neutral gray; baseline, not a competing hue)
  capacity         -> #2a78d6 (blue)
  capacity_minwork -> #1baf7a (aqua)
  offload          -> #e34948 (red)

The requested fraction trade-off chart was changed from a dual-y-axis line
chart to two side-by-side panels sharing one x-axis: accuracy and
slow-subgroup recall are different-scale measures, and overlaying two
y-scales on one plot is a well-documented way to manufacture a visual
correlation that isn't in the data. Two panels show the same two series
without that risk.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from LeoFL.Simulator.analysis_utils import apply_paper_plot_style

LOG_ROOT = "logs_followup"
OUT_DIR = "LeoFL/followup/report"
SEEDS = list(range(10))
TARGET_ACCURACY = 85.0

COLORS = {
    "fixed": "#595955",
    "capacity": "#2a78d6",
    "capacity_minwork": "#1baf7a",
    "offload": "#e34948",
}
MARKERS = {"fixed": "o", "capacity": "s", "capacity_minwork": "^", "offload": "D"}
LINESTYLES = {"fixed": "--", "capacity": "-", "capacity_minwork": "-", "offload": "-"}
DISPLAY = {
    "fixed": "Fixed",
    "capacity": "Proportional-Step",
    "capacity_minwork": "Proportional-Step + MinWork",
    "offload": "Offload",
}

# (log_dir, filename-method-token) for every config, keyed by a run label
CONFIG_DIRS = {
    "fixed": (f"{LOG_ROOT}/method_fixed", "fixed"),
    "capacity": (f"{LOG_ROOT}/method_capacity", "capacity"),
    "capacity_minwork_0.1": (
        f"{LOG_ROOT}/method_capacity_minwork_0.1",
        "capacity_minwork",
    ),
    "capacity_minwork_0.25": (
        f"{LOG_ROOT}/method_capacity_minwork_0.25",
        "capacity_minwork",
    ),
    "capacity_minwork_0.5": (
        f"{LOG_ROOT}/method_capacity_minwork_0.5",
        "capacity_minwork",
    ),
    "capacity_minwork_1.0": (
        f"{LOG_ROOT}/method_capacity_minwork_1.0",
        "capacity_minwork",
    ),
    "offload": (f"{LOG_ROOT}/method_offload", "offload"),
}
COLOR_KEY = {
    "fixed": "fixed",
    "capacity": "capacity",
    "offload": "offload",
    "capacity_minwork_0.1": "capacity_minwork",
    "capacity_minwork_0.25": "capacity_minwork",
    "capacity_minwork_0.5": "capacity_minwork",
    "capacity_minwork_1.0": "capacity_minwork",
}


def _load_accuracy(label: str, seed: int) -> pd.DataFrame:
    d, m = CONFIG_DIRS[label]
    return pd.read_csv(f"{d}/followup_{m}_{seed}_accuracy.csv").sort_values(
        "sim_time_s"
    )


def _load_comm(label: str, seed: int) -> pd.DataFrame:
    d, m = CONFIG_DIRS[label]
    return pd.read_csv(f"{d}/followup_{m}_{seed}_comm.csv")


def multiseed_curve(label: str, grid: np.ndarray):
    vals = [
        np.interp(
            grid,
            _load_accuracy(label, s)["sim_time_s"].values,
            _load_accuracy(label, s)["weighted_accuracy"].values,
        )
        for s in SEEDS
    ]
    arr = np.array(vals)
    return arr.mean(axis=0), arr.std(axis=0)


def final_stats(label: str):
    accs, slow_r, reached = [], [], 0
    for seed in SEEDS:
        df = _load_accuracy(label, seed)
        accs.append(df["weighted_accuracy"].iloc[-1])
        slow_r.append(df["slow_subgroup_recall"].iloc[-1])
        if (df["weighted_accuracy"] >= TARGET_ACCURACY).any():
            reached += 1
    return np.array(accs), np.array(slow_r), reached


def comm_split_stats(label: str):
    """Returns (model_exchange_MB, one_shot_offload_MB) per-seed arrays."""
    model_vals, off_vals = [], []
    for seed in SEEDS:
        c = _load_comm(label, seed)
        off = c.loc[c["kind"] == "data_offload", "bytes"].sum()
        total = c["bytes"].sum()
        model_vals.append((total - off) / 1024**2)
        off_vals.append(off / 1024**2)
    return np.array(model_vals), np.array(off_vals)


def _style_axes(ax):
    ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def fig1_accuracy_vs_time():
    apply_paper_plot_style()
    fig, ax = plt.subplots(figsize=(4, 3.2))
    grid = np.linspace(0, 60000, 121)
    short_label = {
        "fixed": "Fixed",
        "capacity": "Prop-Step",
        "capacity_minwork": "Prop-Step+MinWork",
        "offload": "Offload",
    }
    for label in ["fixed", "capacity", "capacity_minwork_1.0", "offload"]:
        ck = COLOR_KEY[label]
        mean, std = multiseed_curve(label, grid)
        c = COLORS[ck]
        ax.plot(
            grid,
            mean,
            color=c,
            linestyle=LINESTYLES[ck],
            linewidth=2.2,
            marker=MARKERS[ck],
            markevery=12,
            markersize=7,
            label=short_label[ck],
        )
        ax.fill_between(grid, mean - std, mean + std, color=c, alpha=0.15, linewidth=0)
    ax.axhline(TARGET_ACCURACY, color="#898781", linestyle=":", linewidth=1.3)
    ax.text(
        grid[-1],
        TARGET_ACCURACY + 0.6,
        "target 85%",
        ha="right",
        va="bottom",
        color="#52514e",
        fontsize=10,
    )
    ax.set_xlabel("Simulated time (s)", fontsize=15)
    ax.set_ylabel("Accuracy (%)", fontsize=15)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
        fontsize=9,
        frameon=True,
        handlelength=1.3,
        columnspacing=0.8,
        labelspacing=0.3,
    )
    _style_axes(ax)
    fig.tight_layout()
    path = f"{OUT_DIR}/fig1_accuracy_vs_time.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig2_final_accuracy_bar():
    apply_paper_plot_style()
    order = [
        "fixed",
        "capacity",
        "capacity_minwork_0.1",
        "capacity_minwork_0.25",
        "capacity_minwork_0.5",
        "capacity_minwork_1.0",
        "offload",
    ]
    display = [
        "Fixed",
        "Prop-Step",
        "MinWork\n(0.1)",
        "MinWork\n(0.25)",
        "MinWork\n(0.5)",
        "MinWork\n(1.0)",
        "Offload",
    ]
    means, stds = [], []
    for label in order:
        accs, _, _ = final_stats(label)
        means.append(accs.mean())
        stds.append(accs.std())
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(order))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        width=0.6,
        color=[COLORS[COLOR_KEY[l]] for l in order],
        error_kw=dict(ecolor="#52514e", elinewidth=1.3),
    )
    for xi, m, s in zip(x, means, stds):
        ax.text(
            xi,
            m + s + 0.8,
            f"{m:.1f}",
            ha="center",
            va="bottom",
            color="#0b0b0b",
            fontsize=11,
        )
    ax.axhline(TARGET_ACCURACY, color="#898781", linestyle=":", linewidth=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels(display)
    ax.set_ylabel("Final weighted accuracy (%) at 60,000s")
    ax.set_title("Final accuracy by method (mean ± std, 10 seeds)")
    _style_axes(ax)
    fig.text(
        0.99,
        0.01,
        "Prop-Step = Proportional-Step",
        fontsize=8.5,
        color="#898781",
        ha="right",
    )
    fig.tight_layout()
    path = f"{OUT_DIR}/fig2_final_accuracy_bar.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig3_comm_volume_bar():
    apply_paper_plot_style()
    labels = ["fixed", "capacity", "capacity_minwork_1.0", "offload"]
    display = ["Fixed", "Prop-Step", "MinWork", "Offload"]
    model_means, off_means = [], []
    for label in labels:
        model_vals, off_vals = comm_split_stats(label)
        model_means.append(model_vals.mean())
        off_means.append(off_vals.mean())
    fig, ax = plt.subplots(figsize=(4, 3.2))
    x = np.arange(len(labels))
    colors = [COLORS[COLOR_KEY[l]] for l in labels]
    ax.bar(x, model_means, color=colors, width=0.55, label="Model exchange")
    ax.bar(
        x,
        off_means,
        bottom=model_means,
        color="#0b0b0b",
        alpha=0.35,
        width=0.55,
        label="One-shot data move",
    )
    for xi, mm, om in zip(x, model_means, off_means):
        ax.text(
            xi,
            mm + om + 1.0,
            f"{mm + om:.1f}",
            ha="center",
            va="bottom",
            color="#0b0b0b",
            fontsize=12,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=30, ha="right")
    ax.set_ylabel("Comm. volume (MB)", fontsize=15)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.4),
        ncol=1,
        fontsize=9,
        frameon=True,
        handlelength=1.3,
        labelspacing=0.3,
    )
    _style_axes(ax)
    path = f"{OUT_DIR}/fig3_comm_volume_bar.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig4_fraction_tradeoff():
    apply_paper_plot_style()
    fractions = [0.1, 0.25, 0.5, 1.0]
    labels = [f"capacity_minwork_{f}" for f in fractions]
    acc_m, acc_s, slow_m, slow_s = [], [], [], []
    for label in labels:
        accs, slow_r, _ = final_stats(label)
        acc_m.append(accs.mean())
        acc_s.append(accs.std())
        slow_m.append(slow_r.mean())
        slow_s.append(slow_r.std())
    cap_accs, cap_slow, _ = final_stats("capacity")
    c = COLORS["capacity_minwork"]
    cap_c = COLORS["capacity"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    ax1.errorbar(
        fractions,
        acc_m,
        yerr=acc_s,
        color=c,
        marker=MARKERS["capacity_minwork"],
        markersize=8,
        linewidth=2,
        capsize=4,
        label="Prop-Step+MinWork",
    )
    ax1.axhline(
        cap_accs.mean(),
        color=cap_c,
        linestyle="--",
        linewidth=1.6,
        label="Prop-Step (no floor)",
    )
    ax1.set_xlabel("minimum_epoch_fraction")
    ax1.set_ylabel("Final weighted accuracy (%)")
    ax1.set_title("Accuracy")
    ax1.set_xticks(fractions)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.text(
        0.04,
        0.96,
        "0.1, 0.25 ≈ no floor effect\n(steps_per_epoch too small\nfor the floor to bind)",
        transform=ax1.transAxes,
        fontsize=9.5,
        color="#52514e",
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fcfcfb", edgecolor="#e1e0d9"),
    )

    ax2.errorbar(
        fractions,
        slow_m,
        yerr=slow_s,
        color=c,
        marker=MARKERS["capacity_minwork"],
        markersize=8,
        linewidth=2,
        capsize=4,
        label="Prop-Step+MinWork",
    )
    ax2.axhline(
        cap_slow.mean(),
        color=cap_c,
        linestyle="--",
        linewidth=1.6,
        label="Prop-Step (no floor)",
    )
    ax2.set_xlabel("minimum_epoch_fraction")
    ax2.set_ylabel("Slow-subgroup recall")
    ax2.set_title("Slow-subgroup recall")
    ax2.set_xticks(fractions)
    ax2.legend(loc="lower right", fontsize=10)

    for ax in (ax1, ax2):
        _style_axes(ax)
    fig.suptitle("minimum_epoch_fraction trade-off (mean ± std, 10 seeds)")
    fig.text(
        0.99,
        0.01,
        "Prop-Step = Proportional-Step",
        fontsize=8.5,
        color="#898781",
        ha="right",
    )
    fig.tight_layout()
    path = f"{OUT_DIR}/fig4_fraction_tradeoff.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig5_slow_subgroup_bar():
    apply_paper_plot_style()
    labels = ["fixed", "capacity", "capacity_minwork_1.0", "offload"]
    display = ["Fixed", "Prop-Step", "MinWork", "Offload"]
    means, stds = [], []
    for label in labels:
        _, slow_r, _ = final_stats(label)
        means.append(slow_r.mean())
        stds.append(slow_r.std())
    fig, ax = plt.subplots(figsize=(4, 3.2))
    x = np.arange(len(labels))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        width=0.55,
        color=[COLORS[COLOR_KEY[l]] for l in labels],
        error_kw=dict(ecolor="#52514e", elinewidth=1.3),
    )
    for xi, m, s in zip(x, means, stds):
        ax.text(
            xi,
            m + s + 0.012,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            color="#0b0b0b",
            fontsize=12,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=30, ha="right")
    ax.set_ylabel("Slow-recall", fontsize=15)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_ylim(0, 1.0)
    _style_axes(ax)
    fig.tight_layout()
    path = f"{OUT_DIR}/fig5_slow_subgroup_bar.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# Shared figsize for the three title-less abstract-row figures below, so a
# one-line \includegraphics row has matching heights and width:height ratio.
ABSTRACT_FIGSIZE = (7, 4)


def fig_abstract_accuracy():
    """Full paper-quality accuracy-vs-time line (same styling as the original
    standalone report figure), title removed only, figsize unified with
    fig_abstract_fraction_accuracy/comm_volume for a one-line abstract row."""
    apply_paper_plot_style()
    fig, ax = plt.subplots(figsize=ABSTRACT_FIGSIZE)
    grid = np.linspace(0, 60000, 121)
    for label in ["fixed", "capacity", "capacity_minwork_1.0", "offload"]:
        ck = COLOR_KEY[label]
        mean, std = multiseed_curve(label, grid)
        c = COLORS[ck]
        lbl = (
            DISPLAY[ck]
            if ck != "capacity_minwork"
            else "Proportional-Step + MinWork (1.0)"
        )
        ax.plot(
            grid,
            mean,
            color=c,
            linestyle=LINESTYLES[ck],
            linewidth=2.2,
            marker=MARKERS[ck],
            markevery=12,
            markersize=7,
            label=lbl,
        )
        ax.fill_between(grid, mean - std, mean + std, color=c, alpha=0.15, linewidth=0)
    ax.axhline(TARGET_ACCURACY, color="#898781", linestyle=":", linewidth=1.3)
    ax.text(
        grid[-1],
        TARGET_ACCURACY + 0.6,
        "target 85%",
        ha="right",
        va="bottom",
        color="#52514e",
        fontsize=11,
    )
    ax.set_xlabel("Simulated time (s)")
    ax.set_ylabel("Weighted accuracy (%)")
    ax.legend(loc="lower right", frameon=True)
    _style_axes(ax)
    path = f"{OUT_DIR}/accuracy.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _draw_wavy_break(ax_top, ax_bottom):
    """Draws a small wavy (~) break mark at the seam between a stacked
    broken-y-axis pair, at both the left and right edges of the shared width.
    """
    d = 0.015
    xs = np.linspace(-d, d, 60)
    ys = 0.45 * d * np.sin(xs / d * 2.5 * np.pi)
    kwargs = dict(color="#0b0b0b", linewidth=1.3, clip_on=False, solid_capstyle="round")
    for x0 in (0.0, 1.0):
        ax_top.plot(x0 + xs, 0.0 + ys, transform=ax_top.transAxes, **kwargs)
        ax_bottom.plot(x0 + xs, 1.0 + ys, transform=ax_bottom.transAxes, **kwargs)


def fig_abstract_fraction_accuracy():
    """Accuracy-only panel of the minimum_epoch_fraction trade-off (the
    recall panel is dropped per this request), same styling/data as the
    original fig4 left panel (lines/markers/error bars/legend/annotation/
    fonts all unchanged), title removed, figsize unified with
    fig_abstract_accuracy. The y-axis is a broken axis: a thin lower strip
    anchors it at 0, a wavy (~) break mark omits the empty 2-78% range, and
    the top panel shows the actual 79-84%-ish data range at full detail."""
    apply_paper_plot_style()
    fractions = [0.1, 0.25, 0.5, 1.0]
    labels = [f"capacity_minwork_{f}" for f in fractions]
    acc_m, acc_s = [], []
    for label in labels:
        accs, _, _ = final_stats(label)
        acc_m.append(accs.mean())
        acc_s.append(accs.std())
    cap_accs, _, _ = final_stats("capacity")
    c = COLORS["capacity_minwork"]
    cap_c = COLORS["capacity"]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=ABSTRACT_FIGSIZE,
        sharex=True,
        gridspec_kw={"height_ratios": [7, 1], "hspace": 0.06},
    )

    for ax in (ax_top, ax_bottom):
        ax.errorbar(
            fractions,
            acc_m,
            yerr=acc_s,
            color=c,
            marker=MARKERS["capacity_minwork"],
            markersize=8,
            linewidth=2,
            capsize=4,
            label="Prop-Step+MinWork",
        )
        ax.axhline(
            cap_accs.mean(),
            color=cap_c,
            linestyle="--",
            linewidth=1.6,
            label="Prop-Step (no floor)",
        )

    data_min = min(min(a - s for a, s in zip(acc_m, acc_s)), cap_accs.mean())
    data_max = max(max(a + s for a, s in zip(acc_m, acc_s)), cap_accs.mean())
    pad = 0.08 * (data_max - data_min)
    ax_top.set_ylim(data_min - pad, data_max + pad)
    ax_bottom.set_ylim(0, 1.5)
    ax_bottom.set_yticks([0])

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bottom.tick_params(axis="x", top=False)
    _draw_wavy_break(ax_top, ax_bottom)

    ax_bottom.set_xlabel("minimum_epoch_fraction")
    ax_top.set_ylabel("Final weighted accuracy (%)")
    ax_bottom.set_xticks(fractions)
    ax_top.legend(loc="lower right", fontsize=10)
    ax_top.text(
        0.04,
        0.96,
        "0.1, 0.25 ≈ no floor effect\n(steps_per_epoch too small\nfor the floor to bind)",
        transform=ax_top.transAxes,
        fontsize=9.5,
        color="#52514e",
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fcfcfb", edgecolor="#e1e0d9"),
    )
    _style_axes(ax_top)
    _style_axes(ax_bottom)
    path = f"{OUT_DIR}/fraction_accuracy.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_abstract_comm_volume():
    """Full paper-quality communication-volume bar (same styling as the
    original standalone report figure), title removed only, figsize unified
    with fig_abstract_accuracy/fraction_accuracy."""
    apply_paper_plot_style()
    labels = ["fixed", "capacity", "capacity_minwork_1.0", "offload"]
    display = ["Fixed", "Prop-Step", "MinWork (1.0)", "Offload"]
    model_means, off_means = [], []
    for label in labels:
        model_vals, off_vals = comm_split_stats(label)
        model_means.append(model_vals.mean())
        off_means.append(off_vals.mean())
    fig, ax = plt.subplots(figsize=ABSTRACT_FIGSIZE)
    x = np.arange(len(labels))
    colors = [COLORS[COLOR_KEY[l]] for l in labels]
    ax.bar(x, model_means, color=colors, width=0.55, label="Model exchange (recurring)")
    ax.bar(
        x,
        off_means,
        bottom=model_means,
        color="#0b0b0b",
        alpha=0.35,
        width=0.55,
        label="One-shot data move (offload only)",
    )
    for xi, mm, om in zip(x, model_means, off_means):
        ax.text(
            xi,
            mm + om + 1.0,
            f"{mm + om:.1f}",
            ha="center",
            va="bottom",
            color="#0b0b0b",
            fontsize=11,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=30, ha="right")
    ax.set_ylabel("Comm. volume (MB)")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.5),
        ncol=1,
        frameon=True,
        fontsize=10,
    )
    _style_axes(ax)
    fig.text(
        0.99,
        -0.1,
        "Prop-Step = Proportional-Step",
        fontsize=8.5,
        color="#898781",
        ha="right",
    )
    path = f"{OUT_DIR}/comm_volume.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _render_table_png(
    df: pd.DataFrame,
    path: str,
    title: str,
    footnote: str = None,
    colwidths=None,
    min_width=10,
    row_scale=1.6,
):
    max_lines = max(str(v).count("\n") + 1 for v in df.values.ravel())
    fig_h = 0.55 * max_lines * (len(df) + 1) + (0.5 if footnote else 0.2)
    fig, ax = plt.subplots(figsize=(max(min_width, 1.5 * len(df.columns)), fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        colWidths=colwidths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, row_scale)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e1e0d9")
        if row == 0:
            cell.set_facecolor("#f0efec")
            cell.set_text_props(weight="bold", color="#0b0b0b")
        else:
            cell.set_facecolor("#fcfcfb")
    ax.set_title(title, fontsize=13, pad=14, color="#0b0b0b")
    if footnote:
        fig.text(0.02, 0.01, footnote, fontsize=9, color="#52514e", ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def table_summary():
    order = [
        "fixed",
        "capacity",
        "capacity_minwork_0.1",
        "capacity_minwork_0.25",
        "capacity_minwork_0.5",
        "capacity_minwork_1.0",
        "offload",
    ]
    display = [
        "Fixed",
        "Proportional-Step",
        "Proportional-Step +\nMinWork (0.1)",
        "Proportional-Step +\nMinWork (0.25)",
        "Proportional-Step +\nMinWork (0.5)",
        "Proportional-Step +\nMinWork (1.0)",
        "Offload",
    ]
    rows = []
    for label, disp in zip(order, display):
        accs, slow_r, reached = final_stats(label)
        model_vals, off_vals = comm_split_stats(label)
        comm_total = model_vals + off_vals
        rows.append(
            {
                "Method": disp,
                "Final accuracy (%)": f"{accs.mean():.2f} ± {accs.std():.2f}",
                "Reached 85% (seeds)": f"{reached}/10",
                "Slow-subgroup recall": f"{slow_r.mean():.3f} ± {slow_r.std():.3f}",
                "Comm. volume (MB)": f"{comm_total.mean():.2f} ± {comm_total.std():.2f}",
            }
        )
    df = pd.DataFrame(rows)
    csv_path = f"{OUT_DIR}/table_summary_10seed.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    _render_table_png(
        df,
        f"{OUT_DIR}/table_summary_10seed.png",
        "10-seed summary (mean ± std across seeds 0-9; not-reached runs kept in the mean)",
        colwidths=[0.32, 0.16, 0.16, 0.18, 0.16],
        min_width=14.5,
        row_scale=1.7,
    )
    return csv_path, f"{OUT_DIR}/table_summary_10seed.png"


def table_fraction_comparison():
    fractions = [0.1, 0.25, 0.5, 1.0]
    rows = []
    for f in fractions:
        accs, slow_r, reached = final_stats(f"capacity_minwork_{f}")
        rows.append(
            {
                "minimum_epoch_fraction": f,
                "Final accuracy (%)": f"{accs.mean():.2f} ± {accs.std():.2f}",
                "Reached 85% (seeds)": f"{reached}/10",
                "Slow-subgroup recall": f"{slow_r.mean():.3f} ± {slow_r.std():.3f}",
            }
        )
    df = pd.DataFrame(rows)
    csv_path = f"{OUT_DIR}/table_fraction_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    footnote = (
        "* fraction=0.1 and 0.25: step allocation is bit-identical to plain "
        "Proportional-Step in 10/10 and 8/10 seeds respectively (median "
        "steps_per_epoch=3 at this scale, so ceil(fraction*steps_per_epoch)<=1, "
        "already satisfied by Proportional-Step's own >=1-step clamp) -- the floor "
        "is effectively a no-op below ~0.5 here, not a validated beneficial setting."
    )
    _render_table_png(
        df,
        f"{OUT_DIR}/table_fraction_comparison.png",
        "minimum_epoch_fraction comparison (Proportional-Step + MinWork, mean ± std, 10 seeds)",
        footnote=footnote,
    )
    return csv_path, f"{OUT_DIR}/table_fraction_comparison.png"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = []
    paths.append(fig1_accuracy_vs_time())
    paths.append(fig2_final_accuracy_bar())
    paths.append(fig3_comm_volume_bar())
    paths.append(fig4_fraction_tradeoff())
    paths.append(fig5_slow_subgroup_bar())
    paths.append(fig_abstract_accuracy())
    paths.append(fig_abstract_fraction_accuracy())
    paths.append(fig_abstract_comm_volume())
    paths.extend(table_summary())
    paths.extend(table_fraction_comparison())
    print("Generated:")
    for p in paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
