# LeoFL/followup/plotting.py
"""
Comparison plots across followup-study runs. New-method runs read the
``followup_{method}_{seed}_{artifact}.csv`` files written by
``logging_schema.RunLogs``; the legacy offloading reference
(``legacy_reference.py``) writes the *existing* schema
(``{prefix}_accuracy.csv`` with columns ``round, accuracy``;
``{prefix}_comm.csv`` with columns
``time, model_bytes, offload_bytes, model_hops, offload_hops, note``) via the
unmodified ``scenarios.py`` code, so the comm-volume comparison reads both
schemas side by side. Reuses ``analysis_utils.apply_paper_plot_style`` for
visual consistency with the existing project's figures.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from LeoFL.Simulator.analysis_utils import apply_paper_plot_style

# (label, log_dir, seed, method) -- method is required to disambiguate runs
# that share one log_dir (e.g. comparing fixed/capacity/capacity_minwork
# from the same smoke run).
RunRef = Tuple[str, str, int, str]

# (label, log_dir, log_prefix) for a legacy_reference.py run.
LegacyRunRef = Tuple[str, str, str]


def _artifact_path(log_dir: str, seed: int, method: str, artifact: str) -> str:
    path = os.path.join(log_dir, f"followup_{method}_{seed}_{artifact}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def _load_accuracy(log_dir: str, seed: int, method: str) -> pd.DataFrame:
    return pd.read_csv(_artifact_path(log_dir, seed, method, "accuracy"))


def _load_steps(log_dir: str, seed: int, method: str) -> pd.DataFrame:
    return pd.read_csv(_artifact_path(log_dir, seed, method, "steps"))


def plot_accuracy_vs_time(
    runs: Sequence[RunRef],
    save_path: Optional[str] = None,
    show_spread: bool = True,
    target_accuracy: Optional[float] = None,
) -> None:
    """Weighted representative accuracy vs. simulated time for each run,
    with the per-orbit min/max band shaded (never report only the weighted
    number)."""
    apply_paper_plot_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, log_dir, seed, method in runs:
        df = _load_accuracy(log_dir, seed, method).sort_values("sim_time_s")
        ax.plot(df["sim_time_s"], df["weighted_accuracy"], label=label)
        if show_spread and "orbit_acc_min" in df.columns:
            ax.fill_between(
                df["sim_time_s"], df["orbit_acc_min"], df["orbit_acc_max"], alpha=0.15
            )
    if target_accuracy is not None:
        ax.axhline(
            target_accuracy,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label=f"target={target_accuracy}%",
        )
    ax.set_xlabel("Simulated time (s)")
    ax.set_ylabel("Weighted accuracy (%)")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)


def plot_time_to_target_bar(
    results: Dict[
        str, Optional[float]
    ],  # label -> time_to_target_s (None = not reached)
    target_accuracy: float,
    save_path: Optional[str] = None,
) -> None:
    """Bar chart of time-to-target per method. Not-reached runs are drawn as
    a hatched bar with an explicit "NOT REACHED" label rather than being
    dropped (never silently exclude 미도달 runs from a comparison)."""
    apply_paper_plot_style()
    labels = list(results.keys())
    values = [v if v is not None else 0.0 for v in results.values()]
    not_reached = [v is None for v in results.values()]

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(labels)), 5))
    bars = ax.bar(labels, values)
    for bar, nr in zip(bars, not_reached):
        if nr:
            bar.set_hatch("//")
            bar.set_facecolor("lightgray")
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.01,
                "NOT REACHED",
                ha="center",
                va="bottom",
                rotation=90,
            )
    ax.set_ylabel(f"Time to reach {target_accuracy}% accuracy (s)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)


def plot_step_distribution(
    runs: Sequence[RunRef], save_path: Optional[str] = None
) -> None:
    """Boxplot of per-satellite-per-cycle step counts, one box per method --
    direct evidence that capacity-aware allocation actually redistributes
    steps (fixed=constant; capacity=spread proportional to drawn speed;
    capacity_minwork=the slow tail floored up)."""
    apply_paper_plot_style()
    data, labels = [], []
    for label, log_dir, seed, method in runs:
        df = _load_steps(log_dir, seed, method)
        data.append(df["steps"].values)
        labels.append(label)
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(labels)), 5))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel("Local steps per satellite per cycle")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)


def _total_comm_bytes_followup(log_dir: str, seed: int, method: str) -> float:
    df = pd.read_csv(_artifact_path(log_dir, seed, method, "comm"))
    return float(df["bytes"].sum())


def _total_comm_bytes_legacy(log_dir: str, log_prefix: str) -> float:
    path = os.path.join(log_dir, f"{log_prefix}_comm.csv")
    df = pd.read_csv(path)
    model = float(df["model_bytes"].sum()) if "model_bytes" in df.columns else 0.0
    offload = float(df["offload_bytes"].sum()) if "offload_bytes" in df.columns else 0.0
    return model + offload


def plot_comm_volume_comparison(
    new_runs: Sequence[RunRef],
    legacy_runs: Sequence[LegacyRunRef] = (),
    save_path: Optional[str] = None,
) -> None:
    """Total communication volume (MB) per method, including the legacy
    offloading reference's one-shot data-redistribution bytes -- the direct
    evidence for "step throttling costs less communication than offloading"
    (new methods move model parameters only; the legacy reference also moves
    training samples)."""
    apply_paper_plot_style()
    labels, totals_mb = [], []
    for label, log_dir, seed, method in new_runs:
        labels.append(label)
        totals_mb.append(
            _total_comm_bytes_followup(log_dir, seed, method) / (1024.0**2)
        )
    for label, log_dir, log_prefix in legacy_runs:
        labels.append(label)
        totals_mb.append(_total_comm_bytes_legacy(log_dir, log_prefix) / (1024.0**2))

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(labels)), 5))
    ax.bar(labels, totals_mb)
    ax.set_ylabel("Total communication volume (MB)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)


def plot_minwork_fraction_tradeoff(
    fraction_runs: Sequence[
        Tuple[float, str, int, str]
    ],  # (minimum_epoch_fraction, log_dir, seed, method)
    save_path: Optional[str] = None,
) -> None:
    """x = minimum_epoch_fraction, final weighted accuracy vs. final
    slow-subgroup recall on twin y-axes -- the explicit "straggler 완화 vs
    학습 퀄리티" trade-off this study is designed to show. Does not assume
    the floor helps; the plot is just as meaningful if recall is flat or if
    accuracy degrades as the fraction grows."""
    apply_paper_plot_style()
    runs_sorted = sorted(fraction_runs, key=lambda x: x[0])
    fractions, final_acc, slow_recall = [], [], []
    for fraction, log_dir, seed, method in runs_sorted:
        df = _load_accuracy(log_dir, seed, method).sort_values("sim_time_s")
        fractions.append(fraction)
        final_acc.append(float(df["weighted_accuracy"].iloc[-1]))
        slow_recall.append(float(df["slow_subgroup_recall"].iloc[-1]))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    line1 = ax1.plot(
        fractions,
        final_acc,
        marker="o",
        color="tab:blue",
        label="Final weighted accuracy (%)",
    )
    ax1.set_xlabel("minimum_epoch_fraction")
    ax1.set_ylabel("Final weighted accuracy (%)", color="tab:blue")

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        fractions,
        slow_recall,
        marker="s",
        color="tab:orange",
        label="Slow-subgroup recall",
    )
    ax2.set_ylabel("Slow-subgroup recall", color="tab:orange")

    lines = line1 + line2
    ax1.legend(lines, [l.get_label() for l in lines], loc="best")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
