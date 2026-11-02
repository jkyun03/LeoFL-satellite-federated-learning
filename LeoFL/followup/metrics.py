# LeoFL/followup/metrics.py
"""
Evaluation beyond the legacy ``fl_core.evaluate()`` (which returns only a
single overall-accuracy scalar): per-class recall/F1 (via sklearn, already a
project dependency), a standalone time-to-target lookup and its inverse
(accuracy at a fixed time budget -- neither exists today; the closest thing,
``analysis_utils.plot_comm_cost_to_target_bar``, computes "time to target" as
a byproduct of a communication-cost-centric plot), and a slow-satellite
subgroup metric. None of this modifies ``fl_core.py``/``analysis_utils.py``;
it is new, independent code per the task's module-separation requirement.

There is no single global model in this study (each orbit keeps its own
leader model, same as the existing LH-FL baselines). The "representative"
metric reported as THE accuracy/loss/F1 curve is the sample-size-weighted
average across orbit models -- the same convention already used by
``LeoFL/Simulator/scenarios.py``'s ``_evaluate_weighted_model_accuracy``,
reimplemented here independently. Per-orbit min/mean/max are logged
alongside it so the weighted number is never the only figure reported (spec
section 9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support


@dataclass
class EvalResult:
    accuracy: float  # percentage, 0-100 (matches fl_core.evaluate()'s convention)
    loss: float
    macro_f1: float
    per_class_recall: Dict[int, float] = field(default_factory=dict)
    per_class_f1: Dict[int, float] = field(default_factory=dict)
    per_class_support: Dict[int, int] = field(default_factory=dict)


@torch.no_grad()
def evaluate_detailed(model, dataset, device, batch_size: int = 64) -> EvalResult:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    all_preds: List[np.ndarray] = []
    all_true: List[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0

    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        logits = model(X)
        total_loss += float(criterion(logits, Y).item())
        n_batches += 1
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_true.append(Y.cpu().numpy())

    if n_batches == 0:
        return EvalResult(accuracy=0.0, loss=float("nan"), macro_f1=0.0)

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    return EvalResult(
        accuracy=float((y_pred == y_true).mean()) * 100.0,
        loss=total_loss / n_batches,
        macro_f1=float(np.mean(f1)) if len(f1) else 0.0,
        per_class_recall={int(c): float(r) for c, r in zip(labels, recall)},
        per_class_f1={int(c): float(f) for c, f in zip(labels, f1)},
        per_class_support={int(c): int(s) for c, s in zip(labels, support)},
    )


def _all_classes(results: Dict[int, EvalResult]) -> List[int]:
    classes = set()
    for r in results.values():
        classes |= set(r.per_class_recall.keys())
    return sorted(classes)


def weighted_representative_eval(
    orbit_models: Dict[int, nn.Module],
    orbit_weights: Dict[int, float],
    dataset,
    device,
    batch_size: int = 64,
) -> Tuple[EvalResult, Dict[int, EvalResult], Dict[str, float]]:
    """Evaluate every orbit's leader model, then combine via a sample-size-
    weighted average into one "representative" EvalResult. Returns
    (weighted, per_orbit_results, accuracy_spread={"min","mean","max"})."""
    per_orbit = {
        p: evaluate_detailed(m, dataset, device, batch_size)
        for p, m in orbit_models.items()
    }
    total_w = float(sum(orbit_weights[p] for p in per_orbit)) or 1.0

    def wavg(getter) -> float:
        return sum(orbit_weights[p] * getter(per_orbit[p]) for p in per_orbit) / total_w

    classes = _all_classes(per_orbit)
    weighted = EvalResult(
        accuracy=wavg(lambda r: r.accuracy),
        loss=wavg(lambda r: r.loss),
        macro_f1=wavg(lambda r: r.macro_f1),
        per_class_recall={
            c: wavg(lambda r, c=c: r.per_class_recall.get(c, 0.0)) for c in classes
        },
        per_class_f1={
            c: wavg(lambda r, c=c: r.per_class_f1.get(c, 0.0)) for c in classes
        },
    )
    accs = [r.accuracy for r in per_orbit.values()] or [0.0]
    spread = {
        "min": float(min(accs)),
        "mean": float(np.mean(accs)),
        "max": float(max(accs)),
    }
    return weighted, per_orbit, spread


def subgroup_recall(result: EvalResult, class_subset: Sequence[int]) -> float:
    vals = [
        result.per_class_recall[c] for c in class_subset if c in result.per_class_recall
    ]
    return float(np.mean(vals)) if vals else float("nan")


def subgroup_f1(result: EvalResult, class_subset: Sequence[int]) -> float:
    vals = [result.per_class_f1[c] for c in class_subset if c in result.per_class_f1]
    return float(np.mean(vals)) if vals else float("nan")


def slow_quartile_satellites(
    epoch_minutes_per_sat: Dict[int, float], frac: float = 0.25
) -> List[int]:
    """Satellite ids in the slowest ``frac`` fraction by drawn epoch_minutes
    (higher epoch_minutes == slower)."""
    items = sorted(epoch_minutes_per_sat.items(), key=lambda kv: kv[1], reverse=True)
    k = max(1, int(round(len(items) * frac)))
    return [sat_id for sat_id, _ in items[:k]]


def overrepresented_classes(
    client_datasets: Sequence,
    subset_ids: Sequence[int],
    all_ids: Sequence[int],
    n_classes: int,
    top_n: int = 3,
) -> List[int]:
    """Classes whose share of samples is highest among ``subset_ids``
    relative to their share among ``all_ids`` (used to evaluate performance
    on the data the slowest satellites disproportionately hold)."""

    def class_fractions(ids: Sequence[int]) -> np.ndarray:
        counts = np.zeros(n_classes, dtype=np.float64)
        for sid in ids:
            y = client_datasets[sid].Y.numpy().astype(int)
            counts += np.bincount(y, minlength=n_classes)[:n_classes]
        total = counts.sum()
        return counts / total if total > 0 else counts

    subset_frac = class_fractions(subset_ids)
    all_frac = class_fractions(all_ids)
    ratio = subset_frac / np.clip(all_frac, 1e-9, None)
    order = np.argsort(-ratio)
    return [int(c) for c in order[:top_n]]


def time_to_target(
    acc_log: Sequence[Tuple[float, float]], target: float
) -> Optional[float]:
    """First ``sim_time_s`` at which ``weighted_accuracy >= target``, or
    ``None`` ("미도달" / not reached) -- never silently drop not-reached runs
    from a multi-seed average; callers must check for ``None`` explicitly."""
    for t, acc in acc_log:
        if acc >= target:
            return float(t)
    return None


def accuracy_at_budget(
    acc_log: Sequence[Tuple[float, float]], budget_s: float
) -> Optional[float]:
    """Accuracy at the last checkpoint at or before ``budget_s`` (the
    inverse of ``time_to_target``, absent from the existing codebase).
    Assumes ``acc_log`` is sorted ascending by time. Returns ``None`` if no
    checkpoint exists at or before ``budget_s``."""
    best = None
    for t, acc in acc_log:
        if t <= budget_s:
            best = acc
        else:
            break
    return best
