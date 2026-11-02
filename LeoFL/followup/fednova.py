# LeoFL/followup/fednova.py
"""
FedNova intra-orbit aggregation (Wang et al., 2020, "Tackling the Objective
Inconsistency Problem in Heterogeneous Federated Optimization").

Scope (matches spec section 6 exactly): vanilla SGD, momentum=0, a single
fixed learning rate shared by every satellite and fixed during one local
training phase. Any other optimizer/momentum configuration raises
``NotImplementedError`` rather than silently producing an unvalidated
normalization.

Derivation (vanilla SGD, momentum=0):
  Each satellite i starts a cycle from the same leader snapshot ``w_start``
  and performs ``tau_i`` local SGD steps, ending at ``w_i``.
    Delta_i   = w_i - w_start                      (local model change)
    d_i       = Delta_i / tau_i                     (normalized direction;
                                                      average per-step update)
    p_i       = n_i / sum_j n_j                      (static data-size weight,
                                                      same convention as the
                                                      existing LH-FL FedAvg)
    tau_eff   = sum_i p_i * tau_i                    (FedNova's effective
                                                      step count)
    w_new     = w_start + tau_eff * sum_i p_i * d_i

  Validation (see tests/test_fednova_equals_fedavg_equal_steps.py): when
  every tau_i is equal to some tau, tau_eff == tau, and
    w_new = w_start + tau * sum_i p_i * (Delta_i / tau)
          = w_start + sum_i p_i * Delta_i
          = sum_i p_i * w_i
  i.e. exactly weighted FedAvg over the final client models. This is the
  textbook special case confirming the normalization is implemented
  correctly, not merely "looks reasonable".

Non-floating buffers (e.g. BatchNorm ``num_batches_tracked``) are carried
over unchanged from ``w_start``, matching the convention already used by
``LeoFL/Simulator/scenarios.py``'s ``_weighted_average_state_dict_from_models``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch

SUPPORTED_OPTIMIZER = "sgd"
SUPPORTED_MOMENTUM = 0.0


def _check_vanilla_sgd_scope(optimizer_name: str, momentum: float) -> None:
    if (
        optimizer_name.lower() != SUPPORTED_OPTIMIZER
        or float(momentum) != SUPPORTED_MOMENTUM
    ):
        raise NotImplementedError(
            "FedNova aggregation in this module is only implemented/validated for "
            f"vanilla SGD with momentum=0 (got optimizer={optimizer_name!r}, "
            f"momentum={momentum!r}). Supporting momentum or other optimizers requires "
            "a different per-solver normalization coefficient (see Wang et al. 2020, "
            "Sec. 4) that is not implemented here."
        )


def _normalize_weights(weights: Sequence[float]) -> List[float]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return [float(w) / total for w in weights]


def weighted_avg_aggregate(
    client_end_states: List[Dict[str, torch.Tensor]],
    weights: Sequence[float],
) -> Dict[str, torch.Tensor]:
    """Plain data-size-weighted average of final client models (no step-count
    normalization). Used as the explicit ablation that isolates "does the
    FedNova correction matter" from "does step scheduling matter" (spec
    section 4, C/D ablations)."""
    if len(client_end_states) == 0:
        raise ValueError("client_end_states must not be empty")
    p = _normalize_weights(weights)
    keys = client_end_states[0].keys()
    new_state: Dict[str, torch.Tensor] = {}
    for key in keys:
        ref = client_end_states[0][key]
        if torch.is_floating_point(ref):
            acc = torch.zeros_like(ref, dtype=torch.float32)
            for state, p_i in zip(client_end_states, p):
                acc += p_i * state[key].detach().float()
            new_state[key] = acc.to(dtype=ref.dtype)
        else:
            new_state[key] = ref.clone()
    return new_state


def fednova_aggregate(
    start_state: Dict[str, torch.Tensor],
    client_end_states: List[Dict[str, torch.Tensor]],
    weights: Sequence[float],
    client_steps: Sequence[int],
    optimizer_name: str = "sgd",
    momentum: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """FedNova intra-orbit aggregation. See module docstring for the exact
    formula and its FedAvg-equivalence special case."""
    _check_vanilla_sgd_scope(optimizer_name, momentum)

    if len(client_end_states) == 0:
        raise ValueError("client_end_states must not be empty")
    if len(client_end_states) != len(weights) or len(client_end_states) != len(
        client_steps
    ):
        raise ValueError(
            "client_end_states, weights, and client_steps must have the same length"
        )
    if any(tau < 1 for tau in client_steps):
        raise ValueError(
            "FedNova normalizes by local step count; every participating satellite "
            "must have performed at least 1 step (tau_i >= 1)."
        )

    p = _normalize_weights(weights)
    tau_eff = float(sum(p_i * tau_i for p_i, tau_i in zip(p, client_steps)))

    new_state: Dict[str, torch.Tensor] = {}
    for key, start_val in start_state.items():
        if torch.is_floating_point(start_val):
            start_f = start_val.detach().float()
            acc = torch.zeros_like(start_f)
            for end_state, p_i, tau_i in zip(client_end_states, p, client_steps):
                delta = end_state[key].detach().float() - start_f
                acc += p_i * (delta / float(tau_i))
            new_val = start_f + tau_eff * acc
            new_state[key] = new_val.to(dtype=start_val.dtype)
        else:
            new_state[key] = start_val.clone()
    return new_state


def convex_mix_states(
    self_state: Dict[str, torch.Tensor],
    other_state: Dict[str, torch.Tensor],
    beta: float,
) -> Dict[str, torch.Tensor]:
    """Pairwise leader-to-leader mixing: w_tilde = (1-beta)*self + beta*other.

    Re-implements (rather than imports) the same convex-combination rule
    LH-FL's ``OffloadScenario`` uses for inter-leader mixing
    (``scenarios.py:_mix_self_with_neighbor_average``), per spec section 6
    ("leader 간 혼합은 기존 LH-FL 방식을 유지한다"). Kept local to this module
    so the followup package has no training/aggregation dependency on
    ``LeoFL.Simulator.scenarios``.
    """
    beta = float(max(0.0, min(1.0, beta)))
    mixed: Dict[str, torch.Tensor] = {}
    for key, self_val in self_state.items():
        if torch.is_floating_point(self_val):
            mixed_val = (1.0 - beta) * self_val.detach().float() + beta * other_state[
                key
            ].detach().float()
            mixed[key] = mixed_val.to(dtype=self_val.dtype)
        else:
            mixed[key] = self_val.clone()
    return mixed
