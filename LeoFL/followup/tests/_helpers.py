# LeoFL/followup/tests/_helpers.py
"""Shared synthetic fixtures for followup tests. Deliberately avoid the real
EuroSAT dataset (download + cache) so tests run fast and offline; the
EmbedTabularDataset wrapper is reused as-is since it is exactly the plain
(X, Y) tensor container every scenario/engine already expects."""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from LeoFL.EuroSAT.eurosat_embed_dataset import EmbedTabularDataset
from LeoFL.Simulator.fl_core import make_walker_delta_constellation
from LeoFL.followup.topology import Topology, build_topology


def make_synthetic_client_datasets(
    num_clients: int,
    n_per_client: int,
    n_features: int = 8,
    n_classes: int = 3,
    seed: int = 0,
) -> List[EmbedTabularDataset]:
    rng = np.random.default_rng(seed)
    datasets = []
    for _ in range(num_clients):
        X = torch.from_numpy(
            rng.normal(size=(n_per_client, n_features)).astype(np.float32)
        )
        Y = torch.from_numpy(
            rng.integers(0, n_classes, size=n_per_client).astype(np.int64)
        )
        datasets.append(
            EmbedTabularDataset(X, Y, n_classes=n_classes, class_weight=None)
        )
    return datasets


def make_variable_size_client_datasets(
    sizes: List[int], n_features: int = 8, n_classes: int = 3, seed: int = 0
) -> List[EmbedTabularDataset]:
    """Like make_synthetic_client_datasets, but with a different, explicit
    sample count per client -- for tests that need deliberately skewed load
    (e.g. offload-balancing tests)."""
    rng = np.random.default_rng(seed)
    datasets = []
    for n in sizes:
        X = torch.from_numpy(rng.normal(size=(n, n_features)).astype(np.float32))
        Y = torch.from_numpy(rng.integers(0, n_classes, size=n).astype(np.int64))
        datasets.append(
            EmbedTabularDataset(X, Y, n_classes=n_classes, class_weight=None)
        )
    return datasets


def make_synthetic_test_dataset(
    n: int = 200, n_features: int = 8, n_classes: int = 3, seed: int = 999
) -> EmbedTabularDataset:
    rng = np.random.default_rng(seed)
    X = torch.from_numpy(rng.normal(size=(n, n_features)).astype(np.float32))
    Y = torch.from_numpy(rng.integers(0, n_classes, size=n).astype(np.int64))
    return EmbedTabularDataset(X, Y, n_classes=n_classes, class_weight=None)


def make_tiny_topology(
    num_planes: int = 2,
    sats_per_plane: int = 2,
    horizon_s: float = 5000.0,
    leader_top_k: int = 1,
    max_dist_km: float = 20000.0,
) -> Topology:
    configs = make_walker_delta_constellation(
        num_planes=num_planes, sats_per_plane=sats_per_plane, h_km=552.0, incl_deg=53.0
    )
    return build_topology(
        configs,
        horizon_s=horizon_s,
        leader_top_k=leader_top_k,
        leader_select_max_dist_km=max_dist_km,
    )
