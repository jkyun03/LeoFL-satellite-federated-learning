# LeoFL/followup/data_iterators.py
"""
Per-satellite persistent shuffled data iteration.

The legacy ``fl_core.train_one_epoch`` creates a fresh
``DataLoader(dataset, shuffle=True)`` every time it is called, so every
cycle always starts from a freshly-shuffled *first* batch of the client's
data -- fine when every cycle trains exactly one full epoch (every sample is
touched once per epoch regardless of batch order), but wrong once different
cycles can request different, smaller numbers of *steps*: without a
persistent cursor, a satellite doing e.g. 3 steps/cycle would train on the
same leading slice of its shuffled data every single cycle.

``PersistentShuffledIterator`` keeps one shuffled permutation + cursor per
satellite across cycles, reshuffling only when it wraps past the end, and
separately tracks "unique samples seen at least once" vs. "total samples
processed" (spec section 7).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


class PersistentShuffledIterator:
    def __init__(self, n_samples: int, seed: int):
        self.n = int(n_samples)
        self._rng = np.random.default_rng(seed)
        self._perm = (
            self._rng.permutation(self.n) if self.n > 0 else np.zeros(0, dtype=int)
        )
        self._cursor = 0
        self.epochs_completed = 0
        self.total_samples_processed = 0
        self._seen = np.zeros(self.n, dtype=bool)

    @property
    def unique_samples_seen(self) -> int:
        return int(self._seen.sum())

    def next_indices(self, batch_size: int) -> np.ndarray:
        """Advance the cursor by up to ``batch_size`` indices, reshuffling
        on wraparound. Returns an index array (len < batch_size only if
        ``n_samples < batch_size``, matching ``drop_last=False``)."""
        if self.n == 0:
            return np.zeros(0, dtype=int)

        chunks = []
        remaining = min(
            batch_size, self.n
        )  # never form a batch bigger than the dataset
        while remaining > 0:
            if self._cursor >= self.n:
                self._perm = self._rng.permutation(self.n)
                self._cursor = 0
                self.epochs_completed += 1
            take = min(remaining, self.n - self._cursor)
            chunk = self._perm[self._cursor : self._cursor + take]
            chunks.append(chunk)
            self._cursor += take
            remaining -= take

        idx = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        self.total_samples_processed += len(idx)
        self._seen[idx] = True
        return idx


def fetch_batch(dataset, idx: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Index a client dataset (NIDDataset / InMemoryClientDataset /
    EmbedTabularDataset all expose plain ``.X``/``.Y`` tensors)."""
    if len(idx) == 0:
        return dataset.X[:0], dataset.Y[:0]
    idx_t = torch.from_numpy(idx).long()
    return dataset.X[idx_t], dataset.Y[idx_t]
