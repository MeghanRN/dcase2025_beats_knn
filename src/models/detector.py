"""
k‑NN anomaly detector operating in embedding space.
Stores all training embeddings and computes the *mean* cosine distance to the
k nearest neighbours of a test embedding.
"""

from typing import List
import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np


class KNNDetector:
    def __init__(self, k: int = 3):
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        self.memory: np.ndarray | None = None

    def fit(self, features: List[torch.Tensor]):
        # concat variable‑length sequences by mean‑pooling over time
        pooled = [f.mean(dim=0).cpu().numpy() for f in features]
        self.memory = np.stack(pooled, axis=0)
        self.nn.fit(self.memory)

    def score(self, features: torch.Tensor) -> float:
        """Return positive anomaly score (higher == more abnormal)."""
        if self.memory is None:
            raise RuntimeError("Detector has not been fitted!")

        pooled = features.mean(dim=0, keepdim=True).cpu().numpy()
        dists, _ = self.nn.kneighbors(pooled)
        return float(dists.mean())
