"""Regime similarity search inspired by Faiss embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


@dataclass
class RegimeObservation:
    vector: np.ndarray
    label: str
    metadata: Dict[str, Any]


class FaissRegimeRetriever:
    """Lightweight similarity search with optional Faiss acceleration."""

    def __init__(self, max_history: int = 500, k: int = 5) -> None:
        self.max_history = max_history
        self.k = k
        self._history: List[RegimeObservation] = []
        self._faiss_index = None
        self._faiss_available = False
        self._maybe_import_faiss()

    def _maybe_import_faiss(self) -> None:
        try:  # pragma: no cover - optional dependency path
            import faiss

            self._faiss_available = True
            # Index dimension will be set on the first reference added.
            self._faiss_index = None
            self._faiss_index = faiss.IndexFlatL2(0)  # placeholder; dimension set on add
            logger.info("Faiss detected; regime retrieval will use Faiss indexes")
        except Exception:
            self._faiss_available = False
            self._faiss_index = None
            logger.info("Faiss not available; using numpy distance search")

    def add_reference(
        self, vector: List[float], label: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a new historical observation for retrieval."""

        arr = np.asarray(vector, dtype=float)
        obs = RegimeObservation(vector=arr, label=label, metadata=metadata or {})
        self._history.append(obs)
        if len(self._history) > self.max_history:
            self._history.pop(0)

        if self._faiss_available:
            try:  # pragma: no cover - optional dependency path
                import faiss

                dimension = arr.shape[0]
                if self._faiss_index is None:
                    self._faiss_index = faiss.IndexFlatL2(dimension)
                elif self._faiss_index.d != dimension:
                    # Rebuild index when dimension changes to avoid invalid state.
                    self._faiss_index = faiss.IndexFlatL2(dimension)
                    for existing in self._history:
                        self._faiss_index.add(existing.vector.reshape(1, -1))
                self._faiss_index.add(arr.reshape(1, -1))
            except Exception as exc:
                logger.error(f"Faiss add_reference failed: {exc}; disabling Faiss path")
                self._faiss_available = False
                self._faiss_index = None

    def query(self, vector: List[float]) -> Dict[str, Any]:
        """Return nearest regimes and similarity scores."""

        if not self._history:
            logger.warning("No history available for similarity search")
            return {"similarity": 0.0, "neighbors": []}

        query_vec = np.asarray(vector, dtype=float)
        distances: List[float] = []
        neighbors: List[RegimeObservation] = []

        if self._faiss_available and self._faiss_index is not None:
            try:  # pragma: no cover - optional dependency path
                import faiss

                if self._faiss_index.d != query_vec.shape[0]:
                    raise ValueError(
                        f"Query dimension {query_vec.shape[0]} does not match index {self._faiss_index.d}"
                    )

                dists, idx = self._faiss_index.search(
                    query_vec.reshape(1, -1), min(self.k, len(self._history))
                )
                dimension = query_vec.shape[0]
                index = faiss.IndexFlatL2(dimension)
                for obs in self._history:
                    index.add(obs.vector.reshape(1, -1))
                dists, idx = index.search(query_vec.reshape(1, -1), min(self.k, len(self._history)))
                for dist, i in zip(dists.flatten(), idx.flatten()):
                    neighbors.append(self._history[int(i)])
                    distances.append(float(dist))
            except Exception as exc:
                logger.error(f"Faiss search failed: {exc}; falling back to numpy")
                self._faiss_available = False
                self._faiss_index = None

        if not neighbors:
            for obs in self._history:
                dist = float(np.linalg.norm(query_vec - obs.vector))
                neighbors.append(obs)
                distances.append(dist)
            # sort by distance
            neighbors = [n for _, n in sorted(zip(distances, neighbors), key=lambda p: p[0])]
            distances = sorted(distances)
            neighbors = neighbors[: self.k]
            distances = distances[: self.k]

        similarity = float(1.0 / (1.0 + np.mean(distances))) if distances else 0.0
        formatted_neighbors = [
            {"label": n.label, "distance": float(d), "metadata": n.metadata}
            for n, d in zip(neighbors, distances)
        ]

        return {"similarity": similarity, "neighbors": formatted_neighbors}
