"""
Clustering over L2-normalized row embeddings.

Two methods:
  - kmeans:  when you know roughly how many groups to expect
  - hdbscan: when you don't, or when densities vary

Both produce integer labels per row. hdbscan uses -1 for noise; downstream
code skips cluster pairs involving -1.
"""

import numpy as np
from sklearn.cluster import KMeans


def _kmeans(embeddings: np.ndarray, n_clusters: int, random_state: int) -> np.ndarray:
    """K-Means clustering on L2-normalized embeddings."""
    # Embeddings are L2-normalized, so Euclidean distance ranks the same as
    # cosine distance — k-means is optimizing the right thing.
    k = max(2, min(int(n_clusters), len(embeddings)))
    return KMeans(n_clusters=k, random_state=random_state, n_init=10)\
        .fit_predict(embeddings).astype(int)


def _hdbscan(embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """HDBSCAN clustering; marks noise points as -1."""
    import hdbscan as _h
    # L2-normalized vectors — Euclidean and cosine rank identically, so
    # Euclidean is safe and faster than cosine here.
    clusterer = _h.HDBSCAN(
        min_cluster_size=max(2, int(min_cluster_size)),
        metric="euclidean",
        core_dist_n_jobs=-1,
    )
    return clusterer.fit_predict(embeddings).astype(int)


def cluster(
    embeddings: np.ndarray,
    method: str,
    n_clusters: int | None = None,
    min_cluster_size: int | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """Cluster embeddings using the specified method, return integer labels."""
    if method == "kmeans":
        return _kmeans(embeddings, n_clusters or 5, random_state)
    if method == "hdbscan":
        return _hdbscan(embeddings, min_cluster_size or 5)
    raise ValueError(f"Unknown clustering method '{method}'. Use kmeans or hdbscan.")


def n_real_clusters(labels: np.ndarray) -> int:
    """Count non-noise clusters, excluding HDBSCAN's -1 label."""
    """Distinct cluster count, excluding hdbscan noise (-1)."""
    return int(np.sum(np.unique(labels) >= 0))


def distribution(labels: np.ndarray) -> str:
    """Format cluster size distribution as a readable string."""
    unique, counts = np.unique(labels, return_counts=True)
    return ", ".join(
        f"{'noise' if lbl < 0 else f'c{lbl}'}={cnt}"
        for lbl, cnt in zip(unique, counts)
    )
