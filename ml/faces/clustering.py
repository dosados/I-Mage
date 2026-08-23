from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN


def cluster_faces(
    vectors: dict[str, np.ndarray],
    face_ids: list[str],
    *,
    threshold: float = 0.65,
) -> list[list[str]]:
    selected = [face_id for face_id in face_ids if face_id in vectors]
    if not selected:
        return []

    matrix = np.stack([vectors[face_id] for face_id in selected], axis=0)
    eps = max(1.0 - threshold, 1e-6)
    labels = DBSCAN(eps=eps, min_samples=1, metric="cosine").fit_predict(matrix)

    clusters: dict[int, list[str]] = {}
    for face_id, label in zip(selected, labels, strict=True):
        if label < 0:
            clusters.setdefault(-(hash(face_id) % 1_000_000), [face_id])
            continue
        clusters.setdefault(int(label), []).append(face_id)

    return list(clusters.values())
