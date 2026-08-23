from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable

from db.database import Database
from indexing.runner import ScanStopped, collect_scope_paths
from ml.faces.clustering import cluster_faces
from vectors.store import VectorStore

logger = logging.getLogger(__name__)

# Clustering has no natural per-item progress (DBSCAN is one blocking call), so
# we report a coarse 0..100 across its sub-steps. This maps onto the same
# progress_done/progress_total the UI already draws for the other phases.
_CLUSTER_TOTAL = 100

DEFAULT_THRESHOLD = 0.65
DEFAULT_MIN_DETECTION_SCORE = 0.7
# Singletons are noise for a "people" view: a person needs at least two faces to
# be worth surfacing. Leaving singletons unassigned keeps the persons table from
# flooding with millions of one-face rows.
MIN_CLUSTER_SIZE = 2


def run_face_clustering(
    db: Database,
    vector_store: VectorStore,
    *,
    run_id: str | None = None,
    regroup: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    min_detection_score: float = DEFAULT_MIN_DETECTION_SCORE,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Group in-scope faces into persons, reporting progress via ``run_id``.

    Vectors come from Qdrant (the single source of truth for embeddings); person
    assignments are written to SQLite. When ``regroup`` is set, existing *auto*
    (non-manual, non-named) clusters are rebuilt but their person IDs are kept
    stable by matching each new cluster back to the person most of its faces
    previously belonged to.
    """

    def _progress(done: int) -> None:
        if run_id is not None:
            with db:
                db.index_runs.set_phase(
                    run_id, "clustering", progress_done=done, progress_total=_CLUSTER_TOTAL
                )

    def _check_stop() -> None:
        if should_stop is not None and should_stop():
            raise ScanStopped("clustering aborted by shutdown")

    _progress(0)
    if not vector_store.available:
        raise RuntimeError("qdrant is not available")

    # Work out WHICH faces to cluster first (cheap DB reads), then fetch only
    # those vectors. Fetching all vectors every run made incremental runs pay the
    # full-catalog cost even when only a handful of faces were new.
    with db:
        config = db.get_scan_config()
        paths = collect_scope_paths(config)
        path_set = {path.resolve() for path in paths}
        face_ids = db.faces.list_ids_in_scope(path_set)

        prior: dict[str, str] = {}
        if regroup:
            eligible = db.persons.list_regroup_eligible_face_ids(face_ids)
            # Remember prior auto groupings so we can reuse person IDs (stable UI).
            prior = {
                assignment.face_id: assignment.person_id
                for assignment in db.persons.list_assignments_for_faces(eligible)
                if assignment.source not in db.persons.MANUAL_SOURCES
            }
            for face_id in eligible:
                db.persons.clear_auto_assignments([face_id])
            target_ids = eligible
        else:
            target_ids = db.persons.list_unassigned_face_ids(face_ids)

        if min_detection_score > 0:
            target_ids = db.faces.list_confident_ids(target_ids, min_detection_score)
    _check_stop()
    _progress(20)

    vectors = vector_store.scroll_face_vectors(target_ids)
    _check_stop()
    _progress(45)

    clusters = [
        cluster for cluster in cluster_faces(vectors, target_ids, threshold=threshold)
        if len(cluster) >= MIN_CLUSTER_SIZE
    ]
    _progress(60)

    with db:
        total = max(len(clusters), 1)
        for index, cluster_face_ids in enumerate(clusters):
            _check_stop()
            person_id = _stable_person_id(db, cluster_face_ids, prior)
            db.persons.assign_faces(person_id, cluster_face_ids, source="auto_cluster")
            if run_id is not None:
                done = 60 + int(35 * (index + 1) / total)
                db.index_runs.set_phase(
                    run_id, "clustering", progress_done=done, progress_total=_CLUSTER_TOTAL
                )
        db.persons.delete_empty_persons()

    _progress(_CLUSTER_TOTAL)
    logger.info("clustering done: %d groups (regroup=%s)", len(clusters), regroup)


def _stable_person_id(
    db: Database, cluster_face_ids: list[str], prior: dict[str, str]
) -> str:
    """Reuse the existing unnamed person most of this cluster came from, else new.

    Keeps auto-group IDs stable across regroups so the UI (and any external
    references) don't churn every time.
    """
    counts = Counter(
        prior[face_id] for face_id in cluster_face_ids if face_id in prior
    )
    for person_id, _count in counts.most_common():
        person = db.persons.get_person(person_id)
        if person is not None and not person.is_named:
            return person_id
    return db.persons.create_person().id
