from __future__ import annotations

from db.database import Database
from db.models import IndexRun


class TestIndexRuns:
    def test_create_and_get(self, db: Database) -> None:
        with db:
            run = db.index_runs.create(module="clip", mode="full", progress_total=100)
            loaded = db.index_runs.get(run.id)
        assert loaded is not None
        assert loaded.module == "clip"
        assert loaded.mode == "full"
        assert loaded.status == "running"
        assert loaded.phase == "pending"
        assert loaded.progress_total == 100
        assert loaded.progress_done == 0

    def test_phase_and_progress_updates(self, db: Database) -> None:
        with db:
            run = db.index_runs.create(
                module="faces", mode="full", progress_total=0, phase="pending"
            )
            db.index_runs.set_phase(
                run.id, "reconcile", progress_done=0, progress_total=50
            )
            db.index_runs.update_progress(run.id, progress_done=10)
            updated = db.index_runs.get(run.id)
        assert updated is not None
        assert updated.phase == "reconcile"
        assert updated.progress_total == 50
        assert updated.progress_done == 10

    def test_get_active_filters_by_module(self, db: Database) -> None:
        with db:
            clip = db.index_runs.create(module="clip", mode="full", progress_total=1)
            yolo = db.index_runs.create(module="yolo", mode="full", progress_total=1)
            assert db.index_runs.get_active(module="clip").id == clip.id
            assert db.index_runs.get_active(module="yolo").id == yolo.id
            active_any = db.index_runs.get_active()
            assert active_any is not None
            assert active_any.id in {clip.id, yolo.id}

    def test_mark_done_and_failed(self, db: Database) -> None:
        with db:
            ok = db.index_runs.create(module="clip", mode="full", progress_total=1)
            bad = db.index_runs.create(module="yolo", mode="full", progress_total=1)
            db.index_runs.mark_done(ok.id, summary="partial ok")
            db.index_runs.mark_failed(bad.id, "disk full")
            assert db.index_runs.get_active(module="clip") is None
            assert db.index_runs.get_active(module="yolo") is None
            done = db.index_runs.get(ok.id)
            failed = db.index_runs.get(bad.id)
        assert done is not None and done.status == "done" and done.phase == "done"
        assert done.last_error == "partial ok"
        assert failed is not None and failed.status == "failed"
        assert failed.last_error == "disk full"

    def test_get_latest_per_module(self, db: Database) -> None:
        with db:
            first = db.index_runs.create(module="clip", mode="full", progress_total=1)
            db.index_runs.mark_done(first.id)
            second = db.index_runs.create(module="clip", mode="gap", progress_total=1)
            # Force a later timestamp: utc_now_iso() truncates microseconds, so
            # two creates in the same second would otherwise tie.
            row = db.session.get(IndexRun, second.id)
            assert row is not None
            row.started_at = "2099-01-01T00:00:00+00:00"
            db.session.flush()
            latest = db.index_runs.get_latest(module="clip")
        assert latest is not None
        assert latest.id == second.id

    def test_fail_stale_runs(self, db: Database) -> None:
        with db:
            a = db.index_runs.create(module="clip", mode="full", progress_total=1)
            b = db.index_runs.create(module="yolo", mode="full", progress_total=1)
            db.index_runs.mark_done(b.id)
            n = db.index_runs.fail_stale_runs()
            assert n == 1
            stale = db.index_runs.get(a.id)
        assert stale is not None
        assert stale.status == "failed"
        assert "interrupted" in (stale.last_error or "")

    def test_update_missing_run_returns_none(self, db: Database) -> None:
        with db:
            assert db.index_runs.update_progress("missing", progress_done=1) is None
            assert db.index_runs.set_phase("missing", "indexing") is None
            assert db.index_runs.mark_done("missing") is None
            assert db.index_runs.mark_failed("missing", "x") is None
