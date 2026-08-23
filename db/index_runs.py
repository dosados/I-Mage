import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import IndexRun, utc_now_iso
from db.types import IndexRunRecord


class IndexRunService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self, *, module: str, mode: str, progress_total: int, phase: str = "pending"
    ) -> IndexRunRecord:
        row = IndexRun(
            id=str(uuid.uuid4()),
            module=module,
            mode=mode,
            status="running",
            phase=phase,
            progress_done=0,
            progress_total=progress_total,
        )
        self._session.add(row)
        self._session.flush()
        return self._to_record(row)

    def set_phase(
        self,
        run_id: str,
        phase: str,
        *,
        progress_done: int | None = None,
        progress_total: int | None = None,
    ) -> IndexRunRecord | None:
        row = self._session.get(IndexRun, run_id)
        if row is None:
            return None
        row.phase = phase
        if progress_done is not None:
            row.progress_done = progress_done
        if progress_total is not None:
            row.progress_total = progress_total
        self._session.flush()
        return self._to_record(row)

    def update_progress(self, run_id: str, *, progress_done: int) -> IndexRunRecord | None:
        row = self._session.get(IndexRun, run_id)
        if row is None:
            return None
        row.progress_done = progress_done
        self._session.flush()
        return self._to_record(row)

    def set_progress_total(self, run_id: str, progress_total: int) -> IndexRunRecord | None:
        row = self._session.get(IndexRun, run_id)
        if row is None:
            return None
        row.progress_total = progress_total
        self._session.flush()
        return self._to_record(row)

    def fail_stale_runs(self) -> int:
        rows = self._session.scalars(
            select(IndexRun).where(IndexRun.status == "running")
        ).all()
        for row in rows:
            row.status = "failed"
            row.finished_at = utc_now_iso()
            row.last_error = "interrupted by server restart"
        self._session.flush()
        return len(rows)

    def mark_done(self, run_id: str, *, summary: str | None = None) -> IndexRunRecord | None:
        row = self._session.get(IndexRun, run_id)
        if row is None:
            return None
        row.status = "done"
        row.phase = "done"
        row.finished_at = utc_now_iso()
        row.last_error = summary
        self._session.flush()
        return self._to_record(row)

    def mark_failed(self, run_id: str, error: str) -> IndexRunRecord | None:
        row = self._session.get(IndexRun, run_id)
        if row is None:
            return None
        row.status = "failed"
        row.finished_at = utc_now_iso()
        row.last_error = error
        self._session.flush()
        return self._to_record(row)

    def get(self, run_id: str) -> IndexRunRecord | None:
        row = self._session.get(IndexRun, run_id)
        return self._to_record(row) if row is not None else None

    def get_active(self, *, module: str | None = None) -> IndexRunRecord | None:
        # started_at is second-precision ISO; break ties with id so "latest"
        # is stable when two runs start in the same second.
        stmt = (
            select(IndexRun)
            .where(IndexRun.status == "running")
            .order_by(IndexRun.started_at.desc(), IndexRun.id.desc())
        )
        if module is not None:
            stmt = stmt.where(IndexRun.module == module)
        row = self._session.scalar(stmt)
        return self._to_record(row) if row is not None else None

    def get_latest(self, *, module: str | None = None) -> IndexRunRecord | None:
        stmt = select(IndexRun).order_by(IndexRun.started_at.desc(), IndexRun.id.desc())
        if module is not None:
            stmt = stmt.where(IndexRun.module == module)
        row = self._session.scalar(stmt.limit(1))
        return self._to_record(row) if row is not None else None

    @staticmethod
    def _to_record(row: IndexRun) -> IndexRunRecord:
        return IndexRunRecord(
            id=row.id,
            module=row.module,
            mode=row.mode,
            status=row.status,
            phase=row.phase or "indexing",
            progress_done=row.progress_done,
            progress_total=row.progress_total,
            started_at=row.started_at,
            finished_at=row.finished_at,
            last_error=row.last_error,
        )
