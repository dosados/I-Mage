from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base
from db.types import (
    MODULE_CLIP,
    MODULE_FACES,
    MODULE_YOLO,
    DetectionRecord,
    FaceRecord,
    ImageRecord,
    ModuleIndex,
    ModuleStatus,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class SchemaMigration(Base):
    __tablename__ = "schema_migrations"

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    applied_at: Mapped[str] = mapped_column(String, nullable=False)


class Image(Base):
    __tablename__ = "images"
    __table_args__ = (Index("idx_images_content_hash", "content_hash"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    path: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    mtime: Mapped[float] = mapped_column(Float, nullable=False)
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False, default=utc_now_iso)
    updated_at: Mapped[str] = mapped_column(String, nullable=False, default=utc_now_iso)

    yolo_index: Mapped[ImageYolo | None] = relationship(
        back_populates="image",
        cascade="all, delete-orphan",
        uselist=False,
    )
    faces_index: Mapped[ImageFaces | None] = relationship(
        back_populates="image",
        cascade="all, delete-orphan",
        uselist=False,
    )
    clip_index: Mapped[ImageClip | None] = relationship(
        back_populates="image",
        cascade="all, delete-orphan",
        uselist=False,
    )
    detections: Mapped[list[Detection]] = relationship(
        back_populates="image",
        cascade="all, delete-orphan",
    )
    faces: Mapped[list[Face]] = relationship(
        back_populates="image",
        cascade="all, delete-orphan",
    )

    def to_record(self) -> ImageRecord:
        from pathlib import Path

        modules: dict[str, ModuleIndex] = {}
        if self.yolo_index is not None:
            modules[MODULE_YOLO] = self.yolo_index.to_module_index()
        if self.faces_index is not None:
            modules[MODULE_FACES] = self.faces_index.to_module_index()
        if self.clip_index is not None:
            modules[MODULE_CLIP] = self.clip_index.to_module_index()

        return ImageRecord(
            id=self.id,
            path=Path(self.path),
            content_hash=self.content_hash,
            mtime=self.mtime,
            size=self.size,
            created_at=self.created_at,
            updated_at=self.updated_at,
            modules=modules,
        )


class ImageYolo(Base):
    __tablename__ = "image_yolo"
    __table_args__ = (Index("idx_image_yolo_status", "status"),)

    image_id: Mapped[str] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"),
        primary_key=True,
    )
    status: Mapped[str] = mapped_column(String, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False, default="")
    indexed_at: Mapped[str | None] = mapped_column(String, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    image: Mapped[Image] = relationship(back_populates="yolo_index")

    def to_module_index(self) -> ModuleIndex:
        return ModuleIndex(
            status=ModuleStatus(self.status),
            model_version=self.model_version,
            indexed_at=self.indexed_at,
            last_error=self.last_error,
        )


class ImageFaces(Base):
    __tablename__ = "image_faces"
    __table_args__ = (Index("idx_image_faces_status", "status"),)

    image_id: Mapped[str] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"),
        primary_key=True,
    )
    status: Mapped[str] = mapped_column(String, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False, default="")
    indexed_at: Mapped[str | None] = mapped_column(String, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    image: Mapped[Image] = relationship(back_populates="faces_index")

    def to_module_index(self) -> ModuleIndex:
        return ModuleIndex(
            status=ModuleStatus(self.status),
            model_version=self.model_version,
            indexed_at=self.indexed_at,
            last_error=self.last_error,
        )


class ImageClip(Base):
    __tablename__ = "image_clip"
    __table_args__ = (Index("idx_image_clip_status", "status"),)

    image_id: Mapped[str] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"),
        primary_key=True,
    )
    status: Mapped[str] = mapped_column(String, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False, default="")
    indexed_at: Mapped[str | None] = mapped_column(String, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    image: Mapped[Image] = relationship(back_populates="clip_index")

    def to_module_index(self) -> ModuleIndex:
        return ModuleIndex(
            status=ModuleStatus(self.status),
            model_version=self.model_version,
            indexed_at=self.indexed_at,
            last_error=self.last_error,
        )


class Detection(Base):
    __tablename__ = "detections"
    __table_args__ = (
        Index("idx_detections_image_id", "image_id"),
        Index("idx_detections_label", "label"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"), nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x2: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y2: Mapped[float] = mapped_column(Float, nullable=False)

    image: Mapped[Image] = relationship(back_populates="detections")

    def to_record(self) -> DetectionRecord:
        return DetectionRecord(
            id=self.id,
            image_id=self.image_id,
            label=self.label,
            confidence=self.confidence,
            bbox=(self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2),
        )


class Face(Base):
    __tablename__ = "faces"
    __table_args__ = (Index("idx_faces_image_id", "image_id"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"), nullable=False)
    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x2: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y2: Mapped[float] = mapped_column(Float, nullable=False)
    detection_score: Mapped[float] = mapped_column(Float, nullable=False)

    image: Mapped[Image] = relationship(back_populates="faces")

    def to_record(self) -> FaceRecord:
        return FaceRecord(
            id=self.id,
            image_id=self.image_id,
            bbox=(self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2),
            detection_score=self.detection_score,
        )


class AppSetting(Base):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)


class IndexRun(Base):
    __tablename__ = "index_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    module: Mapped[str] = mapped_column(String, nullable=False)
    mode: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    phase: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    progress_done: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[str] = mapped_column(String, nullable=False, default=utc_now_iso)
    finished_at: Mapped[str | None] = mapped_column(String, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    is_named: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False, default=utc_now_iso)
    updated_at: Mapped[str] = mapped_column(String, nullable=False, default=utc_now_iso)

    assignments: Mapped[list[FacePersonAssignment]] = relationship(
        back_populates="person",
        cascade="all, delete-orphan",
    )


class FacePersonAssignment(Base):
    __tablename__ = "face_person_assignments"
    __table_args__ = (Index("idx_face_person_assignments_person_id", "person_id"),)

    face_id: Mapped[str] = mapped_column(
        ForeignKey("faces.id", ondelete="CASCADE"),
        primary_key=True,
    )
    person_id: Mapped[str] = mapped_column(ForeignKey("persons.id", ondelete="CASCADE"), nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    assigned_at: Mapped[str] = mapped_column(String, nullable=False, default=utc_now_iso)

    person: Mapped[Person] = relationship(back_populates="assignments")
