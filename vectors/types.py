from dataclasses import dataclass


@dataclass(frozen=True)
class ContextHit:
    image_id: str
    score: float


@dataclass(frozen=True)
class FaceHit:
    face_id: str
    image_id: str
    score: float
