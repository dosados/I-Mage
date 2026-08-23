import uuid

POINT_NAMESPACE = uuid.UUID("8f4e2a1b-6c3d-4e5f-9a0b-1c2d3e4f5a6b")


def context_point_id(image_id: str, model_version: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"context:{image_id}:{model_version}"))


def face_point_id(face_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"face-point:{face_id}"))
