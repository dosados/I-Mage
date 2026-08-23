import uuid

FACE_NAMESPACE = uuid.UUID("2b7c9e4a-1f30-4d8b-a6e5-3c9f0b2a8d71")


def make_face_id(
    image_id: str,
    bbox: tuple[float, float, float, float],
    *,
    model_version: str,
) -> str:
    bbox_key = ",".join(f"{value:.4f}" for value in bbox)
    return str(uuid.uuid5(FACE_NAMESPACE, f"face:{image_id}:{bbox_key}:{model_version}"))
