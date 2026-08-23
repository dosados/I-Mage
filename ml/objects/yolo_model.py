from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from ml.objects.base import Detection, ImageInput, ObjectsRetriever

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "artifacts" / "yolov8s.pt"
DEFAULT_MODEL_NAME = str(DEFAULT_MODEL_PATH)


class YoloObjectsRetriever(ObjectsRetriever):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.model = YOLO(model_name)

    def detect(self, image: ImageInput) -> list[Detection]:
        results = self.model.predict(
            source=self._to_source(image),
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue

            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        label=names[cls_id],
                        confidence=float(box.conf.item()),
                        bbox=(x1, y1, x2, y2),
                    )
                )

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections

    def detect_labels(self, image: ImageInput) -> list[str]:
        return sorted({detection.label for detection in self.detect(image)})

    def _to_source(self, image: ImageInput) -> str | Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return str(Path(image))
