from dataclasses import dataclass
from typing import List, Tuple

import torch
from ultralytics import YOLO

from .config import settings


@dataclass
class Detection:
    label: str
    confidence: float
    box: Tuple[float, float, float, float]


class PPEDetector:
    """Wraps a YOLOv11 model fine-tuned for PPE compliance detection."""

    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or (settings.device if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path or settings.model_path)
        self.model.to(self.device)

    def detect(self, frame) -> List[Detection]:
        results = self.model.predict(
            frame,
            conf=settings.confidence_threshold,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            label = results.names[int(box.cls)]
            confidence = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection(label, confidence, (x1, y1, x2, y2)))
        return detections

    def missing_ppe(self, detections: List[Detection]):
        present = {d.label for d in detections}
        return set(settings.required_classes) - present
