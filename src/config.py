import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    model_path: str = os.getenv("PPE_MODEL_PATH", "models/ppe-yolov11.pt")
    confidence_threshold: float = float(os.getenv("PPE_CONF_THRESHOLD", "0.5"))
    device: str = os.getenv("PPE_DEVICE", "cuda")
    required_classes: tuple = field(
        default_factory=lambda: ("helmet", "vest", "gloves", "goggles")
    )


settings = Settings()
