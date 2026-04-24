from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from ultralytics import YOLO

# COCO class IDs used by pretrained YOLOv8
COCO_PERSON = 0
COCO_SPORTS_BALL = 32

# Cache: one YOLO instance per model_name, loaded once
_model_cache: Dict[str, YOLO] = {}


def get_model(model_name: str = "yolov8n.pt") -> YOLO:
    if model_name not in _model_cache:
        _model_cache[model_name] = YOLO(model_name)
    return _model_cache[model_name]


@dataclass
class BoundingBox:
    """Single detection: xyxy coordinates and confidence."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    def to_xyxy(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class DetectionResult:
    """Result of running YOLO on an image: player and ball boxes only."""
    player_boxes: List[BoundingBox]
    ball_box: Optional[BoundingBox]


def _ensure_path(image_input: Union[str, Path]) -> Path:
    p = Path(image_input)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return p


def run_detection(
    image_input: Union[str, Path],
    model_name: str = "yolov8n.pt",
    conf_threshold: float = 0.25,
    ball_conf_threshold: float = 0.1,
    imgsz: int = 960,
    ball_imgsz: int = 1280,
) -> DetectionResult:
    path = _ensure_path(image_input)
    model = get_model(model_name)
    results = model.predict(
        str(path),
        conf=conf_threshold,
        imgsz=imgsz,
        classes=[COCO_PERSON],
        verbose=False,
    )
    ball_results = model.predict(
        str(path),
        conf=ball_conf_threshold,
        imgsz=ball_imgsz,
        classes=[COCO_SPORTS_BALL],
        verbose=False,
    )

    player_boxes: List[BoundingBox] = []
    ball_box: Optional[BoundingBox] = None

    if results:
        # Single image -> results[0]
        r = results[0]
        if r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            for i in range(len(conf)):
                x1, y1, x2, y2 = xyxy[i].tolist()
                cf = float(conf[i])
                player_boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=cf))

    if ball_results:
        br = ball_results[0]
        if br.boxes is not None:
            b_xyxy = br.boxes.xyxy.cpu().numpy()
            b_conf = br.boxes.conf.cpu().numpy()
            for i in range(len(b_conf)):
                x1, y1, x2, y2 = b_xyxy[i].tolist()
                cf = float(b_conf[i])
                # Keep highest-confidence ball if multiple
                if ball_box is None or cf > ball_box.confidence:
                    ball_box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=cf)

    return DetectionResult(player_boxes=player_boxes, ball_box=ball_box)
