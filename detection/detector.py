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
    """
    Return a cached YOLO model for the given name. Loads once per model_name, reuses thereafter.
    """
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
) -> DetectionResult:
    path = _ensure_path(image_input)
    model = get_model(model_name)
    results = model(str(path), conf=conf_threshold, verbose=False)

    player_boxes: List[BoundingBox] = []
    ball_box: Optional[BoundingBox] = None

    if not results:
        return DetectionResult(player_boxes=player_boxes, ball_box=ball_box)

    # Single image -> results[0]
    r = results[0]
    if r.boxes is None:
        return DetectionResult(player_boxes=player_boxes, ball_box=ball_box)

    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)

    for i, cid in enumerate(cls_ids):
        x1, y1, x2, y2 = xyxy[i].tolist()
        cf = float(conf[i])
        box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=cf)
        if cid == COCO_PERSON:
            player_boxes.append(box)
        elif cid == COCO_SPORTS_BALL:
            # Keep highest-confidence ball if multiple
            if ball_box is None or cf > ball_box.confidence:
                ball_box = box

    return DetectionResult(player_boxes=player_boxes, ball_box=ball_box)
