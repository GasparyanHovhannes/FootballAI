from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from ultralytics import YOLO

# Class IDs in the football-players-detection model (Roboflow Universe).
# Verify against your weights' `model.names` before relying in production.
CLASS_BALL = 0
CLASS_GOALKEEPER = 1
CLASS_PLAYER = 2
CLASS_REFEREE = 3

DEFAULT_MODEL_PATH = "weights/football_players_v1.pt"

# Cache: one YOLO instance per model_name, loaded once
_model_cache: Dict[str, YOLO] = {}


def get_model(model_name: str = DEFAULT_MODEL_PATH) -> YOLO:
    if model_name not in _model_cache:
        if not Path(model_name).exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {model_name}. "
                f"Download a football-trained YOLOv8 .pt (e.g. Roboflow Universe "
                f"'football-players-detection-3zvbc') and place it there."
            )
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
    """YOLO output split by class. Goalkeepers are included in player_boxes."""
    player_boxes: List[BoundingBox]
    ball_box: Optional[BoundingBox]
    referee_boxes: List[BoundingBox] = field(default_factory=list)


def _ensure_path(image_input: Union[str, Path]) -> Path:
    p = Path(image_input)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return p


def run_detection(
    image_input: Union[str, Path],
    model_name: str = DEFAULT_MODEL_PATH,
    conf_threshold: float = 0.15,
    ball_conf_threshold: float = 0.2,
    imgsz: int = 1280,
) -> DetectionResult:
    path = _ensure_path(image_input)
    model = get_model(model_name)
    results = model.predict(
        str(path),
        conf=conf_threshold,
        imgsz=imgsz,
        verbose=False,
    )

    player_boxes: List[BoundingBox] = []
    referee_boxes: List[BoundingBox] = []
    ball_box: Optional[BoundingBox] = None

    if not results:
        return DetectionResult(player_boxes=[], ball_box=None, referee_boxes=[])

    r = results[0]
    if r.boxes is None:
        return DetectionResult(player_boxes=[], ball_box=None, referee_boxes=[])

    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)

    for i in range(len(conf)):
        x1, y1, x2, y2 = xyxy[i].tolist()
        cf = float(conf[i])
        c = int(cls[i])
        box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=cf)
        if c == CLASS_PLAYER:
            player_boxes.append(box)
        elif c == CLASS_REFEREE:
            referee_boxes.append(box)
        elif c == CLASS_BALL:
            if cf < ball_conf_threshold:
                continue
            if ball_box is None or cf > ball_box.confidence:
                ball_box = box
        # CLASS_GOALKEEPER is intentionally dropped — see TeamDifferentiationResult.

    return DetectionResult(
        player_boxes=player_boxes,
        ball_box=ball_box,
        referee_boxes=referee_boxes,
    )
