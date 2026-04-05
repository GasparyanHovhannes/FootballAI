from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image

COCO_PERSON = 0
COCO_SPORTS_BALL = 32


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


@dataclass
class DetectionResult:
    player_boxes: List[BoundingBox]
    ball_box: Optional[BoundingBox]


def detect(uploaded_file, model, conf=0.25) -> DetectionResult:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    results = model.predict(image_np, conf=conf, verbose=False)
    r = results[0]

    player_boxes = []
    ball_box = None

    if r.boxes is None:
        return DetectionResult(player_boxes=player_boxes, ball_box=ball_box)

    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    class_ids = r.boxes.cls.cpu().numpy().astype(int)

    for i, class_id in enumerate(class_ids):
        x1, y1, x2, y2 = xyxy[i]
        box = BoundingBox(float(x1), float(y1), float(x2), float(y2), float(confs[i]))

        if class_id == COCO_PERSON:
            player_boxes.append(box)
        elif class_id == COCO_SPORTS_BALL:
            if ball_box is None or box.confidence > ball_box.confidence:
                ball_box = box

    return DetectionResult(player_boxes=player_boxes, ball_box=ball_box)