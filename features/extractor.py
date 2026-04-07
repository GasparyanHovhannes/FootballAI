import math
from typing import List, Tuple

import numpy as np

from detection import BoundingBox, DetectionResult
from team import TeamDifferentiationResult

# Fixed feature order for explainability and MLP input
FEATURE_NAMES: Tuple[str, ...] = (
    "n_attacking",
    "n_defending",
    "ball_x_norm",
    "ball_y_norm",
    "min_dist_attacker_ball_norm",
    "min_dist_defender_ball_norm",
    "mean_dist_defender_ball_norm",
    "n_defenders_near_ball",
    "ratio_attacker_defender_near_ball",
    "defender_pressure_density",  # sum 1/(1+d) for defenders -> high when many close
    "n_defenders_very_near",     # count within tight radius -> more opponents close = higher pressure
)
NUM_FEATURES = len(FEATURE_NAMES)

# Radius for "near ball" as fraction of frame diagonal
NEAR_BALL_RADIUS_FRACTION = 0.2
# Tighter radius for "very near" (strong pressure signal)
VERY_NEAR_BALL_RADIUS_FRACTION = 0.1


def _center(box: BoundingBox) -> Tuple[float, float]:
    return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)


def _frame_from_detection(detection: DetectionResult) -> Tuple[float, float, float, float]:
    all_boxes = list(detection.player_boxes)
    if detection.ball_box is not None:
        all_boxes.append(detection.ball_box)
    if not all_boxes:
        return 0.0, 0.0, 1.0, 1.0
    x_min = min(b.x1 for b in all_boxes)
    y_min = min(b.y1 for b in all_boxes)
    x_max = max(b.x2 for b in all_boxes)
    y_max = max(b.y2 for b in all_boxes)
    return x_min, y_min, x_max, y_max


def _dist(
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    return math.hypot(ax - bx, ay - by)


def extract_features(
    detection: DetectionResult,
    teams: TeamDifferentiationResult,
) -> np.ndarray:
    x_min, y_min, x_max, y_max = _frame_from_detection(detection)
    w = max(x_max - x_min, 1.0)
    h = max(y_max - y_min, 1.0)
    diagonal = math.hypot(w, h)

    if detection.ball_box is not None:
        ball_cx, ball_cy = _center(detection.ball_box)
    else:
        # No ball detected: use frame center as fallback so distance features are still meaningful
        ball_cx = (x_min + x_max) / 2.0
        ball_cy = (y_min + y_max) / 2.0
    ball_x_norm = (ball_cx - x_min) / w if w > 0 else 0.5
    ball_y_norm = (ball_cy - y_min) / h if h > 0 else 0.5

    n_attacking = float(len(teams.attacking_boxes))
    n_defending = float(len(teams.defending_boxes))

    min_dist_attacker_ball_norm = 1.0
    for box in teams.attacking_boxes:
        cx, cy = _center(box)
        d = _dist(cx, cy, ball_cx, ball_cy) / diagonal if diagonal > 0 else 1.0
        min_dist_attacker_ball_norm = min(min_dist_attacker_ball_norm, d)

    min_dist_defender_ball_norm = 1.0
    defender_dists: List[float] = []
    for box in teams.defending_boxes:
        cx, cy = _center(box)
        d = _dist(cx, cy, ball_cx, ball_cy) / diagonal if diagonal > 0 else 1.0
        defender_dists.append(d)
        min_dist_defender_ball_norm = min(min_dist_defender_ball_norm, d)
    mean_dist_defender_ball_norm = float(np.mean(defender_dists)) if defender_dists else 0.0

    near_radius = NEAR_BALL_RADIUS_FRACTION * diagonal
    very_near_radius = VERY_NEAR_BALL_RADIUS_FRACTION * diagonal
    n_defenders_near = 0
    n_attackers_near = 0
    n_defenders_very_near = 0
    defender_pressure_density = 0.0  # sum of 1/(1+d) for defenders; higher when more are close
    for box in teams.defending_boxes:
        cx, cy = _center(box)
        d = _dist(cx, cy, ball_cx, ball_cy) / diagonal if diagonal > 0 else 1.0
        defender_pressure_density += 1.0 / (1.0 + d)
        if d * diagonal <= near_radius:
            n_defenders_near += 1
        if d <= VERY_NEAR_BALL_RADIUS_FRACTION:
            n_defenders_very_near += 1
    for box in teams.attacking_boxes:
        cx, cy = _center(box)
        if _dist(cx, cy, ball_cx, ball_cy) <= near_radius:
            n_attackers_near += 1
    ratio_near = n_attackers_near / (n_defenders_near + 1e-6)

    out = np.array([
        n_attacking,
        n_defending,
        ball_x_norm,
        ball_y_norm,
        min_dist_attacker_ball_norm,
        min_dist_defender_ball_norm,
        mean_dist_defender_ball_norm,
        float(n_defenders_near),
        ratio_near,
        defender_pressure_density,
        float(n_defenders_very_near),
    ], dtype=np.float32)
    return out
