from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np

from footAI.detection import BoundingBox, DetectionResult

AttackingTeam = Literal["home", "away"]


@dataclass
class TeamDifferentiationResult:
    """Players partitioned into attacking and defending by spatial heuristic."""
    attacking_boxes: List[BoundingBox]
    defending_boxes: List[BoundingBox]
    goalkeeper_boxes: List[BoundingBox]
    referee_boxes: List[BoundingBox]


def _player_center_x(box: BoundingBox) -> float:
    return (box.x1 + box.x2) / 2.0


def _player_center(box: BoundingBox) -> tuple:
    return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)


def _ball_center(detection: DetectionResult) -> Optional[tuple]:
    """Ball (x, y) or None if no ball."""
    if detection.ball_box is None:
        return None
    return _player_center(detection.ball_box)


def _dist_to_ball(box: BoundingBox, detection: DetectionResult) -> float:
    """Euclidean distance from player center to ball center; inf if no ball."""
    bc = _ball_center(detection)
    if bc is None:
        return float("inf")
    pc = _player_center(box)
    return np.hypot(pc[0] - bc[0], pc[1] - bc[1])


def _ball_reference_x(detection: DetectionResult) -> float:
    """X coordinate used to split pitch: ball center if present, else median of player centers."""
    if detection.ball_box is not None:
        return _player_center_x(detection.ball_box)
    if not detection.player_boxes:
        return 0.0
    centers = [_player_center_x(b) for b in detection.player_boxes]
    centers.sort()
    return centers[len(centers) // 2]


def differentiate(
    detection: DetectionResult,
    attacking_team: AttackingTeam,
) -> TeamDifferentiationResult:
    ref_x = _ball_reference_x(detection)
    attacking_boxes: List[BoundingBox] = []
    defending_boxes: List[BoundingBox] = []

    for box in detection.player_boxes:
        cx = _player_center_x(box)
        if attacking_team == "home":
            is_attacking = cx > ref_x
        else:
            is_attacking = cx < ref_x
        if is_attacking:
            attacking_boxes.append(box)
        else:
            defending_boxes.append(box)

    return TeamDifferentiationResult(
        attacking_boxes=attacking_boxes,
        defending_boxes=defending_boxes,
        goalkeeper_boxes=[],
        referee_boxes=[],
    )


def _load_image_rgb(image_path: Union[str, Path]) -> np.ndarray:
    """Load image as (H, W, 3) RGB uint8."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def _jersey_color_from_crop(img: np.ndarray, box: BoundingBox) -> np.ndarray:
    """Extract a single RGB color for the player (torso region to reduce grass/shorts)."""
    h, w = img.shape[0], img.shape[1]
    x1, x2 = max(0, int(box.x1)), min(w, int(box.x2))
    y1, y2 = max(0, int(box.y1)), min(h, int(box.y2))
    if x2 <= x1 or y2 <= y1:
        return np.array([128.0, 128.0, 128.0])
    crop = img[y1:y2, x1:x2]
    # Use center 50% vertically (torso) to reduce grass and shorts
    cy = crop.shape[0] // 2
    quarter = crop.shape[0] // 4
    torso = crop[quarter : quarter + cy, :]
    if torso.size == 0:
        return np.mean(crop.reshape(-1, 3), axis=0).astype(np.float64)
    # Exclude very green pixels (grass)
    green = torso[:, :, 1]
    red, blue = torso[:, :, 0], torso[:, :, 2]
    not_grass = (green < np.median(green) + 30) | (red > green) | (blue > green)
    if not_grass.sum() > 10:
        pixels = torso[not_grass].reshape(-1, 3)
        return np.mean(pixels, axis=0).astype(np.float64)
    return np.mean(torso.reshape(-1, 3), axis=0).astype(np.float64)


def _kmeans2(data: np.ndarray, max_iters: int = 20) -> np.ndarray:
    """Cluster (N, D) into 2 groups; returns labels (N,) 0 or 1."""
    n = data.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.int32)
    c0 = data[0].copy()
    c1 = data[-1].copy() if n > 1 else data[0].copy()
    for _ in range(max_iters):
        d0 = np.linalg.norm(data - c0, axis=1)
        d1 = np.linalg.norm(data - c1, axis=1)
        labels = (d1 < d0).astype(np.int32)
        n0, n1 = labels.sum(), n - labels.sum()
        if n0 > 0:
            c0 = data[labels == 0].mean(axis=0)
        if n1 > 0:
            c1 = data[labels == 1].mean(axis=0)
    return labels


def differentiate_by_color(
    image_path: Union[str, Path],
    detection: DetectionResult,
    attacking_team: Optional[AttackingTeam] = None,
) -> TeamDifferentiationResult:
    if len(detection.player_boxes) < 2:
        if attacking_team is None:
            attacking_team = "home"
        return differentiate(detection, attacking_team)

    img = _load_image_rgb(image_path)
    colors = np.array([_jersey_color_from_crop(img, b) for b in detection.player_boxes])

    # Referee heuristic: very dark kit (low brightness)
    brightness = colors.mean(axis=1)
    referee_mask = brightness < 45.0
    referee_boxes = [detection.player_boxes[i] for i in range(len(detection.player_boxes)) if referee_mask[i]]

    # Cluster remaining players into two teams
    keep_idxs = [i for i in range(len(detection.player_boxes)) if not referee_mask[i]]
    if len(keep_idxs) < 2:
        if attacking_team is None:
            attacking_team = "home"
        base = differentiate(detection, attacking_team)
        return TeamDifferentiationResult(
            attacking_boxes=base.attacking_boxes,
            defending_boxes=base.defending_boxes,
            goalkeeper_boxes=[],
            referee_boxes=referee_boxes,
        )

    kept_colors = colors[keep_idxs]
    kept_labels = _kmeans2(kept_colors)

    # Infer attacking = team with the ball (player closest to ball), or use home/away
    if attacking_team is None and _ball_center(detection) is not None:
        closest_idx = min(
            range(len(detection.player_boxes)),
            key=lambda i: _dist_to_ball(detection.player_boxes[i], detection),
        )
        # If the closest player got filtered as referee, fall back to home/away heuristic.
        if closest_idx in keep_idxs:
            attacking_cluster = int(kept_labels[keep_idxs.index(closest_idx)])
        else:
            attacking_team = "home"
            attacking_cluster = 0
    else:
        if attacking_team is None:
            attacking_team = "home"
        mean_x_by_cluster = [0.0, 0.0]
        count = [0, 0]
        for j, i in enumerate(keep_idxs):
            box = detection.player_boxes[i]
            c = int(kept_labels[j])
            mean_x_by_cluster[c] += _player_center_x(box)
            count[c] += 1
        for c in range(2):
            if count[c] > 0:
                mean_x_by_cluster[c] /= count[c]
        if attacking_team == "home":
            attacking_cluster = 1 if mean_x_by_cluster[1] > mean_x_by_cluster[0] else 0
        else:
            attacking_cluster = 0 if mean_x_by_cluster[0] < mean_x_by_cluster[1] else 1

    attacking_boxes: List[BoundingBox] = []
    defending_boxes: List[BoundingBox] = []
    for j, i in enumerate(keep_idxs):
        if int(kept_labels[j]) == int(attacking_cluster):
            attacking_boxes.append(detection.player_boxes[i])
        else:
            defending_boxes.append(detection.player_boxes[i])

    return TeamDifferentiationResult(
        attacking_boxes=attacking_boxes,
        defending_boxes=defending_boxes,
        goalkeeper_boxes=[],
        referee_boxes=referee_boxes,
    )
