from typing import Tuple

# CSV column names
COLUMN_IMAGE_PATH = "image_path"
COLUMN_ATTACKING_TEAM = "attacking_team"
COLUMN_PRESSURE_SCORE = "pressure_score"
COLUMN_PRESSURE_LABEL = "pressure_label"
COLUMN_NEXT_ACTION = "next_action"

# Required: must have one of these for regression target
PRESSURE_COLUMNS = (COLUMN_PRESSURE_SCORE, COLUMN_PRESSURE_LABEL)

# Mapping: discrete label -> continuous score for regression
PRESSURE_LABEL_TO_SCORE = {0: 0.0, 1: 0.5, 2: 1.0}

REQUIRED_COLUMNS: Tuple[str, ...] = (
    COLUMN_IMAGE_PATH,
    COLUMN_ATTACKING_TEAM,
    COLUMN_NEXT_ACTION,
)

VALID_ATTACKING_TEAMS = ("home", "away")
VALID_PRESSURE_LABELS = (0, 1, 2)
