from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
import torch

from footAI.data.format import (
    COLUMN_ATTACKING_TEAM,
    COLUMN_IMAGE_PATH,
    COLUMN_PRESSURE_LABEL,
    COLUMN_PRESSURE_SCORE,
    PRESSURE_LABEL_TO_SCORE,
    REQUIRED_COLUMNS,
    VALID_ATTACKING_TEAMS,
    VALID_PRESSURE_LABELS,
)
from footAI.detection import run_detection
from footAI.features import extract_features
from footAI.team import differentiate_by_color


class PressureDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        csv_path: Union[str, Path],
        base_dir: Optional[Union[str, Path]] = None,
    ):
        self.csv_path = Path(csv_path)
        self.base_dir = Path(base_dir) if base_dir is not None else self.csv_path.parent

        self.df = pd.read_csv(self.csv_path)
        for col in REQUIRED_COLUMNS:
            if col not in self.df.columns:
                raise ValueError(f"CSV missing required column: {col}")

        self.df[COLUMN_ATTACKING_TEAM] = self.df[COLUMN_ATTACKING_TEAM].str.strip().str.lower()
        invalid = ~self.df[COLUMN_ATTACKING_TEAM].isin(VALID_ATTACKING_TEAMS)
        if invalid.any():
            raise ValueError(
                f"Invalid attacking_team values: {self.df.loc[invalid, COLUMN_ATTACKING_TEAM].unique().tolist()}. "
                f"Must be one of {VALID_ATTACKING_TEAMS}."
            )

        # Regression target: use pressure_score if present, else map pressure_label
        if COLUMN_PRESSURE_SCORE in self.df.columns:
            self._use_label = False
            self.df[COLUMN_PRESSURE_SCORE] = pd.to_numeric(self.df[COLUMN_PRESSURE_SCORE], errors="coerce")
            if self.df[COLUMN_PRESSURE_SCORE].isna().any():
                raise ValueError("pressure_score must be numeric.")
            if ((self.df[COLUMN_PRESSURE_SCORE] < 0) | (self.df[COLUMN_PRESSURE_SCORE] > 1)).any():
                raise ValueError("pressure_score must be in [0, 1].")
        elif COLUMN_PRESSURE_LABEL in self.df.columns:
            self._use_label = True
            self.df[COLUMN_PRESSURE_LABEL] = pd.to_numeric(self.df[COLUMN_PRESSURE_LABEL], errors="coerce")
            if self.df[COLUMN_PRESSURE_LABEL].isna().any():
                raise ValueError("pressure_label must be numeric (0, 1, or 2).")
            invalid_label = ~self.df[COLUMN_PRESSURE_LABEL].isin(VALID_PRESSURE_LABELS)
            if invalid_label.any():
                raise ValueError(
                    f"pressure_label must be 0 (low), 1 (medium), or 2 (high). "
                    f"Found: {self.df.loc[invalid_label, COLUMN_PRESSURE_LABEL].unique().tolist()}."
                )
        else:
            raise ValueError("CSV must contain either pressure_score or pressure_label.")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if not p.is_absolute():
            p = self.base_dir / p
        return p

    def _get_pressure_score(self, row: pd.Series) -> float:
        if self._use_label:
            label = int(row[COLUMN_PRESSURE_LABEL])
            return float(PRESSURE_LABEL_TO_SCORE[label])
        return float(row[COLUMN_PRESSURE_SCORE])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = self._resolve_image_path(str(row[COLUMN_IMAGE_PATH]))
        attacking_team = row[COLUMN_ATTACKING_TEAM]
        pressure_score = self._get_pressure_score(row)

        detection = run_detection(image_path)
        teams = differentiate_by_color(image_path, detection, attacking_team=attacking_team)
        features = extract_features(detection, teams)

        x = torch.from_numpy(features).float()
        y = torch.tensor(pressure_score, dtype=torch.float32)
        return x, y
