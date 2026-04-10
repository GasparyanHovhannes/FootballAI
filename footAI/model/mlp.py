from typing import List, Literal, Optional

import torch
import torch.nn as nn

from footAI.features import NUM_FEATURES

PressureClass = Literal["Low", "Medium", "High"]

# Score thresholds for derived class (locked per project spec)
THRESHOLD_LOW_MAX = 0.2
THRESHOLD_MEDIUM_MAX = 0.7


def score_to_class(score: float) -> PressureClass:

    if score <= THRESHOLD_LOW_MAX:
        return "Low"
    if score <= THRESHOLD_MEDIUM_MAX:
        return "Medium"
    return "High"


class PressureMLP(nn.Module):
    """
    MLP: input (spatial features) -> hidden layers -> 1 output with Sigmoid (pressure score in [0, 1]).
    """

    def __init__(
        self,
        input_size: int = NUM_FEATURES,
        hidden_sizes: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        layers: List[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        return torch.sigmoid(self.head(out)).squeeze(-1)

    def predict_class(self, score: float) -> PressureClass:
        return score_to_class(score)
