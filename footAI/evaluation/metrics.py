from typing import Dict, Union

import numpy as np
import torch

from footAI.model import score_to_class


def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x).flatten()


def mse(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray]) -> float:

    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray]) -> float:

    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def derived_class_accuracy(
    y_true_scores: Union[torch.Tensor, np.ndarray],
    y_pred_scores: Union[torch.Tensor, np.ndarray],
) -> float:
    y_true = _to_numpy(y_true_scores)
    y_pred = _to_numpy(y_pred_scores)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true_scores and y_pred_scores must have the same length")
    if len(y_true) == 0:
        return 0.0
    pred_classes = np.array([score_to_class(float(s)) for s in y_pred])
    true_classes = np.array([score_to_class(float(s)) for s in y_true])
    return float(np.mean(pred_classes == true_classes))


def compute_metrics(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
) -> Dict[str, float]:

    return {
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "derived_class_accuracy": derived_class_accuracy(y_true, y_pred),
    }
