from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from footAI.evaluation import compute_metrics
from footAI.model import PressureMLP


def _evaluate_regression_metrics(
    model: PressureMLP,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Run model on all batches and return MSE, MAE, derived_class_accuracy."""
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            pred = model(x)
            preds.append(pred.cpu())
            targets.append(y)
    if not preds:
        return {"mse": 0.0, "mae": 0.0, "derived_class_accuracy": 0.0}
    y_pred = torch.cat(preds)
    y_true = torch.cat(targets)
    return compute_metrics(y_true, y_pred)


def train_pressure_model(
    model: PressureMLP,
    train_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    test_loader: Optional[DataLoader] = None,
) -> PressureMLP:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if n_batches > 0:
            epoch_loss /= n_batches
        msg = f"Epoch {epoch + 1}/{epochs} - train_loss: {epoch_loss:.6f}"
        if test_loader is not None:
            metrics = _evaluate_regression_metrics(model, test_loader, device)
            msg += (
                f" | test_mse: {metrics['mse']:.6f} | test_mae: {metrics['mae']:.6f} "
                f"| test_acc (derived class): {metrics['derived_class_accuracy']:.4f}"
            )
        print(msg)

    model.eval()
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
    return model
