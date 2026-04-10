from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from footAI.model import PressureMLP


def train_pressure_model(
    model: PressureMLP,
    train_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> PressureMLP:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
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
        print(f"Epoch {epoch + 1}/{epochs} - train_loss: {epoch_loss:.6f}")

    model.eval()
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
    return model
