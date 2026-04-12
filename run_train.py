from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from footAI.data import PressureDataset
from footAI.model import PressureMLP
from footAI.training import train_pressure_model

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "footAI" / "data" / "dataset" / "train.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "pressure_mlp.pt"

TRAIN_FRACTION = 0.8
SPLIT_SEED = 42


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    dataset = PressureDataset(CSV_PATH)
    n = len(dataset)
    if n < 2:
        raise ValueError(
            "The CSV must contain at least 2 rows for an 80/20 train/test split."
        )
    n_train = int(TRAIN_FRACTION * n)
    n_test = n - n_train
    if n_train < 1 or n_test < 1:
        n_train = max(1, n - 1)
        n_test = n - n_train
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_ds, test_ds = random_split(dataset, [n_train, n_test], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    model = PressureMLP()
    train_pressure_model(
        model,
        train_loader,
        epochs=20,
        lr=1e-3,
        checkpoint_path=CHECKPOINT_PATH,
        test_loader=test_loader,
    )
    print(f"Training complete. Model saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
