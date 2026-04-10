from pathlib import Path

from torch.utils.data import DataLoader

from footAI.data import PressureDataset
from footAI.model import PressureMLP
from footAI.training import train_pressure_model

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "footAI" / "data" / "dataset" / "train.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "pressure_mlp.pt"

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    dataset = PressureDataset(CSV_PATH)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    model = PressureMLP()
    train_pressure_model(
        model,
        loader,
        epochs=20,
        lr=1e-3,
        checkpoint_path=CHECKPOINT_PATH,
    )
    print(f"Training complete. Model saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()