# FootAI

Football pressure analytics from a single broadcast image.

The system predicts:
- a continuous `pressure_score` in `[0, 1]`
- a derived pressure class (`Low`, `Medium`, `High`)
- a visual overlay (team with ball, opponents, ball)

## Tech Stack

- Python 3.10+
- PyTorch
- Ultralytics YOLO (detection only)
- NumPy, Pandas
- Streamlit

## Project Structure

- `app.py` - Streamlit interface (inference + overlay)
- `run_train.py` - training entry script
- `footAI/detection/` - YOLO player/ball detection
- `footAI/team/` - team split by jersey color and ball-carrier logic
- `footAI/features/` - explainable spatial feature extraction
- `footAI/data/` - CSV format + dataset loader
- `footAI/model/` - PressureMLP and score/class mapping
- `footAI/training/` - training loop (Adam + MSE)
- `footAI/evaluation/` - MSE, MAE, derived-class accuracy

## Data Format

Use a CSV (for example: `footAI/data/dataset/train.csv`) with:

- `image_path`
- `attacking_team` (`home` or `away`) = team with the ball
- `pressure_label` in `{0,1,2}` mapped to `{0.0, 0.5, 1.0}`

Example:

```csv
image_path,attacking_team,pressure_score,next_action
images/image_1.png,away,0.25
images/image_2.png,home,0.70
```

If `image_path` is relative, it is resolved relative to the CSV directory.

## Setup

From project root:

```bash
python -m pip install -r requirements.txt
```

## Train

Train on real CSV data:

```bash
python run_train.py
```

Current training config in `run_train.py`:
- epochs: `20`
- batch size: `8`
- lr: `1e-3`

Checkpoint output:
- `checkpoints/pressure_mlp.pt` (folder auto-created)

## Run App

```bash
python -m streamlit run app.py
```

In the app:
- upload an image
- choose **Team with the ball** (`Auto`, `Home`, `Away`)
- checkpoint is autoloaded from `checkpoints/pressure_mlp.pt` if it exists

Outputs:
- `Pressure score`
- `Pressure class`
- `Safest next action`
- overlay:
  - green = team with the ball
  - red = opponents applying pressure
  - blue = ball

## Notes

- Pressure is modeled as **regression** (`MSELoss`), not classification.
- Score class mapping:
  - `0.00-0.33` -> `Low`
  - `0.34-0.66` -> `Medium`
  - `0.67-1.00` -> `High`
- If no ball is detected, feature extraction uses frame-center fallback for ball position.
