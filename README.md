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

## Detector weights

The detector uses a football-trained YOLOv8 model that distinguishes `ball / goalkeeper / player / referee` natively (no COCO `person`-vs-`ref` heuristic).

Download the weights once and place them at `weights/football_players_v1.pt`:

- Source: Roboflow Universe project `football-players-detection-3zvbc`
- Pick a YOLOv8 version → **Download Weights** → save the `.pt`

If the file is missing, `run_detection` raises a `FileNotFoundError` with a hint.

Note: features change with the detector, so retrain the MLP (`python run_train.py`) after switching weights.

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
python main.py
```

A Gradio app opens at http://127.0.0.1:7860.

**Image tab:** upload → click **Analyze** → see overlay + pressure metrics.

**Video tab:** upload → click **Process video** → backend samples at 2 fps and plots pressure-over-time. Drag the time slider to inspect any moment; the frame's overlay and metrics update.

Common controls:
- **Team with the ball** radio (`Auto` / `Home` / `Away`) — pressure is measured on this team's ball carrier.
- Checkpoint is autoloaded from `checkpoints/pressure_mlp.pt` if present.

Outputs:
- `Pressure score`, `Pressure class`, `Safest next action`
- Overlay: green = team with the ball; red = opponents; black = referee(s); blue = ball.

## Notes

- Pressure is modeled as **regression** (`MSELoss`), not classification.
- Score class mapping:
  - `0.00-0.33` -> `Low`
  - `0.34-0.66` -> `Medium`
  - `0.67-1.00` -> `High`
- If no ball is detected, feature extraction uses frame-center fallback for ball position.
