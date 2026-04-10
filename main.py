import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
import torch
from PIL import Image, ImageDraw

from footAI.detection import run_detection
from footAI.features import extract_features
from footAI.model import PressureMLP, score_to_class
from footAI.team import differentiate_by_color

DEFAULT_CHECKPOINT_PATH = Path("checkpoints") / "pressure_mlp.pt"


def safest_next_action(pressure_class: str) -> str:
    """Suggest safest next action from pressure class (rule-based)."""
    if pressure_class == "Low":
        return "sprint, dribble"
    if pressure_class == "Medium":
        return "pass, cross"
    return "short pass, clearing"


st.set_page_config(page_title="FootAI — Pressure Analytics", layout="centered")
st.title("FootAI — Pressure Analytics")
st.write(
    "We measure **pressure on the player with the ball**. Choose which team has the ball "
    "(or Auto to detect from the player closest to the ball). Defenders near the ball = pressure on the ball carrier."
)


def draw_overlay(
    image_path: Path,
    attacking_boxes: list,
    defending_boxes: list,
    ball_box: Optional[object],
) -> Image.Image:
    """Draw bounding boxes on the image: attacking=green, defending=red, ball=blue."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in attacking_boxes:
        draw.rectangle(
            [(box.x1, box.y1), (box.x2, box.y2)],
            outline="lime",
            width=3,
        )
    for box in defending_boxes:
        draw.rectangle(
            [(box.x1, box.y1), (box.x2, box.y2)],
            outline="red",
            width=3,
        )
    if ball_box is not None:
        draw.rectangle(
            [(ball_box.x1, ball_box.y1), (ball_box.x2, ball_box.y2)],
            outline="blue",
            width=4,
        )
    return img


def load_model() -> PressureMLP:
    """Load PressureMLP and autoload the default checkpoint when available."""
    model = PressureMLP()
    if DEFAULT_CHECKPOINT_PATH.exists():
        model.load_state_dict(torch.load(DEFAULT_CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    return model


team_with_ball = st.radio(
    "Team with the ball (pressure is measured on their ball carrier)",
    ["Auto (detect from ball)", "Home", "Away"],
    horizontal=True,
)
attacking_team = None if team_with_ball == "Auto (detect from ball)" else team_with_ball.lower()
if DEFAULT_CHECKPOINT_PATH.exists():
    st.sidebar.success(f"Using checkpoint: {DEFAULT_CHECKPOINT_PATH}")
else:
    st.sidebar.warning(
        "No checkpoint found at checkpoints/pressure_mlp.pt. "
        "Using untrained model (scores are not meaningful)."
    )
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded is not None and (attacking_team is not None or team_with_ball == "Auto (detect from ball)"):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)

    try:
        detection = run_detection(tmp_path)
        teams = differentiate_by_color(tmp_path, detection, attacking_team if attacking_team else None)
        features = extract_features(detection, teams)

        model = load_model()
        with torch.no_grad():
            x = torch.from_numpy(features).float().unsqueeze(0)
            score = model(x).squeeze().item()
        pressure_class = score_to_class(score)
        suggested_action = safest_next_action(pressure_class)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pressure score", f"{score:.3f}")
        with col2:
            st.metric("Pressure class", pressure_class)
        with col3:
            st.metric("Safest next action", suggested_action)

        overlay = draw_overlay(
            tmp_path,
            teams.attacking_boxes,
            teams.defending_boxes,
            detection.ball_box,
        )
        st.image(overlay, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)
