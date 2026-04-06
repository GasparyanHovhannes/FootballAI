import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
import torch
from PIL import Image, ImageDraw

from detection import run_detection
from team import differentiate_by_color


def safest_next_action(pressure_class: str) -> str:
    """Suggest safest next action from pressure class (rule-based)."""
    if pressure_class == "Low":
        return "dribble or short pass"
    if pressure_class == "Medium":
        return "short pass"
    return "clear or long ball"


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

team_with_ball = st.radio(
    "Team with the ball (pressure is measured on their ball carrier)",
    ["Auto (detect from ball)", "Home", "Away"],
    horizontal=True,
)
attacking_team = None if team_with_ball == "Auto (detect from ball)" else team_with_ball.lower()
checkpoint_path = st.sidebar.text_input(
    "Model checkpoint (optional)",
    value="",
    help="Path to pressure_mlp.pt. Leave empty for untrained weights.",
)
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded is not None and (attacking_team is not None or team_with_ball == "Auto (detect from ball)"):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)

    try:
        detection = run_detection(tmp_path)
        teams = differentiate_by_color(tmp_path, detection, attacking_team if attacking_team else None)

        overlay = draw_overlay(
            tmp_path,
            teams.attacking_boxes,
            teams.defending_boxes,
            detection.ball_box,
        )
        st.image(overlay, caption="Green: team with the ball (pressure on their carrier). Red: opponents applying pressure. Blue: ball.", use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)
