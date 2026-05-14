import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw

from footAI.detection import run_detection
from footAI.features import extract_features
from footAI.model import PressureMLP, score_to_class
from footAI.team import differentiate_by_color

DEFAULT_CHECKPOINT_PATH = Path("checkpoints") / "pressure_mlp.pt"


def _center_x(box: object) -> float:
    return (float(box.x1) + float(box.x2)) / 2.0


def _opposite_goalkeeper_seen(attacking_team: Optional[str], ball_box: object, goalkeeper_boxes: list) -> bool:
    if attacking_team not in ("home", "away"):
        return False
    ball_x = _center_x(ball_box)
    if attacking_team == "away":
        return any(_center_x(gk) < ball_x for gk in goalkeeper_boxes)
    return any(_center_x(gk) > ball_x for gk in goalkeeper_boxes)


def safest_next_action(
    pressure_class: str,
    score: float,
    attacking_team: Optional[str],
    ball_box: object,
    goalkeeper_boxes: list,
) -> str:
    """Suggest safest next action from pressure class (rule-based)."""
    if score < 0.15:
        if goalkeeper_boxes and _opposite_goalkeeper_seen(attacking_team, ball_box, goalkeeper_boxes):
            return "shoot or dribble"
        return "sprint or dribble"
    if pressure_class == "Low":
        return "dribble or short pass"
    if pressure_class == "Medium":
        return "short pass"
    return "clear or long ball"


def draw_overlay(
    image_source: Union[Path, Image.Image],
    attacking_boxes: list,
    defending_boxes: list,
    goalkeeper_boxes: list,
    referee_boxes: list,
    ball_box: Optional[object],
) -> Image.Image:
    """Draw bounding boxes on the image: attacking=green, defending=red, GK=white, referee=black, ball=blue."""
    if isinstance(image_source, Image.Image):
        img = image_source.convert("RGB")
    else:
        img = Image.open(image_source).convert("RGB")
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
    for box in goalkeeper_boxes:
        draw.rectangle(
            [(box.x1, box.y1), (box.x2, box.y2)],
            outline="white",
            width=4,
        )
    for box in referee_boxes:
        draw.rectangle(
            [(box.x1, box.y1), (box.x2, box.y2)],
            outline="black",
            width=4,
        )
    if ball_box is not None:
        draw.rectangle(
            [(ball_box.x1, ball_box.y1), (ball_box.x2, ball_box.y2)],
            outline="blue",
            width=4,
        )
    return img


def load_model() -> PressureMLP:
    """Load PressureMLP and auto-load the default checkpoint when available."""
    model = PressureMLP()
    if DEFAULT_CHECKPOINT_PATH.exists():
        model.load_state_dict(torch.load(DEFAULT_CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    return model


def analyze_frame_path(
    frame_path: Path,
    attacking_team: Optional[str],
) -> Tuple[object, object, np.ndarray, float, str, str, Image.Image]:
    """
    Run detection → teams → features → model → overlay.
    Returns (detection, teams, features, score, pressure_class, suggested_action, overlay_pil).
    """
    detection = run_detection(frame_path)
    if detection.ball_box is None:
        raise ValueError("NO_BALL")
    teams = differentiate_by_color(frame_path, detection, attacking_team if attacking_team else None)
    features = extract_features(detection, teams)
    model = load_model()
    with torch.no_grad():
        x = torch.from_numpy(features).float().unsqueeze(0)
        score = float(model(x).squeeze().item())
    pressure_class = score_to_class(score)
    suggested_action = safest_next_action(
        pressure_class=pressure_class,
        score=score,
        attacking_team=attacking_team,
        ball_box=detection.ball_box,
        goalkeeper_boxes=teams.goalkeeper_boxes,
    )
    overlay = draw_overlay(
        frame_path,
        teams.attacking_boxes,
        teams.defending_boxes,
        teams.goalkeeper_boxes,
        teams.referee_boxes,
        detection.ball_box,
    )
    return detection, teams, features, score, pressure_class, suggested_action, overlay


def _read_video_frame(video_path: Path, frame_index: int) -> Optional[np.ndarray]:
    """Return BGR frame (OpenCV convention) or None."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n <= 0:
        cap.release()
        return None
    frame_index = int(np.clip(frame_index, 0, n - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def _bgr_to_rgb_display(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


st.set_page_config(page_title="FootAI — Pressure Analytics", layout="centered")
st.title("FootAI — Pressure Analytics")
st.write(
    "We measure **pressure on the player with the ball**. Choose which team has the ball "
    "(or Auto to detect from the player closest to the ball). Defenders near the ball = pressure on the ball carrier."
)

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

input_mode = st.radio("Input", ["Image", "Video (upload)"], horizontal=True)

can_run = attacking_team is not None or team_with_ball == "Auto (detect from ball)"

if input_mode == "Image":
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded is not None and can_run:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)
        try:
            try:
                _, _, _, score, pressure_class, suggested_action, overlay = analyze_frame_path(
                    tmp_path, attacking_team
                )
            except ValueError as e:
                if str(e) == "NO_BALL":
                    st.warning("No ball detected — it may not be a football match (or the ball is occluded).")
                    st.stop()
                raise
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pressure score", f"{score:.3f}")
            with col2:
                st.metric("Pressure class", pressure_class)
            with col3:
                st.metric("Safest next action", suggested_action)
            st.image(
                overlay,
                caption="Green: team with the ball (pressure on their carrier). Red: opponents applying pressure. White: goalkeeper(s). Black: referee(s). Blue: ball.",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            tmp_path.unlink(missing_ok=True)

else:
    st.caption(
        "Upload a video file, scrub to the frame you want "
        "then run FootAI on that single frame."
    )
    video_file = st.file_uploader("Upload video", type=["mp4", "webm", "avi", "mov", "mkv"])

    if video_file is not None and can_run:
        suffix = Path(video_file.name).suffix if video_file.name else ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as vtmp:
            vtmp.write(video_file.getvalue())
            video_path = Path(vtmp.name)

        try:
            vid_key = f"{video_file.name}:{video_path.stat().st_size}"
            if st.session_state.get("video_session_key") != vid_key:
                st.session_state.video_session_key = vid_key
                st.session_state["video_frame_slider"] = 0

            cap_probe = cv2.VideoCapture(str(video_path))
            if not cap_probe.isOpened():
                st.error("Could not open video file.")
                st.stop()
            n_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = cap_probe.get(cv2.CAP_PROP_FPS) or 0.0
            cap_probe.release()
            if n_frames <= 0:
                st.error("Could not read frame count from this video (codec may be unsupported).")
                st.stop()

            cur = int(np.clip(int(st.session_state.get("video_frame_slider", 0)), 0, n_frames - 1))
            st.session_state["video_frame_slider"] = cur

            c_prev, c_slider, c_next, c_analyze = st.columns([1, 6, 1, 2])
            with c_prev:
                if st.button("◀", help="Previous frame"):
                    st.session_state["video_frame_slider"] = max(0, cur - 1)
                    st.rerun()
            with c_next:
                if st.button("▶", help="Next frame"):
                    st.session_state["video_frame_slider"] = min(n_frames - 1, cur + 1)
                    st.rerun()
            with c_slider:
                st.slider(
                    "Frame (scrub = pause on a frame)",
                    0,
                    n_frames - 1,
                    key="video_frame_slider",
                )
            with c_analyze:
                analyze_clicked = st.button("Analyze this frame", type="primary", use_container_width=True)

            frame_index = int(np.clip(int(st.session_state["video_frame_slider"]), 0, n_frames - 1))
            frame_bgr = _read_video_frame(video_path, frame_index)
            if frame_bgr is None:
                st.error("Failed to read this frame from the video.")
                st.stop()

            st.image(
                _bgr_to_rgb_display(frame_bgr),
                caption=f"Preview — frame {frame_index + 1}/{n_frames}"
                + (f" (~{fps:.2f} fps)" if fps > 1e-3 else ""),
                use_container_width=True,
            )

            if analyze_clicked:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as ftmp:
                    fpath = Path(ftmp.name)
                try:
                    cv2.imwrite(str(fpath), frame_bgr)
                    try:
                        _, _, _, score, pressure_class, suggested_action, overlay = analyze_frame_path(
                            fpath, attacking_team
                        )
                    except ValueError as e:
                        if str(e) == "NO_BALL":
                            st.warning(
                                "No ball detected on this frame — try another frame or a clearer shot of the ball."
                            )
                            st.stop()
                        raise
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Pressure score", f"{score:.3f}")
                    with col2:
                        st.metric("Pressure class", pressure_class)
                    with col3:
                        st.metric("Safest next action", suggested_action)
                    st.image(
                        overlay,
                        caption="Green: team with the ball (pressure on their carrier). Red: opponents applying pressure. White: goalkeeper(s). Black: referee(s). Blue: ball.",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    fpath.unlink(missing_ok=True)

        finally:
            video_path.unlink(missing_ok=True)
