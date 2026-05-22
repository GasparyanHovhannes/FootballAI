import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from footAI.detection import run_detection
from footAI.features import extract_features
from footAI.model import PressureMLP, score_to_class
from footAI.team import differentiate_by_color

DEFAULT_CHECKPOINT_PATH = Path("checkpoints") / "pressure_mlp.pt"
SAMPLE_FPS = 2  # how many frames per second of video we analyze for the timeline


def safest_next_action(pressure_class: str, score: float) -> str:
    if score < 0.15:
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
    referee_boxes: list,
    ball_box: Optional[object],
) -> Image.Image:
    img = image_source.convert("RGB") if isinstance(image_source, Image.Image) else Image.open(image_source).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in attacking_boxes:
        draw.rectangle([(box.x1, box.y1), (box.x2, box.y2)], outline="lime", width=3)
    for box in defending_boxes:
        draw.rectangle([(box.x1, box.y1), (box.x2, box.y2)], outline="red", width=3)
    for box in referee_boxes:
        draw.rectangle([(box.x1, box.y1), (box.x2, box.y2)], outline="black", width=4)
    if ball_box is not None:
        draw.rectangle([(ball_box.x1, ball_box.y1), (ball_box.x2, ball_box.y2)], outline="blue", width=4)
    return img


def load_model() -> PressureMLP:
    model = PressureMLP()
    if DEFAULT_CHECKPOINT_PATH.exists():
        model.load_state_dict(torch.load(DEFAULT_CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    return model


def analyze_frame_path(
    frame_path: Path,
    attacking_team: Optional[str],
) -> Tuple[object, object, np.ndarray, float, str, str, Image.Image]:
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
    suggested_action = safest_next_action(pressure_class=pressure_class, score=score)
    overlay = draw_overlay(
        frame_path,
        teams.attacking_boxes,
        teams.defending_boxes,
        teams.referee_boxes,
        detection.ball_box,
    )
    return detection, teams, features, score, pressure_class, suggested_action, overlay


def _team_to_internal(team_choice: str) -> Optional[str]:
    return None if team_choice == "Auto" else team_choice.lower()


def analyze_image(image_path: Optional[str], team_choice: str):
    """Image tab: upload → overlay + metrics."""
    if image_path is None:
        return None, "—", "—", "—"
    try:
        _, _, _, score, pclass, action, overlay = analyze_frame_path(Path(image_path), _team_to_internal(team_choice))
        return overlay, f"{score:.3f}", pclass, action
    except ValueError as e:
        if str(e) == "NO_BALL":
            return None, "—", "No ball detected", "—"
        raise


def process_video(video_path: Optional[str], team_choice: str, progress=gr.Progress()):
    if video_path is None:
        return None, None, gr.update(maximum=1.0, value=0.0), "No video uploaded."

    team = _team_to_internal(team_choice)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, gr.update(maximum=1.0, value=0.0), "Could not open video."
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n_frames <= 0 or fps <= 0:
        cap.release()
        return None, None, gr.update(maximum=1.0, value=0.0), "Could not read video metadata."

    step = max(1, int(round(fps / SAMPLE_FPS)))
    tmp_dir = Path(tempfile.mkdtemp(prefix="footai_vid_"))
    rows = []
    sample_indices = list(range(0, n_frames, step))
    total = len(sample_indices)

    for k, frame_idx in enumerate(sample_indices):
        progress(k / total, desc=f"Analyzing frame {k + 1}/{total}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        t = frame_idx / fps
        frame_path = tmp_dir / f"f_{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame_bgr)
        try:
            _, _, _, score, pclass, _, _ = analyze_frame_path(frame_path, team)
            rows.append({"time": float(t), "score": float(score), "class": pclass, "frame_path": str(frame_path)})
        except ValueError as e:
            if str(e) == "NO_BALL":
                rows.append({"time": float(t), "score": None, "class": "No ball", "frame_path": str(frame_path)})
            else:
                rows.append({"time": float(t), "score": None, "class": "Error", "frame_path": str(frame_path)})

    cap.release()
    df = pd.DataFrame(rows)
    duration = float(df["time"].max()) if not df.empty else 0.0

    valid = df.dropna(subset=["score"])
    if not valid.empty:
        init_time = float(valid.loc[valid["score"].idxmax(), "time"])
    else:
        init_time = 0.0

    plot_df = df.dropna(subset=["score"])[["time", "score"]]
    n_valid = len(plot_df)
    status = f"Processed {total} frames, {n_valid} with a detected ball."

    return df, plot_df, gr.update(maximum=duration, value=init_time, step=0.5), status


def on_plot_click(evt: gr.SelectData):
    """Click a point on the LinePlot -> move the slider to that x value."""
    val = evt.value
    t = None
    if isinstance(val, dict):
        t = val.get("time") or val.get("x")
    elif isinstance(val, (list, tuple)) and len(val) >= 1:
        t = val[0]
    else:
        t = val
    try:
        return float(t)
    except (TypeError, ValueError):
        return gr.update()


def select_video_frame(df, selected_time: float, team_choice: str):
    """Re-analyze the sampled frame closest to selected_time and return its overlay + metrics."""
    if df is None or df.empty:
        return None, "—", "—", "—"
    df["_dt"] = (df["time"] - float(selected_time)).abs()
    row = df.loc[df["_dt"].idxmin()]
    df.drop(columns=["_dt"], inplace=True, errors="ignore")
    frame_path = Path(str(row["frame_path"]))
    if not frame_path.exists():
        return None, "—", "Frame expired (re-process video)", "—"
    try:
        _, _, _, score, pclass, action, overlay = analyze_frame_path(frame_path, _team_to_internal(team_choice))
        return overlay, f"{score:.3f}", pclass, action
    except ValueError as e:
        if str(e) == "NO_BALL":
            return Image.open(frame_path), "—", "No ball detected", "—"
        raise


CUSTOM_CSS = """
.gradio-container { max-width: 980px !important; margin: 0 auto !important; }
#result-image img { max-height: 520px !important; object-fit: contain; }
#video-upload video { max-height: 360px !important; }
.metric-box textarea { text-align: center; font-weight: 600; }
"""

with gr.Blocks(title="FootAI — Pressure Analytics", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# FootAI — Pressure Analytics")
    gr.Markdown(
        "Measure **pressure on the player with the ball**. Pick which team has the ball "
        "(or **Auto** to detect from the player closest to the ball). Defenders near the ball = pressure on the ball carrier."
    )
    if not DEFAULT_CHECKPOINT_PATH.exists():
        gr.Markdown(
            "**Warning:** no checkpoint at `checkpoints/pressure_mlp.pt`. "
            "Scores will be from an untrained model and won't be meaningful. Run `python run_train.py` first."
        )

    team_choice = gr.Radio(
        choices=["Auto", "Home", "Away"],
        value="Auto",
        label="Team with the ball",
    )

    with gr.Tabs():
        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="filepath", label="Upload image", height=320)
                    image_btn = gr.Button("Analyze", variant="primary", size="lg")
            image_overlay = gr.Image(label="Result overlay", interactive=False, height=520, elem_id="result-image")
            with gr.Row():
                image_score = gr.Textbox(label="Pressure score", interactive=False, elem_classes=["metric-box"])
                image_class = gr.Textbox(label="Pressure class", interactive=False, elem_classes=["metric-box"])
                image_action = gr.Textbox(label="Safest next action", interactive=False, elem_classes=["metric-box"])
            image_btn.click(
                analyze_image,
                inputs=[image_input, team_choice],
                outputs=[image_overlay, image_score, image_class, image_action],
            )

        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload video", height=360, elem_id="video-upload")
                    process_btn = gr.Button("Process video", variant="primary", size="lg")
                    status_md = gr.Markdown("")

            df_state = gr.State(value=None)

            timeline_plot = gr.LinePlot(
                value=None,
                x="time",
                y="score",
                x_title="Time (s)",
                y_title="Pressure score",
                title="Pressure over time (click a point to jump)",
                height=260,
            )
            time_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                step=0.5,
                label="Time (s) — drag to inspect a moment",
                interactive=True,
            )
            video_overlay = gr.Image(label="Frame at selected time", interactive=False, height=520, elem_id="result-image")
            with gr.Row():
                video_score = gr.Textbox(label="Pressure score", interactive=False, elem_classes=["metric-box"])
                video_class = gr.Textbox(label="Pressure class", interactive=False, elem_classes=["metric-box"])
                video_action = gr.Textbox(label="Safest next action", interactive=False, elem_classes=["metric-box"])

            process_btn.click(
                process_video,
                inputs=[video_input, team_choice],
                outputs=[df_state, timeline_plot, time_slider, status_md],
            ).then(
                select_video_frame,
                inputs=[df_state, time_slider, team_choice],
                outputs=[video_overlay, video_score, video_class, video_action],
            )

            time_slider.release(
                select_video_frame,
                inputs=[df_state, time_slider, team_choice],
                outputs=[video_overlay, video_score, video_class, video_action],
            )

            timeline_plot.select(
                on_plot_click,
                inputs=None,
                outputs=time_slider,
            ).then(
                select_video_frame,
                inputs=[df_state, time_slider, team_choice],
                outputs=[video_overlay, video_score, video_class, video_action],
            )


if __name__ == "__main__":
    demo.queue().launch()
