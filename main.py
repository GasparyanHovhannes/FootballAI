import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO

from detector import detect

st.title("Player and Ball Detection")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    uploaded_file.seek(0)
    result = detect(uploaded_file, model)

    draw = ImageDraw.Draw(image)

    for box in result.player_boxes:
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline="green", width=3)

    if result.ball_box is not None:
        b = result.ball_box
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline="red", width=3)

    st.image(image, caption="Detected image", use_container_width=True)
    st.write("Players detected:", len(result.player_boxes))
    st.write("Ball detected:", result.ball_box is not None)