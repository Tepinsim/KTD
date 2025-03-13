import streamlit as st
import torch
import numpy as np
from PIL import Image
import os

# Model file path
MODEL_V8_PATH = "/Users/Cs-Store/Desktop/intern2/text detection/best.pt"  # Raw string to handle backslashes

# Load model with error handling
try:
    if not os.path.exists(MODEL_V8_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_V8_PATH}")
    model_v8 = torch.hub.load('ultralytics/yolov8', 'custom', path=MODEL_V8_PATH)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit interface
st.title("Khmer Language Detection with YOLOv8 25epochs")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Khmer Text (YOLOv8)"):
        img_np = np.array(image)
        results_v5 = model_v8(img_np)
        annotated_img_v5 = results_v5.render()[0]
        st.image(annotated_img_v5, caption="YOLOv8 Detection Results", use_column_width=True)
        st.write(results_v5.pandas().xyxy[0])
