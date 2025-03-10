import streamlit as st
import torch
import numpy as np
from PIL import Image
import os

# Model file path
MODEL_V5_PATH = r"C:\Users\Cs-Store\Desktop\intern2\text detection\best_yolov5.pt"  # Raw string to handle backslashes

# Load model with error handling
try:
    if not os.path.exists(MODEL_V5_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_V5_PATH}")
    model_v5 = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_V5_PATH)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit interface
st.title("Khmer Language Detection with YOLOv5")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Khmer Text (YOLOv5)"):
        img_np = np.array(image)
        results_v5 = model_v5(img_np)
        annotated_img_v5 = results_v5.render()[0]
        st.image(annotated_img_v5, caption="YOLOv5 Detection Results", use_column_width=True)
        st.write(results_v5.pandas().xyxy[0])
