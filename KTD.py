import streamlit as st
from ultralytics import YOLO  # Import YOLO from ultralytics
from PIL import Image
import numpy as np
import os
import pandas as pd #import pandas.

# Model file path
MODEL_V8_PATH = "/Users/Cs-Store/Desktop/intern2/text detection/2603yolo8m.pt"

# Load model with error handling
try:
    if not os.path.exists(MODEL_V8_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_V8_PATH}")
    model_v8 = YOLO(MODEL_V8_PATH)  # Load YOLOv8 model correctly
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
        results_v8 = model_v8(img_np)
        annotated_img_v8 = results_v8[0].plot()  # Get plotted image
        st.image(annotated_img_v8, caption="YOLOv8 Detection Results", use_column_width=True)

        # Convert bounding box data to Pandas DataFrame
        boxes = results_v8[0].boxes.xyxy.cpu().numpy()
        confidences = results_v8[0].boxes.conf.cpu().numpy()
        class_ids = results_v8[0].boxes.cls.cpu().numpy()

        if len(boxes) > 0:
            df = pd.DataFrame({
                'x1': boxes[:, 0],
                'y1': boxes[:, 1],
                'x2': boxes[:, 2],
                'y2': boxes[:, 3],
                'confidence': confidences,
                'class': class_ids
            })
            st.write(df)
        else:
            st.write("No detections found.")
