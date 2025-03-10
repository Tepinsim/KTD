import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# Load models
try:
    model_v5 = torch.hub.load('ultralytics/yolov5', 'custom', path='best_yolov5.pt')
    model_v8 = torch.hub.load('ultralytics/ultralytics', 'custom', path='V8.pt')
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure 'best_yolov5.pt' and 'V8.pt' are in the correct directory.")
    st.stop()

# Streamlit interface
st.title("Khmer Language Detection with YOLOv5 and YOLOv8")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Khmer Text (YOLOv5)"):
        img_np = np.array(image)
        results_v5 = model_v5(img_np)

        # Draw bounding boxes and labels on the image
        annotated_img_v5 = results_v5.render()[0]  # Get the annotated image

        st.image(annotated_img_v5, caption="YOLOv5 Detection Results", use_column_width=True)
        st.write(results_v5.pandas().xyxy[0]) #display results as a dataframe

    if st.button("Detect Khmer Text (YOLOv8)"):
        img_np = np.array(image)
        results_v8 = model_v8(img_np)

        annotated_img_v8 = results_v8[0].plot() # get the annotated image from results

        st.image(annotated_img_v8, caption="YOLOv8 Detection Results", use_column_width=True)
        st.write(results_v8[0].pandas().xyxy[0]) #display results as a dataframe