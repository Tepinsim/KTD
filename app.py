import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
import os

# Set paths for models
YOLO_MODEL_PATH = "/Users/Cs-Store/Desktop/intern2/text detection/best_yolov5.pt"
TROCR_MODEL_PATH = "/Users/Cs-Store/Desktop/intern2/text detection/fine_tuned_trocr_khmer"

# Load YOLOv5 text detection model
if os.path.exists(YOLO_MODEL_PATH):
    yolo_model = YOLO(YOLO_MODEL_PATH)
else:
    st.error(f"YOLO model not found at: {YOLO_MODEL_PATH}")

# Load TrOCR Khmer OCR model
if os.path.exists(TROCR_MODEL_PATH):
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_PATH)
    ocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_PATH)
else:
    st.error(f"TrOCR model not found at: {TROCR_MODEL_PATH}")

# Function to detect text regions using YOLOv5
def detect_text(image, conf_threshold=0.5):
    results = yolo_model(image)
    text_regions = []
    boxes = []

    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            if conf >= conf_threshold:  # Apply confidence threshold
                cropped = image[int(y1):int(y2), int(x1):int(x2)]
                text_regions.append(cropped)
                boxes.append((int(x1), int(y1), int(x2), int(y2)))

    return text_regions, boxes

# Function to recognize text using TrOCR
def recognize_text(text_regions):
    recognized_texts = []

    for region in text_regions:
        image_pil = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
        pixel_values = processor(images=image_pil, return_tensors="pt").pixel_values

        with torch.no_grad():
            generated_text = ocr_model.generate(pixel_values)
            text = processor.batch_decode(generated_text, skip_special_tokens=True)[0]

        recognized_texts.append(text)

    return recognized_texts

# Function to process an image
def process_image(image, conf_threshold=0.5):
    text_regions, boxes = detect_text(image, conf_threshold)
    
    if not text_regions:
        return image, ["No text detected"]

    recognized_texts = recognize_text(text_regions)

    # Draw bounding boxes on image
    for (box, text) in zip(boxes, recognized_texts):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, recognized_texts

# Streamlit UI
st.set_page_config(page_title="Khmer OCR POC", layout="wide")
st.title("📸 Khmer OCR (POC) - Short Words")

# Sidebar settings
st.sidebar.header("🔧 Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

# Upload Image
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image = np.array(image, dtype=np.uint8)  # Convert PIL image to numpy

    # Run OCR pipeline
    processed_image, extracted_text = process_image(image, conf_threshold)

    # Show results
    st.image(processed_image, caption="Detected Text with OCR", use_column_width=True)
    
    st.write("### 📝 Extracted Khmer Text:")
    for idx, text in enumerate(extracted_text):
        st.write(f"**{idx+1}.** {text}")
