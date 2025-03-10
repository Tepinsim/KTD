import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os

# ------------------------------
# 🔹 Model File Paths
# ------------------------------
YOLO_MODEL_PATH = r"C:\Users\Cs-Store\Desktop\intern2\text detection\best_yolov5.pt"
TROCR_MODEL_PATH = r"C:\Users\Cs-Store\Desktop\intern2\text detection\fine_tuned_trocr_khmer"

# ------------------------------
# 🔹 Load YOLOv5 Model (Text Detection)
# ------------------------------
try:
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"YOLOv5 model not found: {YOLO_MODEL_PATH}")
    
    yolo_model = torch.hub.load("ultralytics/yolov5", "custom", path=YOLO_MODEL_PATH, source="local")

except Exception as e:
    st.error(f"Error loading YOLOv5 model: {e}")
    st.stop()

# ------------------------------
# 🔹 Load TrOCR Model (OCR)
# ------------------------------
try:
    if not os.path.exists(TROCR_MODEL_PATH):
        raise FileNotFoundError(f"TrOCR model not found: {TROCR_MODEL_PATH}")

    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_PATH)
    ocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_PATH)

except Exception as e:
    st.error(f"Error loading TrOCR model: {e}")
    st.stop()

# ------------------------------
# 🔹 Text Detection Function (YOLOv5)
# ------------------------------
def detect_text(image, conf_threshold=0.5):
    results = yolo_model(image)  # Get predictions
    text_regions = []
    boxes = []

    for index, row in results.pandas().xyxy[0].iterrows():
        x1, y1, x2, y2, conf = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"]), row["confidence"]

        if conf >= conf_threshold:
            cropped = image[y1:y2, x1:x2]  # Crop detected text region
            text_regions.append(cropped)
            boxes.append((x1, y1, x2, y2))

    return text_regions, boxes

# ------------------------------
# 🔹 OCR Function (TrOCR)
# ------------------------------
def recognize_text(text_regions):
    recognized_texts = []

    for region in text_regions:
        image_pil = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))  # Convert OpenCV image to PIL
        pixel_values = processor(images=image_pil, return_tensors="pt").pixel_values

        with torch.no_grad():
            generated_text = ocr_model.generate(pixel_values)
            text = processor.batch_decode(generated_text, skip_special_tokens=True)[0]

        recognized_texts.append(text)

    return recognized_texts

# ------------------------------
# 🔹 Process Image (Detection + OCR)
# ------------------------------
def process_image(image, conf_threshold=0.5):
    text_regions, boxes = detect_text(image, conf_threshold)

    if not text_regions:
        return image, ["No text detected"]

    recognized_texts = recognize_text(text_regions)

    # Draw bounding boxes & OCR results
    for (box, text) in zip(boxes, recognized_texts):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, recognized_texts

# ------------------------------
# 🔹 Streamlit UI
# ------------------------------
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
