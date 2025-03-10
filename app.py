import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

# Load YOLOv5 model for text detection (POC Model)
yolo_model = YOLO("/Users/Cs-Store/Desktop/intern2/text detection/best_yolov5.pt")

# Load fine-tuned TrOCR Khmer OCR model (POC Model)
processor = TrOCRProcessor.from_pretrained("/Users/Cs-Store/Desktop/intern2/text detection/fine_tuned_trocr_khmer")
ocr_model = VisionEncoderDecoderModel.from_pretrained("/Users/Cs-Store/Desktop/intern2/text detection/fine_tuned_trocr_khmer")


# Function to detect text regions in image using YOLOv5
def detect_text(image):
    results = yolo_model(image)
    text_regions = []
    boxes = []
    
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
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
def process_image(image):
    text_regions, boxes = detect_text(image)
    recognized_texts = recognize_text(text_regions)
    
    # Draw bounding boxes on image
    for (box, text) in zip(boxes, recognized_texts):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image, recognized_texts

# Streamlit UI
st.title("📸 Khmer OCR (POC) - Short Words")

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Run OCR pipeline
    processed_image, extracted_text = process_image(image)

    # Show output
    st.image(processed_image, caption="Detected Text with OCR", use_column_width=True)
    
    st.write("### Extracted Khmer Text:")
    for idx, text in enumerate(extracted_text):
        st.write(f"**{idx+1}.** {text}")
