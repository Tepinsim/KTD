import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import khmernltk  # Import KhmerNLTK

def normalize_text(text):
    """Normalizes the text using KhmerNLTK."""
    normalized_text = khmernltk.word_tokenize(text)
    return " ".join(normalized_text)

def main():
    st.title("Khmer Text Detection and OCR with YOLOv8 and TrOCR")

    # Add controls
    confidence_threshold = st.slider("Confidence Threshold:", 0.0, 1.0, 0.5)
    iou_threshold = st.slider("Overlap Threshold (IoU):", 0.0, 1.0, 0.5)
    label_display_mode = st.selectbox("Label Display Mode:",
                                      ["Draw Boxes", "Draw Confidence", "Draw Labels", "Censor Predictions"])

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect and Recognize Khmer Text"):
            with st.spinner("Detecting and recognizing Khmer text..."):
                # Convert PIL Image to OpenCV format
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                # Load the YOLOv8 model
                model = YOLO("/Users/Cs-Store/Desktop/intern2/yolov8/173.pt")

                # Perform inference with thresholds
                results = model(image_cv, conf=confidence_threshold, iou=iou_threshold)

                # Load TrOCR model and processor
                processor = TrOCRProcessor.from_pretrained("/Users/Cs-Store/Desktop/intern2/yolov8/fine_tuned_trocr_khmer")
                ocr_model = VisionEncoderDecoderModel.from_pretrained("/Users/Cs-Store/Desktop/intern2/yolov8/fine_tuned_trocr_khmer")

                # Process and display the results
                for r in results:
                    pil_image = image.copy()
                    draw = ImageDraw.Draw(pil_image)
                    font_path = "/Users/Cs-Store/Desktop/intern2/yolov8/Siemreap-Regular.ttf"
                    try:
                        font = ImageFont.truetype(font_path, 20)
                    except OSError:
                        st.error(f"Could not load font from {font_path}. Using default font.")
                        font = ImageFont.load_default()

                    boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                    combined_text = ""

                    for box in boxes:
                        x1, y1, x2, y2 = box
                        cropped_image = image_cv[y1:y2, x1:x2]
                        cropped_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

                        # OCR processing
                        pixel_values = processor(images=cropped_pil, return_tensors="pt").pixel_values
                        generated_ids = ocr_model.generate(pixel_values)
                        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                        normalized_generated_text = normalize_text(generated_text)  # Normalize the text

                        combined_text += normalized_generated_text + " "  # Normalize before combining.

                        if label_display_mode == "Draw Boxes":
                            draw.rectangle((x1, y1, x2, y2), outline=(0, 0, 255), width=2)
                            draw.text((x1, y2 + 5), normalized_generated_text, font=font, fill=(0, 255, 0))  # Normalize before drawing.
                        elif label_display_mode == "Draw Confidence":
                            confidence = r.boxes.conf[list(boxes).index(box)].item()
                            draw.text((x1, y1 - 20), f"Conf: {confidence:.2f}", font=font, fill=(255, 0, 0))
                            draw.text((x1, y2 + 5), normalized_generated_text, font=font, fill=(0, 255, 0))
                        elif label_display_mode == "Draw Labels":
                            class_id = int(r.boxes.cls[list(boxes).index(box)].item())
                            class_name = r.names[class_id]
                            draw.text((x1, y1 - 20), class_name, font=font, fill=(255, 0, 0))
                            draw.text((x1, y2 + 5), normalized_generated_text, font=font, fill=(0, 255, 0))
                        elif label_display_mode == "Censor Predictions":
                            draw.rectangle((x1, y1, x2, y2), fill=(0, 0, 0))
                            draw.text((x1, y2 + 5), "Censored", font=font, fill=(255, 255, 255))

                    st.image(pil_image, caption="Detected Khmer Text Boxes with Recognized Text", use_column_width=True)
                    st.write(f"Combined Normalized Recognized Text: {combined_text}")  # Normalize the combined text.

                st.success("Khmer text detection and recognition complete!")

if __name__ == "__main__":
    main()
