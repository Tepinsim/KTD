import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

def main():
    st.title("Khmer Text Detection and OCR with YOLOv8 and TrOCR")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect and Recognize Khmer Text"):
            with st.spinner("Detecting and recognizing Khmer text..."):
                # Convert PIL Image to OpenCV format
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                pil_image = image.convert("RGB") #keep pil image for drawing text later.

                # Load the YOLOv8 model
                model = YOLO("/Users/Cs-Store/Desktop/intern2/yolov8/173.pt")

                # Perform inference
                results = model(image_cv)

                # Load TrOCR model and processor
                processor = TrOCRProcessor.from_pretrained("/Users/Cs-Store/Desktop/intern2/yolov8/fine_tuned_trocr_khmer")
                ocr_model = VisionEncoderDecoderModel.from_pretrained("/Users/Cs-Store/Desktop/intern2/yolov8/fine_tuned_trocr_khmer")

                # Process and display the results
                for r in results:
                    im_array = r.plot(labels=False)
                    im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
                    pil_im = Image.fromarray(im)
                    draw = ImageDraw.Draw(pil_im)
                    font_path = "/Users/Cs-Store/Desktop/intern2/yolov8/Siemreap-Regular.ttf"
                    try:
                        font = ImageFont.truetype(font_path, 20)  # Adjust font size as needed
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

                        combined_text += generated_text + " "
                        draw.text((x1, y2 + 5), generated_text, font=font, fill=(0, 255, 0)) #draw text under box.

                    st.image(pil_im, caption="Detected Khmer Text Boxes with Recognized Text", use_column_width=True)
                    st.write(f"Combined Recognized Text: {combined_text}")

                st.success("Khmer text detection and recognition complete!")

if __name__ == "__main__":
    main()
