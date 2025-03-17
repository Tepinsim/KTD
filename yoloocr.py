import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
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

                    boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                    results_text = [] #list to store results
                    for i, box in enumerate(boxes):  # Enumerate boxes for numbering
                        x1, y1, x2, y2 = box
                        cv2.putText(im, str(i + 1), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Add box number
                        cropped_image = image_cv[y1:y2, x1:x2]
                        cropped_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

                        # OCR processing
                        pixel_values = processor(images=cropped_pil, return_tensors="pt").pixel_values
                        generated_ids = ocr_model.generate(pixel_values)
                        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                        results_text.append(f"Box {i + 1}: Recognized Text: {generated_text}") #append results to list

                    st.image(im, caption="Detected Khmer Text Boxes", use_column_width=True)

                    for result in results_text: #display results
                        st.write(result)

                st.success("Khmer text detection and recognition complete!")

if __name__ == "__main__":
    main()
