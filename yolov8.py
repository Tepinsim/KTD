import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

def main():
    st.title("Khmer Text Detection with YOLOv8")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Khmer Text"):
            with st.spinner("Detecting Khmer text..."):
                # Convert PIL Image to OpenCV format
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                # Load the YOLOv8 model
                model = YOLO("/Users/Cs-Store/Desktop/intern2/yolov8/173.pt")

                # Perform inference
                results = model(image_cv)

                # Process and display the results
                for r in results:
                    im_array = r.plot(labels=False)
                    im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
                    st.image(im, caption="Detected Khmer Text", use_column_width=True)
                st.success("Khmer text detection complete!")

if __name__ == "__main__":
    main()
