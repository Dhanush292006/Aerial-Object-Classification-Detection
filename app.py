import streamlit as st
from PIL import Image

st.title("Aerial Object Detection (Bird vs Drone)")

# Step 1: App starts without model
st.write("Upload an image to detect objects")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if st.button("Run Detection"):
        try:
            from ultralytics import YOLO

            # Load model ONLY after button click
            model = YOLO("yolov8n.pt")

            results = model(image)
            st.image(results[0].plot(), caption="Detection Result")

        except Exception as e:
            st.error(f"Error: {e}")
