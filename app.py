import streamlit as st
from PIL import Image

st.title("Aerial Object Detection (Bird vs Drone)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image")

    try:
        from ultralytics import YOLO
        
        # Use lightweight model (NO crash)
        model = YOLO("yolov8n.pt")

        results = model(img)
        st.image(results[0].plot(), caption="Detection Result")

    except Exception as e:
        st.error(f"Model Error: {e}")
