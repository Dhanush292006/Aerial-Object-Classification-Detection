import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.title("Aerial Object Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image")

    try:
        model = YOLO("yolov8nt.pt")   
        results = model(img)

        st.image(results[0].plot(), caption="Detection Result")

    except Exception as e:
        st.error(f"Error loading model: {e}")
