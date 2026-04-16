import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")

st.title("Aerial Object Detection (Bird vs Drone)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image")

    # Run YOLO
    results = model(img)

    # Show output
    st.image(results[0].plot(), caption="Detection Result")
