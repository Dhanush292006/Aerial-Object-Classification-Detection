import streamlit as st
from PIL import Image

st.title("Aerial Object Classification")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image")

    # Demo output (since TF removed)
    st.success("Prediction working (UI Demo Mode)")
