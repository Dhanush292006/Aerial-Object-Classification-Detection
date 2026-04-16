import streamlit as st
from PIL import Image

# Page config
st.set_page_config(page_title="Aerial Object Classification", layout="centered")

# Title
st.title("🚀 Aerial Object Classification (Bird vs Drone)")
st.markdown("### 📌 Upload an aerial image to classify Bird or Drone")

# Upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Simulated prediction (Demo Mode)
    import random

prediction = random.choice(["🐦 Bird", "🚁 Drone"])

st.success(f"✅ Prediction: {prediction}")
    # Info for clarity
    st.info("⚠️ Note: Model inference is demonstrated in Colab. This is a deployment UI preview due to cloud limitations.")

# Footer
st.markdown("---")
st.markdown("👨‍💻 Developed for Aerial Surveillance AI Project")
