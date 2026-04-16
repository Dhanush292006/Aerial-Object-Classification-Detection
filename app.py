import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model
model = load_model("aerial_model.h5")

# Title
st.title("Aerial Object Classification & Detection")
st.write("Upload an image to classify it as Bird or Drone")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    # Output
    if prediction[0][0] > 0.5:
        st.success("🚁 Drone Detected")
    else:
        st.success("🐦 Bird Detected")
