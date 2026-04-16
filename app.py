import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

st.set_page_config(page_title="Aerial Detection", layout="centered")

st.title("🚀 Aerial Object Classification & Detection")
st.markdown("Upload an aerial image to classify **Bird or Drone**")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    try:
        # Load YOLO model
        model = YOLO("yolov8n.pt")

        results = model(image)[0]

        # Extract detections
        names = results.names
        boxes = results.boxes

        prediction = "❓ Unknown"
        confidence = 0

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls_id]

            if label == "bird":
                prediction = "🐦 Bird"
                confidence = conf
                break
            elif label in ["airplane", "kite"]:
                prediction = "🚁 Drone"
                confidence = conf
                break

        # If nothing detected
        if prediction == "❓ Unknown":
            st.warning("No Bird/Drone detected")
        else:
            st.success(f"✅ Prediction: {prediction}")
            st.write(f"🔍 Confidence: {round(confidence*100,2)}%")

        # 🔲 Show detection image (bounding boxes)
        st.image(results.plot(), caption="📦 Detection Result")

        # 📊 Optional graph (confidence bar)
        st.markdown("### 📊 Confidence Visualization")
        st.bar_chart({
            "Confidence": [confidence]
        })

    except Exception as e:
        st.error(f"Error: {e}")
