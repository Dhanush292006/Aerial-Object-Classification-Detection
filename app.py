import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.title("🚀 Aerial Object Classification (Real AI Prediction)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    try:
        # Load YOLO model (real AI)
        model = YOLO("yolov8n.pt")

        results = model(image)
        detected_classes = results[0].names
        boxes = results[0].boxes

        prediction = "Unknown"

        for box in boxes:
            cls_id = int(box.cls[0])
            label = detected_classes[cls_id]

            if label == "bird":
                prediction = "🐦 Bird"
                break
            elif label in ["airplane", "kite"]:
                prediction = "🚁 Drone"
                break

        if prediction == "Unknown":
            prediction = "❓ Not Bird/Drone"

        st.success(f"✅ Prediction: {prediction}")
        st.image(results[0].plot(), caption="Detection Result")

    except Exception as e:
        st.error(f"Error: {e}")
