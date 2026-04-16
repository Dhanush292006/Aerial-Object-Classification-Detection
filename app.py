import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Page settings
st.set_page_config(page_title="Aerial Object Detection", layout="centered")

st.title("🚀 Aerial Object Classification & Detection")
st.markdown("Upload an aerial image to classify **Bird or Drone**")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    try:
        # Load YOLO model (SAFE - no corruption)
        model = YOLO("yolov8n.pt")

        # Run detection
        results = model(image)[0]

        names = results.names
        boxes = results.boxes

        prediction = "❓ Unknown"
        confidence = 0

        # Check detected objects
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

        # Output
        if prediction == "❓ Unknown":
            st.warning("⚠️ No Bird/Drone detected")
        else:
            st.success(f"✅ Prediction: {prediction}")
            st.write(f"🔍 Confidence: {round(confidence * 100, 2)}%")

        # Show detection with bounding boxes
        st.image(results.plot(), caption="📦 Detection Result")

        # Confidence graph
        st.markdown("### 📊 Confidence Score")
        st.bar_chart({"Confidence": [confidence]})

    except Exception as e:
        st.error(f"Error: {e}")
