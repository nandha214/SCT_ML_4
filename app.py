import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 1. Page Configuration (Sets the browser tab title and link preview)
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="👋",
    layout="centered"
)

# 2. Gesture class labels mapping
CLASSES = [
    "01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb", 
    "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"
]

FRIENDLY_NAMES = {
    "01_palm": "Palm (Open Hand)",
    "02_l": "L Shape",
    "03_fist": "Fist",
    "04_fist_moved": "Fist Moved",
    "05_thumb": "Thumb",
    "06_index": "Index Finger",
    "07_ok": "OK Sign",
    "08_palm_moved": "Palm Moved",
    "09_c": "C Shape",
    "10_down": "Down Gesture"
}

# 3. Load Trained Model
@st.cache_resource
def load_trained_model():
    model_path = os.path.join('model', 'gesture_model.keras')
    return tf.keras.models.load_model(model_path)

model = load_trained_model()

# 4. User Interface Header
st.title("👋 Hand Gesture Recognition")
st.write("Upload a hand gesture image and let the CNN predict the gesture.")

# Custom Pinned Information Box matching your design
info_html = """
<div style="
    background-color: #f0f4f9;
    padding: 16px 20px;
    border-radius: 8px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    border: 1px solid #dbe2ea;
">
    <div style="font-size: 24px; margin-right: 14px;">📌</div>
    <div style="color: #0b57d0; font-size: 15px; line-height: 1.4;">
        For the best results, upload images that are similar to the training dataset: grayscale images with a similar background, hand position, and image style.
    </div>
</div>
"""
st.markdown(info_html, unsafe_allow_html=True)

# 5. File Uploader
uploaded_file = st.file_uploader(
    "Upload a hand gesture image",
    type=["png", "jpg", "jpeg"]
)

# 6. Prediction Execution
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.write("### Uploaded Image:")
    st.image(image, width=280)
    
    with st.spinner("Classifying gesture..."):
        # Preprocessing: Grayscale, resize to 128x128, normalize
        img_gray = image.convert('L')
        img_resized = img_gray.resize((128, 128))
        img_array = np.array(img_resized, dtype="float32") / 255.0
        img_input = img_array.reshape(1, 128, 128, 1)
        
        # Make prediction
        prediction = model.predict(img_input)
        class_idx = np.argmax(prediction[0])
        predicted_raw = CLASSES[class_idx]
        gesture_name = FRIENDLY_NAMES.get(predicted_raw, predicted_raw)
        confidence = float(prediction[0][class_idx]) * 100
        
        # Display Result
        st.success(f"**Predicted Gesture:** {gesture_name}")
        st.info(f"**Confidence:** {confidence:.2f}%")