import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load model
model = tf.keras.models.load_model("deepfake_model.h5")

st.set_page_config(page_title="DeepFake Detection", layout="centered")

st.title("🕵️ DeepFake Detection System")
st.write("Upload an image to check whether it is **Fake or Real**")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.reshape(img, (1, 128, 128, 3))

    if st.button("🔍 Predict"):
        prediction = model.predict(img)

        if prediction[0][0] < 0.2:
            st.error("❌ FAKE IMAGE DETECTED")
        elif prediction[0][0] >= 0.8:
            st.success("✅ REAL IMAGE")

        st.write("Prediction Score:", prediction[0][0])
