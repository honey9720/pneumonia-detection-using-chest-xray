import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from model_loader import load_model

# Load model
with st.spinner("Loading model..."):
    model = load_model()

# Define classes
class_names = ["Normal", "Pneumonia"]

st.title("🩻 Pneumonia Identification System")

file = st.file_uploader("Upload a chest scan file", type=["jpg", "jpeg", "png"])

def import_and_predict(image_data, model):
    size = (180, 180)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[np.newaxis, ...]
    prediction = model.predict(img)
    return prediction

if file is not None:
    image = Image.open(file)
    st.image(image, width="stretch")
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])

    st.write("Confidence scores:", score.numpy())
    st.success(
        f"This image most likely belongs to **{class_names[np.argmax(score)]}** "
        f"with a **{100 * np.max(score):.2f}%** confidence."
    )
else:
    st.info("Please upload an image file.")
