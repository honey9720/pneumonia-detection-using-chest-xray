from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from PIL import Image, ImageOps
import tensorflow as tf
from model_loader import load_model
import streamlit as st

app = FastAPI()
model = load_model()
class_names = ["Normal", "Pneumonia"]

@app.get("/")
def root():
    return {"message": "Pneumonia Detection API is running 🚀"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    size = (180, 180)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[np.newaxis, ...]

    predictions = model.predict(img)
    score = tf.nn.softmax(predictions[0])
    result = {
        "class": class_names[np.argmax(score)],
        "confidence": float(100 * np.max(score))
    }
    return result
