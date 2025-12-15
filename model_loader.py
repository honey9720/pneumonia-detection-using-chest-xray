# model_loader.py
import tensorflow as tf
from functools import lru_cache

@lru_cache(maxsize=1)
def load_model():
    model = tf.keras.models.load_model("model/xray_model.hdf5", compile=False)
    return model
