import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout

# 1️Recreate architecture
def build_model():
    densenet_base = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    densenet_base.trainable = False  # freeze base layers

    model = Sequential([
        densenet_base,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # binary classification
    ])
    return model


def classify(image: Image.Image, model):
    # Resize
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    img_arr = np.asarray(image).astype(np.float32) / 255.0
    data = np.expand_dims(img_arr, axis=0)

    prediction = model.predict(data, verbose=0)

    # Since sigmoid output
    prob = float(prediction[0][0])

    if prob >= 0.5:
        return "PNEUMONIA", prob
    else:
        return "NORMAL", 1 - prob

def confidence_bar_chart(confidence):
    """
    Plots confidence for NORMAL vs PNEUMONIA
    """
    
    probs = {"NORMAL": 1-confidence, "PNEUMONIA": confidence}
    df = pd.DataFrame(list(probs.items()), columns=["Class", "Probability"])
    st.bar_chart(df.set_index("Class"))
