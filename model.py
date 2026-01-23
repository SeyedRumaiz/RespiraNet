import base64   # encode binary image data into textual format (embed an image into HTML)
import streamlit as st
from PIL import ImageOps, Image
import numpy as np

def set_background(image_file):
    """

    """

    with open(image_file, "rb") as f:
        img_data = f.read()
        b64_encoded = base64.b64encode(img_data).decode()

        # Full screen image in the root container
        style = f"""
            <style>
            .stApp {{
            background-image: url(data:image/png;base64,
            {b64_encoded});

            background-size: cover;

            }}

            </style>
        
        """
        st.markdown(style, unsafe_allow_html=True)  # allow raw HTML/CSS

def classify(image, model, class_names):
    """

    """

    # Convert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # Convert image to numpy array
    img_arr = np.asarray(image)

    # Normalize image
    norm_img_arr = (img_arr.astype(np.float32) / 127.5) - 1 # between [-1, 1]

    # Model output
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32) # batch size = 1, 3 channels
    data[0] = norm_img_arr

    # Run model
    prediction = model.predict(data)
    index = np.argmax(prediction)

    # Map index -> label
    class_name = class_names[index]
    confidence = prediction[0][index]

    return class_name, confidence
