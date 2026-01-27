import numpy as np
from PIL import Image, ImageOps

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
