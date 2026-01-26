import numpy as np
from PIL import Image, ImageOps

def classify(image: Image.Image, model=None, class_names=['NORMAL', 'PNEUMONIA']):
    """
    Preprocess image and predict class
    """

    # Convert to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # Convert image to numpy array
    img_arr = np.asarray(image)

    # Normalize image to [-1, 1]
    norm_img_arr = (img_arr.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)
    data[0] = norm_img_arr
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence = float(prediction[0][index])

    return class_name, confidence
