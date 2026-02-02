import numpy as np
from PIL import Image, ImageOps
from abc import ABC, abstractmethod
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout


class Model(ABC):
    @abstractmethod
    def preprocess(self, image: Image.Image) -> np.ndarray:
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def predict(self, image: Image.Image):
        pass


class DenseNetModel121(Model):
    def __init__(self, weights_path: str) -> None:
        self.__model = self.build_model()
        self.__model.load_weights(weights_path)


    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        img_arr = np.asarray(image).astype(np.float32) / 255.0
        return np.expand_dims(img_arr, axis=0)
    

    def build_model(self):
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
    

    def predict(self, image: Image.Image):
        data = self.preprocess(image)
        prediction = self.__model.predict(data, verbose=0)
        prob = float(prediction[0][0])

        if prob >= 0.5:
            return "PNEUMONIA", prob
        else:
            return "NORMAL", 1 - prob
        

    @property
    def model(self):
        return self.__model
