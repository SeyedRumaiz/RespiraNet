import numpy as np
from PIL import Image, ImageOps
from typing import final
from abc import ABC, abstractmethod
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout


class Model(ABC):
    def __init__(self):
        self.__model = None

    @final
    def run(self, image: Image.Image):
        if self.__model is None:
            self.__model = self.build_model()
        data = self.preprocess(image)
        return self.predict(data)

    @abstractmethod
    def preprocess(self, image: Image.Image) -> np.ndarray:
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def predict(self, data: np.ndarray):
        pass


    @property
    def model(self):
        return self.__model


class DenseNetModel121(Model):
    def __init__(self, weights_path: str) -> None:
        super().__init__()
        self.weights_path = weights_path


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

        if hasattr(self, "weights_path") and self.weights_path:
            model.load_weights(self.weights_path)

        return model
    

    def predict(self, data: np.ndarray):
        prediction = self.model.predict(data, verbose=0)
        prob = float(prediction[0][0])

        if prob >= 0.5:
            return "PNEUMONIA", prob
        else:
            return "NORMAL", 1 - prob

