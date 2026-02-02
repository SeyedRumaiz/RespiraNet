import pytest
import os
import sys

# First add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIL import Image
import numpy as np
from app.model import DenseNetModel121


def create_dummy_image():
    return Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

@pytest.fixture
def model():
    weights_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../models/best_densenet.weights.h5")
    )
    model = DenseNetModel121(weights_path)
    return model

def test_preprocess(model):
    img = create_dummy_image()
    processed = model.preprocess(img)

    # Check shape
    assert processed.shape == (1,224,224,3)

    # Check range
    assert processed.max() <= 1.0
    assert processed.min() >= 0.0


def test_predict(model):
    img = create_dummy_image()
    label, prob = model.predict(img)
    assert isinstance(label, str)
    assert isinstance(prob, float)