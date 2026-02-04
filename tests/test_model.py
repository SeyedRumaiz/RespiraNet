import pytest
import os
import sys
import streamlit as st

# First add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIL import Image
import numpy as np
from app.model import DenseNetModel121
from app.respiranet import RespiraNetApp

class DummyModel:
    def run(self, img):
        return "NORMAL", 0.7
    

class DummyConfidencePlot:
    def plot(self, value):
        return value
    
@pytest.fixture
def dummy_app():
    return RespiraNetApp(DummyModel(), DummyConfidencePlot())

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

def test_init_session_state(dummy_app):
    st.session_state.clear()
    dummy_app.init_session_state()
    assert st.session_state.predicted == False
    assert st.session_state.uploaded_file == False
    assert st.session_state.result == None
    assert st.session_state.confidence == None


def test_render_results(dummy_app):
    st.session_state.predicted = True
    st.session_state.result = "NORMAL"
    st.session_state.confidence = 0.75
    dummy_app.render_results()

def test_run(model):
    img = create_dummy_image()
    label, confidence = model.run(img)
    assert label in ["NORMAL", "PNEUMONIA"]
    assert 0 <= confidence <= 1

def test_session_reset(dummy_app):
    st.session_state.predicted = True
    st.session_state.result = "PNEUMONIA"
    st.session_state.confidence = 0.9

    # Simulate reset
    dummy_app.init_session_state()
    st.session_state.predicted = False
    st.session_state.result = None
    st.session_state.confidence = 0.0

    assert st.session_state.predicted == False
    assert st.session_state.result == None
    assert st.session_state.confidence == 0.0
