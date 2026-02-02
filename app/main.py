from model import DenseNetModel121
from respiranet import RespiraNetApp
import streamlit as st

model = DenseNetModel121("../models/best_densenet.weights.h5")
app = RespiraNetApp(model)

if not st.session_state.predicted:
    app.rending_landing()
else:
    app.render_results()
