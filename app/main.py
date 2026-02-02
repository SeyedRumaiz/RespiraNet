from model import DenseNetModel121
from visualizations import ConfidenceBarChart
from respira_net import RespiraNetApp
import streamlit as st

model = DenseNetModel121("../models/best_densenet.weights.h5")
confidence_bar_chart = ConfidenceBarChart()
app = RespiraNetApp(model, confidence_bar_chart)

if not st.session_state.predicted:
    app.rending_landing()
else:
    app.render_results()
