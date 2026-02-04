from model import DenseNetModel121
from respiranet import RespiraNetApp
import streamlit as st

app = RespiraNetApp()

if not st.session_state.predicted:
    app.rending_landing()
else:
    app.render_results()
