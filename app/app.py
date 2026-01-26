import streamlit as st
from model import classify

# Setup config
st.set_page_config(page_title='RespiraNet', layout='wide')

# Initialize session state
if "predicted" not in st.session_state:
    st.session_state.predicted = False
    st.session_state.uploaded_file = False
    st.session_state.result = None
    st.session_state.confidence = None

# Use the CSS file
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

