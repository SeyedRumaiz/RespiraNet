import streamlit as st
from model import classify

# Setup config
st.set_page_config(page_title='RespiraNet', layout='wide')  # page_icon=""

# Initialize session state
if "predicted" not in st.session_state:
    st.session_state.predicted = False
    st.session_state.uploaded_file = False
    st.session_state.result = None
    st.session_state.confidence = None
    st.session_state.accuracy = None
    st.session_state.location = None

if not st.session_state.predicted:
    pass    # Navigation page

else:
    pass    # Results page


# Use the CSS file
def load_css(file):
    """
    Used to load a specific CSS file into the app
    
    Paramaters
        file: The CSS file being loaded
    """
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
