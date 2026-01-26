import streamlit as st
import time
from model import classify
from PIL import Image

# Setup config
st.set_page_config(page_title='RespiraNet', layout='wide')  # page_icon=""

# Use the CSS file
def load_css(file):
    """
    Used to load a specific CSS file into the app
    
    Paramaters
        file: The CSS file being loaded
    """
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("/Users/seyedrumaiz/Documents/Pneumonia/app/styles.css")

# Initialize session state
if "predicted" not in st.session_state:
    st.session_state.predicted = False
    st.session_state.uploaded_file = False
    st.session_state.result = None
    st.session_state.confidence = None
    st.session_state.accuracy = None
    st.session_state.location = None

if not st.session_state.predicted:

    st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

    # Introduction section
    st.markdown("<h1 class='brand-title'>RespiraNet</h1>", unsafe_allow_html=True)

    st.markdown("<h3 class='hero-title'>AI-Based Pneumonia Detection System</h3>", unsafe_allow_html=True)

    st.markdown("<p class='intro-text'>Next-generation diagnostic framework</p>", unsafe_allow_html=True)

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

    # How it works section
    st.markdown("<h2 class='how-it-works'>How It Works</h2>", unsafe_allow_html=True)

    s1, a1, s2, a2, s3 = st.columns([2,2,2,2,2])

    with s1:
        st.markdown("<div class='step-node'>1</div><p><b>Upload</b></p>", unsafe_allow_html=True)
    with a1:
        st.markdown("<div class='connector'>────➤</div>", unsafe_allow_html=True)
    with s2:
        st.markdown("<div class='step-node'>2</div><p><b>Analyze</b></p>", unsafe_allow_html=True)
    with a2:
        st.markdown("<div class='connector'>────➤</div>", unsafe_allow_html=True)
    with s3:
        st.markdown("<div class='step-node'>3</div><p><b>Results</b></p>", unsafe_allow_html=True)


    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

    # Upload file section
    uploaded_file = st.file_uploader(
        "Upload Chest X-ray",
        type = ['jpg', 'png', 'jpeg']
    )

    # If user uploaded the file
    if uploaded_file:
        st.image(uploaded_file, use_column_width=True)
        
        if st.button("PREDICT"):
            with st.spinner("Analyzing scan..."):
                time.sleep(2)
                img = Image.open(uploaded_file).convert("RGB")
                result, confidence = classify(img)

                st.session_state.result = result
                st.session_state.confidence = confidence
                st.session_state.predicted = True
                st.rerun()

else:
    pass    # Results page
