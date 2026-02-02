import streamlit as st
import time
from PIL import Image
from visualizations import plot_pie_chart
import os


class RespiraNetApp:
    def __init__(self, model):
        self.model = model
        self.load_css(os.path.join(os.path.dirname(__file__), "styles.css"))
        self.setup_page()
        self.init_session_state()

    
    def init_session_state(self):
        # Initialize session state
        if "predicted" not in st.session_state:
            st.session_state.predicted = False
            st.session_state.uploaded_file = False
            st.session_state.result = None
            st.session_state.confidence = None


    def setup_page(self):
        st.set_page_config(page_title='RespiraNet', layout='wide')


    def rending_landing(self):
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
            st.markdown("<div class='step-container'><div class='step-node'>1</div><p class='internal-text'><b>Upload</b></p>" \
            "<p class='step-desc'>Upload a chest X-ray image for analysis.</p></div>", unsafe_allow_html=True)
        with a1:
            st.markdown("<div class='connector'>────➤</div>", unsafe_allow_html=True)
        with s2:
            st.markdown("<div class='step-container'><div class='step-node'>2</div><p class='internal-text'><b>Analyze</b></p>" \
            "<p class='step-desc'>AI model extracts deep visual features.</p></div>", unsafe_allow_html=True)
        with a2:
            st.markdown("<div class='connector'>────➤</div>", unsafe_allow_html=True)
        with s3:
            st.markdown("<div class='step-container'><div class='step-node'>3</div><p class='internal-text'><b>Results</b></p>" \
            "<p class='step-desc'>Prediction with confidence score is generated.</p></div>", unsafe_allow_html=True)


        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

        # Model overview section

        st.markdown("<div style='max-width:1000px; margin: auto;'>", unsafe_allow_html=True)
        with st.container():
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            c1, c2 = st.columns([1,1])

            with c1:
                st.markdown("### Engine Stats")
                st.markdown("- **Core:** DenseNet121 CNN\n- **Training Set:** 5,863 images" \
                "\n - **Output**: Binary Classification\n- **Activation**: Sigmoid")

            with c2:
                st.markdown("### Performance")
                st.image("newplot.png")
                
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
        

        # Upload file section
        st.markdown("<h2 style='text-align:center;'>Diagnostic Portal</h2>", unsafe_allow_html=True)
        _, center_col, _ = st.columns([1,2,1])
        with center_col:
            uploaded_file = st.file_uploader("Drop Scan to Initialize", type=["jpg","png","jpeg"])
            if uploaded_file:
                st.image(uploaded_file, width=500)
                if st.button("PREDICT"):
                    with st.spinner("Decoding Neural Layers..."):
                        img = Image.open(uploaded_file).convert("RGB")
                        class_name, confidence = self.model.run(img)
                        st.session_state.result = class_name
                        st.session_state.confidence = confidence
                        st.session_state.predicted = True
                        st.rerun()
        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#30363d'>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#555;'>RespiraNet © 2026</p>", unsafe_allow_html=True)


    def render_results(self):
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align:center; color:#00D4FF;'>Diagnostic Report</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'>Analysis Results</h2>", unsafe_allow_html=True)
        st.divider()

        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown("### Primary Classification")
            if st.session_state.result == "NORMAL":
                st.markdown(f"<h1 style='color:#00E676; font-size:4rem;'>{st.session_state.result}</h1>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h1 style='color:#FF0000; font-size:4rem;'>{st.session_state.result}</h1>", unsafe_allow_html=True)
            st.write("Scan analyzed by RespiraNet.")
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown("### Confidence Score")
            confidence = st.session_state.confidence * 100
            st.markdown(f"<h1 style='color:#00D4FF; font-size:4rem;'>{confidence:.2f}%", unsafe_allow_html=True)
            st.progress(confidence/100)
            if st.session_state.confidence >= 0.8:
                st.write("The model is highly confident in this classification")
            elif st.session_state.confidence >= 0.6:
                st.write("The model is moderately confident in this classification.")
            else:
                st.write("the model is uncertain. Consider further clinical evaluation.")
            st.markdown("</div>", unsafe_allow_html=True)
        plot_pie_chart(st.session_state.confidence)
        report_text = f"""
        RespiraNet Diagnostic Report
        ----------------------------

        Result:         {st.session_state.result}
        Confidence:     {st.session_state.confidence * 100:.2f}%
        Timestamp:      {time.strftime('%Y-%m-%d %H:%M:%S')}
        """
        st.download_button(
            label="📄 Export Report",
            data=report_text,
            file_name="respiranet_report.txt",
            mime="text/plain",
            key="export-btn"
        )

        with st.container():
            left, _ = st.columns([1,5])
            with left:
                if st.button("← New Scan"):
                    st.session_state.predicted = False
                    st.session_state.uploaded_file = None
                    st.session_state.result = None
                    st.session_state.confidence = 0
                    st.rerun()
        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#30363d'>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#555;'>RespiraNet © 2026</p>", unsafe_allow_html=True)


    def load_css(self, file):
        """
        Used to load a specific CSS file into the app
        
        Paramaters
            file: The CSS file being loaded
        """
        with open(file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
