# Home.py
import streamlit as st
import base64

st.set_page_config(
    page_title="NeuroVision — MRI Classifier Suite",
    page_icon="🧠",
    layout="wide",
)

# --- Function to load local image as Base64 ---
def get_base64(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- Load background image from local machine ---
BG_PATH = "C:/Users/Kruthika/Downloads/mri_background.jpeg"
BG_IMAGE = get_base64(BG_PATH)

page_bg = f"""
<style>
/* Full-page background */
[data-testid="stAppViewContainer"] {{
  background: url("data:image/jpg;base64,{BG_IMAGE}") no-repeat center center fixed;
  background-size: cover;
}}

/* Dark overlay */
[data-testid="stAppViewContainer"]::before {{
  content: "";
  position: fixed;
  inset: 0;
  background: rgba(5, 10, 25, 0.55);
  z-index: 0;
}}

/* Ensure app content sits above overlay */
.block-container {{
  position: relative;
  z-index: 1;
}}

/* Hide sidebar */
section[data-testid="stSidebar"] {{
  display: none !important;
}}

/* Hide Streamlit's default menu/footer */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}

/* --- TOP NAVBAR --- */
.navbar {{
  display: flex;
  justify-content: center;
  gap: 30px;
  margin: 20px 0 40px 0;
}}

.navbar button {{
  background-color: #1E88E5 !important;
  color: white !important;
  font-size: 18px !important;
  padding: 10px 24px !important;
  border-radius: 10px !important;
  border: none !important;
  cursor: pointer !important;
  transition: background 0.3s !important;
}}

.navbar button:hover {{
  background-color: #1565C0 !important;
  color: white !important;
}}

/* --- Force ALL text white --- */
h1, h2, h3, h4, h5, h6, p, span, div, label {{
  color: white !important;
}}

/* --- CTA Buttons --- */
.stButton>button {{
  background-color: #1E88E5 !important;
  color: white !important;
  font-size: 16px !important;
  font-weight: bold !important;
  border-radius: 8px !important;
  padding: 10px 20px !important;
  border: none !important;
  transition: background 0.3s !important;
}}

.stButton>button:hover {{
  background-color: #1565C0 !important;
  color: white !important;
}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# --- TOP NAVIGATION BAR ---
# --- TOP NAVIGATION BAR ---
st.markdown('<div class="navbar">', unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)  # 👈 now we actually have 5 columns

with col1:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("Home.py")
with col2:
    if st.button("🧠 MRI Classifier", use_container_width=True):
        st.switch_page("pages/1_MRI_Classifier.py")
with col3:
    if st.button("📚 Resources & About", use_container_width=True):
        st.switch_page("pages/2_Resources.py")
with col4:
    if st.button("🤖 Chatbot", use_container_width=True):
        st.switch_page("pages/3_Chatbot.py")
with col5:
    if st.button("🤖 CV Rehab", use_container_width=True):
        st.switch_page("pages/4_CV_Rehab.py")        

with col6:
    if st.button("🎮 Games", use_container_width=True):
        st.switch_page("pages/5_Games.py")
st.markdown('</div>', unsafe_allow_html=True)


# --- Hero section ---
st.markdown(
    """
    <div style="padding: 8vh 2vw; text-align: center;">
      <h1 style="font-size: 56px; line-height:1.1; margin-bottom: 12px;">NeuroVision</h1>
      <p style="font-size: 22px; max-width: 860px; margin:auto; opacity: 0.95;">
        Upload an MRI, classify probable conditions, visualize Grad-CAM attention,
        simulate disease trajectories, compare treatments, and generate plain-language explanations.
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Main Call-to-Action buttons ---
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Get started", unsafe_allow_html=True)
    st.markdown("Jump straight into the classifier to upload an MRI and explore dashboards.", unsafe_allow_html=True)
    if st.button("🚀 Open MRI Classifier", use_container_width=True):
        st.switch_page("pages/1_MRI_Classifier.py")

with col2:
    st.markdown("### Learn more", unsafe_allow_html=True)
    st.markdown("Read about features, safety notes, and quick tips.", unsafe_allow_html=True)
    if st.button("ℹ️ Resources & About", use_container_width=True):
        st.switch_page("pages/2_Resources.py")

st.markdown("<hr style='border:1px solid white;'>", unsafe_allow_html=True)
st.markdown("✅ Use the <b>top navigation bar</b> to switch pages anytime.", unsafe_allow_html=True)
