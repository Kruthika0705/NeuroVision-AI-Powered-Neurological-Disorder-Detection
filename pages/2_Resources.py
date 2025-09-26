# pages/2_Resources.py
import streamlit as st

st.set_page_config(page_title="Resources & About", page_icon="📚", layout="wide")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Navigation")
    st.page_link("Home.py", label="🏠 Home", icon="🏠")
    st.page_link("pages/1_MRI_Classifier.py", label="🧠 MRI Classifier", icon="🧠")
    st.page_link("pages/2_Resources.py", label="📚 Resources & About", icon="📚")

# ---------------- MAIN CONTENT ----------------
st.title("📚 Resources & About")

st.markdown(
    """
### 🔍 Project Overview
This platform combines **computer vision, deep learning, and language models** to support neurological research and education.

**Modules included:**
- 🧠 **MRI Classifier** – GoogLeNet-based classification with Grad-CAM heatmaps  
- 🎯 **CV Rehab Tracker** – Real-time hand tracking, adaptive exercise, and ML predictions  
- 💬 **NeuroChat** – Multilingual neurology Q&A chatbot (Gemini, voice input/output)  
"""
)

st.markdown("---")

st.subheader("⚖ Safety Notes")
st.warning(
    """
- This app is **for educational and research use only**.  
- It is **NOT a medical device** and does not provide diagnosis or treatment.  
- Always consult a qualified clinician for medical concerns.  
"""
)

st.markdown("---")

st.subheader("📖 Learning Resources")
st.markdown(
    """
- [Deep Learning for Medical Imaging (Stanford)](https://stanfordmlgroup.github.io/projects/medical/)
- [Mediapipe Hand Tracking](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [Google Gemini API](https://ai.google.dev/)
- [PyTorch Grad-CAM Tutorial](https://pytorch.org/tutorials/intermediate/vision_gradcam.html)
"""
)

st.markdown("---")

st.subheader("👩‍💻 Contributors")
st.markdown(
    """
- **Research & Development:** Your Team Name  
- **AI Models:** GoogLeNet (PyTorch), Random Forest, Gemini, Groq (optional)  
- **App Framework:** Streamlit  
"""
)

st.info("💡 Tip: Explore other pages using the sidebar navigation.")
