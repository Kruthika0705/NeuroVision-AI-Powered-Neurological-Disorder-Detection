# pages/1_MRI_Classifier.py
import os
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import GoogLeNet_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import cv2
import streamlit as st
import plotly.graph_objects as go

# Optional: Groq LLM
try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except Exception:
    _GROQ_AVAILABLE = False

st.set_page_config(
    page_title="MRI Classifier + Disease Dashboards + LLM",
    page_icon="🧠",
    layout="wide",
)

# -----------------------------
# 🔥 Function: Set background image from local system
# -----------------------------
def set_bg_from_local(image_path: str):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}
        [data-testid="stSidebar"] {{
            background-color: rgba(255,255,255,0.95) !important;
        }}
        h1, h2, h3, h4, h5, h6, p, span, div, label {{
            color: black !important;
        }}
        .stButton>button {{
            background-color: #1E88E5 !important;
            color: white !important;
            border-radius: 8px !important;
            font-weight: bold !important;
            border: none !important;
        }}
        .stButton>button:hover {{
            background-color: #1565C0 !important;
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ✅ Change this to your local image path
set_bg_from_local("C:/Users/Kruthika/Downloads/mri2.jpeg")

# -----------------------------
# Sidebar navigation
# -----------------------------
with st.sidebar:
    st.header("Navigation")
    st.page_link("Home.py", label="🏠 Home", icon="🏠")
    st.page_link("pages/1_MRI_Classifier.py", label="🧠 MRI Classifier", icon="🧠")
    st.page_link("pages/2_Resources.py", label="📚 Resources & About", icon="📚")
    st.markdown("---")

st.title("🧠 MRI Classifier + Disease Dashboards + LLM")
st.caption("Upload an MRI, classify the condition, visualize Grad-CAM, simulate disease-specific trajectories, compare treatments, and get a plain-language explanation.")

# -----------------------------
# Utilities (Model + Grad-CAM)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str, num_classes: int = 5):
    device = torch.device("cpu")
    model = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    class_names = checkpoint.get(
        "class_names",
        ["Alzheimers", "Glioma", "Meningioma", "Normal", "Parkinson"],
    )
    return model, device, class_names

preprocess = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def pil_to_cv(img_pil: Image.Image):
    arr = np.array(img_pil.convert("RGB"))
    return arr[:, :, ::-1].copy()

def overlay_cam_on_image(img_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    cam_uint8 = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    blended = cv2.addWeighted(heatmap, alpha, img_bgr, 1 - alpha, 0)
    return blended

class GoogLeNetGradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fh = target_layer.register_forward_hook(self._save_activation)
        self.bh = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, inp: torch.Tensor, class_idx: int = None):
        logits = self.model(inp)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[:, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)
        grads = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (grads * self.activations).sum(dim=1, keepdim=False)
        cam = F.relu(cam)
        cam_np = cam.squeeze(0).cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() + 1e-8)
        return cam_np, class_idx, logits.softmax(dim=1).detach().cpu().numpy()

    def remove(self):
        self.fh.remove()
        self.bh.remove()

# -----------------------------
# 🧪 LLM Explanation + Nearby Hospitals (Groq + OSM + Excel + Streamlit Map)
# -----------------------------
def generate_llm_explanation(disease_display: str, selected_treatments: list[str], model_outputs: dict | None = None):
    """
    Uses Groq with a hardcoded API key if available. 
    Falls back to a simple summary if Groq API call fails.
    Shows nearby hospitals (from OpenStreetMap + Excel) relative to user’s real-time geolocation.
    Renders explanation first, then map, then hospital details.
    """
    import json
    import pandas as pd
    import requests
    import streamlit as st
    import folium
    from streamlit_folium import st_folium
    from streamlit_javascript import st_javascript
    from geopy.distance import geodesic
    import time

    # 🔑 Hardcoded Groq API key
    api_key = ""

    # Check if Groq SDK is installed
    try:
        from groq import Groq
        _GROQ_AVAILABLE = True
    except ImportError:
        _GROQ_AVAILABLE = False
        print("Groq SDK not installed. Install with: pip install groq")

    treatments_text = ", ".join(selected_treatments) if selected_treatments else "no specific treatment selected"
    model_context = f"\nModel outputs (if any): {json.dumps(model_outputs, default=str)}" if model_outputs else ""

    explanation_text = None

    # --- Try Groq API first ---
    if _GROQ_AVAILABLE and api_key:
        try:
            client = Groq(api_key=api_key)
            prompt = f"""
You are writing a short, patient-friendly MRI report summary.

Condition predicted: {disease_display}
Selected treatments being considered: {treatments_text}.{model_context}

Write 6-8 sentences that:
1) Briefly state what the condition is in plain language.
2) Explain what makes this scan look different from a normal brain scan (mention 2–3 key features in simple terms).
3) List common symptoms patients might notice.
4) Describe the typical progression over time if untreated.
5) Explain how the selected treatment(s) generally help with recovery or symptom control.
6) Add one or two practical tips.

Style:
- Be clear, calm, non-alarming; avoid jargon and absolutes.
- Use “suggests” or “is consistent with”.
- Keep it concise (~110–160 words). End with a practical next step.
"""
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=500,
            )
            choice = resp.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                explanation_text = choice.message.content
            else:
                explanation_text = "LLM returned a response that couldn't be parsed."
        except Exception as e:
            print(f"⚠ Groq API call failed: {e}")

    # --- Fallback if Groq fails ---
    if not explanation_text:
        base = {
            "Alzheimer's": (
                "This pattern suggests changes commonly seen in Alzheimer’s disease. "
                "Without treatment, memory ability usually decreases over years; medications and lifestyle changes can slow decline. "
                "Treatments aim to preserve independence longer; combine medicine with exercise and social engagement."
            ),
            "Parkinson's": (
                "This suggests changes often associated with Parkinson’s disease. Movement (walking, balance, fine motor skills) may worsen slowly. "
                "Medications and deep brain stimulation can reduce symptoms. Work with a movement-disorders specialist."
            ),
            "Glioma": (
                "The imaging pattern is consistent with a glioma. Tumor size can grow if untreated. "
                "Treatment (surgery, radiation, chemotherapy) helps slow growth. Consult neuro-oncology for tailored care."
            ),
            "Meningioma": (
                "This looks like a meningioma. Many grow slowly; some are removed surgically or treated with radiosurgery. "
                "If surgery is done, symptoms may improve. Discuss risks with a neurosurgeon."
            ),
            "Normal": (
                "The MRI appears within expected ranges. Maintain brain-healthy habits: exercise, diet, sleep, and regular checkups."
            )
        }
        explanation_text = base.get(
            disease_display,
            f"This indicates {disease_display}. Please consult a specialist."
        ) + f" Current treatment choices: {treatments_text}."

    # --- Show Report First ---
    st.subheader("🧾 MRI Report Summary")
    st.write(explanation_text)

    # --- Detect Location (Browser GPS → fallback to IP) ---
    coords = st_javascript(
        "await new Promise(r => navigator.geolocation.getCurrentPosition(p => r([p.coords.latitude, p.coords.longitude]), e => r(null)))"
    )

    if not coords:
        try:
            ip_resp = requests.get("https://ipapi.co/json/")
            ip_data = ip_resp.json()
            coords = [ip_data["latitude"], ip_data["longitude"]]
        except Exception:
            coords = None

    if coords:
        lat, lon = coords

        # Reverse geocode
        try:
            nominatim_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=16&addressdetails=1"
            resp = requests.get(nominatim_url, headers={'User-Agent': 'Streamlit App'})
            display_location = resp.json().get("display_name", "Unknown Location")
        except Exception:
            display_location = "Unknown Location"

        st.success(f"📍 Location detected: {display_location}")

        # --- Overpass Query for Hospitals ---
        overpass_url = "http://overpass-api.de/api/interpreter"
        query = f"""
        [out:json];
        node(around:5000,{lat},{lon})[amenity=hospital];
        out;
        """
        try:
            response = requests.get(overpass_url, params={"data": query}, timeout=60)
            data = response.json()
        except Exception:
            data = {"elements": []}

        # --- Map ---
        st.subheader("🗺 Nearby Neuro & Cancer Hospitals")
        m = folium.Map(location=[lat, lon], zoom_start=14)

        folium.Marker(
            [lat, lon],
            popup=f"📍 You are here ({display_location})",
            icon=folium.Icon(color="blue", icon="user")
        ).add_to(m)

        st_folium(m, width=600, height=400)  # smaller map

        # --- Hospital Details ---
        st.markdown("### 🏥 Hospital Details")
        count = 0
        for element in data.get("elements", []):
            h_lat, h_lon = element["lat"], element["lon"]
            tags = element.get("tags", {})
            name = tags.get("name", "Unnamed Hospital")

            if any(keyword in name.lower() for keyword in ["neuro", "cancer", "oncology", "brain"]):
                count += 1
                distance_km = round(geodesic((lat, lon), (h_lat, h_lon)).km, 2)
                address = tags.get("addr:full", "Address not available")
                phone = tags.get("phone") or tags.get("contact:phone", "Not available")
                website = tags.get("website", "Not available")
                directions_url = f"https://www.google.com/maps/dir/?api=1&origin={lat},{lon}&destination={h_lat},{h_lon}"

                st.markdown(
                    f"{count}. {name}\n\n"
                    f"📏 {distance_km} km away\n\n"
                    f"📍 {address}\n\n"
                    
                    f"[🗺 Get Directions]({directions_url})\n\n---"
                )

        if count == 0:
            st.warning("❌ No specialized Neuro or Cancer hospitals found nearby.")

    else:
        st.warning("⚠ Could not detect location. Please allow location access.")

    return explanation_text



# -----------------------------
# Sidebar: Settings
# -----------------------------
with st.sidebar:
    st.header("Settings")
    default_model_path = "C:/Users/Kruthika/Downloads/np/np/GN82.pth"
    model_path = st.text_input("Model .pth path", value=default_model_path)
    num_classes = st.number_input("Number of classes", min_value=2, max_value=50, value=5, step=1)
    run_demo = st.checkbox("Use demo mode (no model file required)", value=False)
    st.caption("For Groq LLM, set the GROQ_API_KEY environment variable.")

# -----------------------------
# The rest of your code (upload, inference, Grad-CAM, plots, explanation, notes)
# -----------------------------
# ⚠ Keep everything else from your original file unchanged
# Upload MRI
uploaded = st.file_uploader("Upload MRI image (PNG/JPG)", type=["png", "jpg", "jpeg"])

# Load model (or demo)
model = None
class_names = ["Alzheimers", "Glioma", "Meningioma", "Normal", "Parkinson"]
device = torch.device("cpu")
model_loaded_ok = False

if not run_demo:
    if os.path.exists(model_path):
        try:
            model, device, class_names = load_model(model_path, num_classes=num_classes)
            model_loaded_ok = True
        except Exception as e:
            st.error(f"Failed to load model: {e}")
    else:
        st.warning("Model path not found. Enable 'demo mode' to try the UI without weights.")
else:
    model_loaded_ok = True

# Inference + Grad-CAM + Dashboards
if uploaded and model_loaded_ok:
    pil_img = Image.open(uploaded).convert("RGB")
    orig_w, orig_h = pil_img.size

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Current MRI")
        st.image(pil_img, caption="Uploaded MRI", width=350)  # reduced size

    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    if run_demo:
        probs = np.array([[0.72, 0.06, 0.05, 0.12, 0.05]])
        pred_idx = int(np.argmax(probs))
        cam = cv2.GaussianBlur(np.random.rand(224, 224).astype(np.float32), (31, 31), 5)
    else:
        target_layer = model.inception5b
        cam_engine = GoogLeNetGradCAM(model, target_layer)
        cam, pred_idx, probs = cam_engine(input_tensor)
        cam_engine.remove()

    internal_to_display = {
        "Alzheimers": "Alzheimer's",
        "Parkinson": "Parkinson's",
        "Glioma": "Glioma",
        "Meningioma": "Meningioma",
        "Normal": "Normal"
    }
    pred_name_internal = class_names[pred_idx] if pred_idx < len(class_names) else f"Class {pred_idx}"
    pred_name_display = internal_to_display.get(pred_name_internal, pred_name_internal)
    pred_prob = float(probs[0, pred_idx])

    img_bgr = pil_to_cv(pil_img)
    cam_resized = cv2.resize(cam, (orig_w, orig_h))
    overlay = overlay_cam_on_image(img_bgr, cam_resized, alpha=0.45)
    overlay_rgb = overlay[:, :, ::-1]

    with c2:
        st.subheader("Highlighted Regions (Grad-CAM)")
        st.image(overlay_rgb, caption=f"Prediction: {pred_name_display} (p={pred_prob:.2f})", width=350)  # reduced size

    # 🔥 Highlight Classification Result with CSS
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, #1E88E5, #1565C0);
            padding: 16px;
            border-radius: 12px;
            text-align: center;
            margin: 20px 0;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
        ">
            <h2 style="color:white; margin:0; font-size:28px;">
                ✅ Classified as <b>{pred_name_display}</b><br>
                Confidence: {pred_prob:.2f}
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.subheader("📊 Treatment Simulation & Comparison")

    treatment_options = {
        "Alzheimer's": ["None", "Cholinesterase inhibitors", "lecanemab(strong)", "Lifestyle (Exercise + Diet)", "Combined"],
        "Parkinson's": ["None", "Dopamine Therapy", "Deep Brain Stimulation (DBS)", "Physical Therapy", "Combined"],
        "Glioma": ["None", "Chemo/Radiation", "Targeted Therapy", "Clinical Trial", "Combined"],
        "Meningioma": ["None", "Surgery", "Radiosurgery", "Watchful Waiting", "Combined"],
        "Normal": ["None", "Preventive (Lifestyle)"]
    }

    treatments_to_compare = st.multiselect(
        "Select treatments to compare (pick 1+)",
        treatment_options.get(pred_name_display, ["None"]),
        default=[treatment_options.get(pred_name_display, ["None"])[0]]
    )

    cognition_modifiers = {
        "None": 1.0, "Cholinesterase inhibitors": 0.5, "lecanemab(strong)": 0.3, "Lifestyle (Exercise + Diet)": 0.75, "Combined": 0.2, "Preventive (Lifestyle)": 0.8
    }
    motor_modifiers = {
        "None": 1.0, "Dopamine Therapy": 0.5, "Deep Brain Stimulation (DBS)": 0.27, "Physical Therapy": 0.8, "Combined": 0.17
    }
    tumor_modifiers = {
        "None": 1.0, "Chemo/Radiation": 0.25, "Targeted Therapy": 0.4, "Clinical Trial": 0.6, "Combined": 0.1, "Radiosurgery": 0.05, "Surgery": 0.0, "Watchful Waiting": 1.0
    }

    def plot_cognition_comparison(selected):
        years = np.arange(0, 11)
        fig = go.Figure()
        for t in selected:
            rate = 0.20 * cognition_modifiers.get(t, 1.0)
            values = np.exp(-rate * years) * 100
            fig.add_trace(go.Scatter(x=years, y=values, mode="lines+markers", name=t))
        fig.update_layout(title="Cognitive Trajectory (Memory ability, 0-100%)",
                          xaxis_title="Years", yaxis_title="Memory ability (%)", yaxis=dict(range=[0,100]))
        st.plotly_chart(fig, use_container_width=True)

    def plot_motor_comparison(selected):
        years = np.arange(0, 11)
        fig = go.Figure()
        for t in selected:
            rate = 0.30 * motor_modifiers.get(t, 1.0)
            impairment = (1 - np.exp(-rate * years))
            values = (1 - impairment) * 100
            fig.add_trace(go.Scatter(x=years, y=values, mode="lines+markers", name=t))
        fig.update_layout(title="Motor Function Trajectory (Movement ability, 0-100%)",
                          xaxis_title="Years", yaxis_title="Movement ability (%)", yaxis=dict(range=[0,100]))
        st.plotly_chart(fig, use_container_width=True)

    def plot_tumor_comparison_and_survival(selected):
        years = np.arange(0, 11)
        fig_g = go.Figure()
        for t in selected:
            mod = tumor_modifiers.get(t, 1.0)
            if pred_name_display == "Meningioma" and t == "Surgery":
                growth = np.zeros_like(years)
            else:
                rate = 0.20 * mod
                raw = np.exp(rate * years)
                growth = (raw - raw.min()) / (raw.max() - raw.min()) * 100
            fig_g.add_trace(go.Scatter(x=years, y=growth, mode="lines+markers", name=t))
        fig_g.update_layout(title=f"{pred_name_display} Progression (Tumor size, 0-100%)",
                            xaxis_title="Years", yaxis_title="Tumor size (%)", yaxis=dict(range=[0,100]))
        st.plotly_chart(fig_g, use_container_width=True)

        fig_s = go.Figure()
        for t in selected:
            base_hazard = 0.15
            hazard = base_hazard * tumor_modifiers.get(t, 1.0)
            surv = np.exp(-hazard * years) * 100
            fig_s.add_trace(go.Scatter(x=years, y=surv, mode="lines+markers", name=t))
        fig_s.update_layout(title="Illustrative Survival Probability (%)",
                            xaxis_title="Years", yaxis_title="Survival probability (%)", yaxis=dict(range=[0,100]))
        st.plotly_chart(fig_s, use_container_width=True)

    if pred_name_display == "Alzheimer's":
        if treatments_to_compare: plot_cognition_comparison(treatments_to_compare)
    elif pred_name_display == "Parkinson's":
        if treatments_to_compare: plot_motor_comparison(treatments_to_compare)
    elif pred_name_display in ["Glioma", "Meningioma"]:
        if treatments_to_compare: plot_tumor_comparison_and_survival(treatments_to_compare)
    else:
        st.info("Normal: no disease-specific curves to compare. Preventive lifestyle helps maintain health.")

    st.markdown("---")
    st.subheader("Current vs Predicted (Grad-CAM)")

    years_cam = st.slider("Years into future for Grad-CAM projection", min_value=0, max_value=10, value=0)
    colA, colB = st.columns(2)

    cam_treatment_effectiveness = {
        "None": 1.0, "Cholinesterase inhibitors": 0.6, "lecanemab(strong)": 0.4, "Lifestyle (Exercise + Diet)": 0.75, "Combined": 0.35,
        "Dopamine Therapy": 0.6, "Deep Brain Stimulation (DBS)": 0.45, "Physical Therapy": 0.8,
        "Chemo/Radiation": 0.3, "Targeted Therapy": 0.5, "Clinical Trial": 0.7,
        "Radiosurgery": 0.2, "Surgery": 0.1, "Preventive (Lifestyle)": 0.85, "Watchful Waiting": 1.0
    }

    if pred_name_display == "Normal":
        overlay_future = overlay_rgb
    else:
        base_scale = np.interp(years_cam, [0, 10], [1.0, 1.6])
        treatment_eff = min([cam_treatment_effectiveness.get(t, 1.0) for t in treatments_to_compare]) if treatments_to_compare else 1.0
        scale = base_scale * treatment_eff
        cam_future = np.clip(cam_resized * scale, 0, 1)
        future_alpha = 0.55
        overlay_future = overlay_cam_on_image(img_bgr, cam_future, alpha=future_alpha)[:, :, ::-1]

    with colA:
        st.image(overlay_rgb, caption="Current (Grad-CAM)", use_container_width=True)
    with colB:
        st.image(overlay_future, caption=f"Predicted Future (Year {years_cam}) — Treatments: {', '.join(treatments_to_compare) if treatments_to_compare else 'None'}", use_container_width=True)

    st.markdown("---")
    st.subheader("🤖 Explanation & Practical Advice")
    explanation_text = generate_llm_explanation(pred_name_display, treatments_to_compare, model_outputs={"prob": pred_prob})
    st.write(explanation_text)

st.markdown(
    """
---
Notes  
- Grad-CAM highlights are approximate and intended for visualization only — not diagnostic.  
- Simulations are illustrative; not clinical predictions.  
- LLM text is educational; consult qualified clinicians for decisions.
"""
)