import streamlit as st
import time
import google.generativeai as genai
import base64
from deep_translator import GoogleTranslator
from gtts import gTTS
import tempfile
import streamlit.components.v1 as components

# ---------------- CONFIG ----------------
API_KEY = ""  # Replace with your actual API key
MODEL = "gemini-1.5-flash"
LANGS = {"English": "en", "Kannada": "kn", "Hindi": "hi"}

st.set_page_config(page_title="NeuroChat (Gemini)", page_icon="🧠", layout="centered")
st.title("🧠 NeuroChat — Neurology Q&A (Gemini)")
st.caption("Educational chatbot powered by Google Gemini. Not a replacement for medical advice.")

# ---------------- Background + TEXT COLOR ----------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}

        /* Make ALL text white */
        html, body, [class*="st"], p, span, div, label, h1, h2, h3, h4, h5, h6 {{
            color: white !important;
        }}

        /* Sidebar background + white text */
        [data-testid="stSidebar"] {{
            background-color: rgba(0,0,0,0.7) !important;
        }}
        [data-testid="stSidebar"] * {{
            color: white !important;
        }}

        /* Buttons */
        .stButton>button {{
            background-color: #1E88E5 !important;
            color: white !important;
            font-weight: bold;
            border-radius: 8px;
            border: none;
        }}
        .stButton>button:hover {{
            background-color: #1565C0 !important;
            color: white !important;
        }}

        /* Input fields */
        input, textarea {{
            color: white !important;
            background-color: rgba(0,0,0,0.5) !important;
        }}

        /* Selectbox options text black for readability */
        div[data-baseweb="select"] * {{
            color: black !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("C:/Users/Kruthika/Downloads/brain.jpeg")  # Change path if needed

# ---------------- SETUP ----------------
genai.configure(api_key=API_KEY)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# ---------------- PROMPT ----------------
SYSTEM_MSG = (
    "You are a concise assistant specialized in neurology. "
    "Answer clearly, evidence-based, non-diagnostic. "
    "If asked for diagnosis, politely refuse and suggest consulting a clinician. "
    "Keep answers 3–7 sentences."
)

def build_prompt(history, user_msg):
    parts = [f"SYSTEM: {SYSTEM_MSG}"]
    for m in history[-6:]:
        parts.append(f"{m['role'].upper()}: {m['content']}")
    parts.append(f"USER: {user_msg}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)

def call_gemini(prompt):
    try:
        model = genai.GenerativeModel(MODEL)
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"⚠ Error: {str(e)}"

def speak(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
        tts.save(tf.name)
        st.audio(tf.name, format="audio/mp3")

def translate_text(text, target_lang):
    if target_lang == "en":
        return text
    return GoogleTranslator(source="auto", target=target_lang).translate(text)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Chat Language", options=list(LANGS.keys()), index=0)
    enable_voice_in = st.checkbox("🎤 Enable Voice Typing")
    enable_voice_out = st.checkbox("🔊 Enable Voice Output")
    if st.button("🧹 Clear Chat"):
        st.session_state["history"] = []

# ---------------- DISPLAY CHAT ----------------
for msg in st.session_state["history"]:
    role = "You" if msg["role"] == "user" else "NeuroChat"
    st.markdown(f"{role}:** {msg['content']}")

# ---------------- USER INPUT ----------------
st.write("---")
st.markdown("### Ask a neurology question:")
user_input = st.text_input("Type or use voice below...", key="user_input", value="")

# ---------------- VOICE TYPING ----------------
if enable_voice_in:
    components.html(
        """
        <script>
        function startRecognition() {
            var recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.lang = 'en-IN';
            recognition.interimResults = false;
            recognition.maxAlternatives = 1;

            recognition.onresult = function(event) {
                var result = event.results[0][0].transcript;
                const streamlitInput = window.parent.document.querySelector('input[aria-label="Type or use voice below..."]');
                if (streamlitInput) {
                    streamlitInput.value = result;
                    streamlitInput.dispatchEvent(new Event('input', { bubbles: true }));
                    streamlitInput.dispatchEvent(new Event('change', { bubbles: true }));
                    streamlitInput.blur();
                    setTimeout(() => {
                        const sendBtn = window.parent.document.querySelector('button[kind="primary"]');
                        if (sendBtn) sendBtn.click();
                    }, 500);
                }
            };
            recognition.start();
        }
        </script>
        <button onclick="startRecognition()">🎙 Click to Speak</button>
        """,
        height=100,
    )

# ---------------- PROCESS INPUT ----------------
if st.button("Send") and user_input.strip():
    lang_code = LANGS[language]
    translated_input = translate_text(user_input, "en") if lang_code != "en" else user_input
    st.session_state["history"].append({"role": "user", "content": user_input})

    prompt = build_prompt(st.session_state["history"], translated_input)
    gemini_output = call_gemini(prompt)

    final_output = translate_text(gemini_output, lang_code) if lang_code != "en" else gemini_output
    st.session_state["history"].append({"role": "assistant", "content": final_output})
    st.rerun()

# ---------------- SAVE CHAT ----------------
if st.session_state["history"]:
    if st.button("💾 Save Conversation"):
        fname = f"neurochat_{int(time.time())}.txt"
        with open(fname, "w", encoding="utf-8") as f:
            for msg in st.session_state["history"]:
                f.write(f"{msg['role'].upper()}: {msg['content']}\n\n")
        st.success(f"Conversation saved to {fname}")

# ---------------- VOICE SPEAKING ----------------
if enable_voice_out:
    for msg in st.session_state["history"][::-1]:
        if msg["role"] == "assistant":
            speak(msg["content"])
            break