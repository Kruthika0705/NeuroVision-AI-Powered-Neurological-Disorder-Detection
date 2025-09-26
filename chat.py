import streamlit as st
import google.generativeai as genai
from deep_translator import GoogleTranslator
from gtts import gTTS
import tempfile
import queue
import numpy as np
import speech_recognition as sr
import base64
import time
from scipy.io.wavfile import write
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# ---------------- CONFIG ----------------
API_KEY = "AIzaSyC3EfYB16D1cgtnzgGTFZdPvVWNAkIXi_M"  # Replace with your Google Gemini API key
MODEL = "gemini-1.5-flash"
LANGS = {"English": "en", "Kannada": "kn", "Hindi": "hi"}

st.set_page_config(page_title="NeuroChat Live Audio", layout="centered")
st.title("🧠 NeuroChat — Neurology Q&A (Gemini)")
st.caption("Educational chatbot powered by Google Gemini. Not a replacement for medical advice.")

# ---------------- BACKGROUND IMAGE ----------------
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
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("C:/Users/Kruthika/Downloads/game.jpeg")  # Update with your image path

# ---------------- GEMINI SETUP ----------------
genai.configure(api_key=API_KEY)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

SYSTEM_MSG = (
    "You are a concise assistant specialized in neurology. "
    "Answer clearly, evidence-based, non-diagnostic. "
    "If asked for diagnosis, politely refuse and suggest consulting a clinician. "
    "Keep answers 3–7 sentences."
)

# ---------------- FUNCTIONS ----------------
def build_prompt(history, user_msg):
    """Builds a structured conversation prompt for Gemini"""
    prompt = [f"SYSTEM: {SYSTEM_MSG}"]
    for m in history[-6:]:
        prompt.append(f"{m['role'].upper()}: {m['content']}")
    prompt.append(f"USER: {user_msg}")
    prompt.append("ASSISTANT:")
    return "\n".join(prompt)

def call_gemini(prompt):
    """Get Gemini response"""
    try:
        model = genai.GenerativeModel(MODEL)
        resp = model.generate_content(prompt, temperature=0.7, max_output_tokens=500)
        return resp.text if hasattr(resp, "text") else "⚠ No response from Gemini."
    except Exception as e:
        return f"⚠ Error: {e}"

def translate_text(text, target_lang):
    if target_lang == "en":
        return text
    return GoogleTranslator(source="auto", target=target_lang).translate(text)

def speak(text):
    """Convert text to speech and play it"""
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
        tts.save(tf.name)
        st.audio(tf.name)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Chat Language", list(LANGS.keys()), index=0)
    enable_voice_out = st.checkbox("🔊 Speak Gemini's Reply", value=True)
    if st.button("🧹 Clear Chat"):
        st.session_state.history = []

# ---------------- DISPLAY CHAT ----------------
st.markdown("## Conversation History")
for msg in st.session_state.history:
    role = "You" if msg["role"] == "user" else "NeuroChat"
    st.markdown(f"**{role}:** {msg['content']}")

st.markdown("---")

# ---------------- TEXT INPUT ----------------
st.markdown("### 💬 Type your question")
user_input = st.text_input("Type here and press Enter")

if st.button("Send Text"):
    if user_input.strip():
        lang_code = LANGS[language]
        translated_input = translate_text(user_input, "en") if lang_code != "en" else user_input

        st.session_state.history.append({"role": "user", "content": user_input})

        # Get response
        prompt = build_prompt(st.session_state.history, translated_input)
        gemini_output = call_gemini(prompt)
        final_output = translate_text(gemini_output, lang_code) if lang_code != "en" else gemini_output

        st.session_state.history.append({"role": "assistant", "content": final_output})

        # Speak if enabled
        if enable_voice_out:
            speak(final_output)

        st.experimental_rerun()

# ---------------- LIVE AUDIO INPUT ----------------
st.markdown("---")
st.markdown("### 🎤 Live Audio Chat")

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.q = queue.Queue()

    def recv(self, frame):
        audio = frame.to_ndarray()
        self.q.put(audio)
        return frame

# WebRTC Audio Stream
ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

# ---------------- AUDIO PROCESSING ----------------
if st.button("Send Audio"):
    if ctx.audio_processor and not ctx.audio_processor.q.empty():
        # Gather audio chunks
        chunks = []
        while not ctx.audio_processor.q.empty():
            chunks.append(ctx.audio_processor.q.get())

        audio_np = np.concatenate(chunks, axis=0)

        # Save to temp WAV
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        write(tmp_wav.name, 44100, audio_np)

        # Speech Recognition
        r = sr.Recognizer()
        with sr.AudioFile(tmp_wav.name) as source:
            audio_data = r.record(source)
            try:
                spoken_text = r.recognize_google(audio_data)
                st.success(f"🎙 You said: **{spoken_text}**")
            except sr.UnknownValueError:
                st.error("⚠ Could not understand the audio clearly.")
                spoken_text = ""
            except sr.RequestError:
                st.error("⚠ Speech Recognition service error.")
                spoken_text = ""

        # If speech was recognized
        if spoken_text:
            lang_code = LANGS[language]
            translated_input = translate_text(spoken_text, "en") if lang_code != "en" else spoken_text

            # Add to conversation
            st.session_state.history.append({"role": "user", "content": spoken_text})

            # Get Gemini response
            prompt = build_prompt(st.session_state.history, translated_input)
            gemini_output = call_gemini(prompt)
            final_output = translate_text(gemini_output, lang_code) if lang_code != "en" else gemini_output

            st.session_state.history.append({"role": "assistant", "content": final_output})

            # Display and speak
            st.markdown(f"**NeuroChat:** {final_output}")
            if enable_voice_out:
                speak(final_output)
    else:
        st.warning("No audio detected. Press 'Start', speak, then click 'Send Audio'.")

# ---------------- SAVE CHAT ----------------
if st.session_state.history:
    if st.button("💾 Save Conversation"):
        fname = f"neurochat_{int(time.time())}.txt"
        with open(fname, "w", encoding="utf-8") as f:
            for msg in st.session_state.history:
                f.write(f"{msg['role'].upper()}: {msg['content']}\n\n")
        st.success(f"Conversation saved to {fname}")