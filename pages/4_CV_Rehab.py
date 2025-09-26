import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from collections import deque
import math
import joblib

st.set_page_config(page_title="CV Rehab Tracker", layout="wide")

# ---------------- Configuration ----------------
FPS = 20
WINDOW_SECONDS = 3
BUFFER_LEN = int(FPS * WINDOW_SECONDS)
CSV_OUTPUT = "session_log.csv"
PREDICT_INTERVAL = 5
PRED_WINDOW_LEN = 6

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.title("Settings")
    target_radius = st.slider("Initial target radius (px)", 20, 150, 60)
    difficulty_step = st.slider("Difficulty adjustment step (px)", 2, 30, 8)
    tremor_threshold = st.slider("Tremor threshold (px std)", 1, 50, 8)
    speed_threshold = st.slider("Speed threshold (px/s)", 10, 1000, 80)
    show_video = st.checkbox("Show camera feed", value=True)
    save_every = st.number_input("Save every N seconds", min_value=5, value=10)

# ---------------- Mediapipe setup ----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

@st.cache_data
def empty_df():
    cols = ["timestamp","frame","x","y","speed","tremor","smoothness","target_x","target_y","target_radius","hit"]
    return pd.DataFrame(columns=cols)

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def compute_metrics(buf):
    if len(buf) < 3:
        return 0.0, 0.0, 0.0
    arr = np.array([[p[0], p[1]] for p in buf])
    times = np.array([p[2] for p in buf])
    dt = np.diff(times)
    dt[dt==0] = 1e-6
    diffs = np.diff(arr, axis=0)
    v = np.linalg.norm(diffs, axis=1) / dt
    speed = np.mean(v)
    mean_pos = np.mean(arr, axis=0)
    tremor = np.std(np.linalg.norm(arr - mean_pos, axis=1))
    if len(v) < 3:
        smoothness = 0.0
    else:
        a = np.diff(v) / dt[1:]
        j = np.diff(a) / dt[2:] if len(a) > 1 else np.array([])
        smoothness = np.sqrt(np.mean(j**2)) if j.size > 0 else 0.0
    return speed, tremor, smoothness

@st.cache_resource
def load_models():
    models = {}
    try:
        models["rf"] = joblib.load("neuro_rf_model.pkl")
    except Exception:
        pass
    try:
        models["scaler"] = joblib.load("scaler.pkl")
    except Exception:
        pass
    try:
        models["encoder"] = joblib.load("label_encoder.pkl")
    except Exception:
        pass
    return models

models = load_models()

def extract_features(df):
    return {
        "avg_speed": df["speed"].mean(),
        "std_speed": df["speed"].std(),
        "avg_tremor": df["tremor"].mean(),
        "max_tremor": df["tremor"].max(),
        "avg_smooth": df["smoothness"].mean(),
        "std_smooth": df["smoothness"].std(),
        "accuracy": df["hit"].mean(),
        "reaction_time": estimate_reaction_time(df),
    }

def estimate_reaction_time(df):
    hits = df[df["hit"] == 1]["timestamp"].values
    if len(hits) < 2:
        return 2.0
    return np.mean(np.diff(hits))

# ---------------- Session state ----------------
if "data" not in st.session_state:
    st.session_state.data = empty_df()
if "buf" not in st.session_state:
    st.session_state.buf = deque(maxlen=BUFFER_LEN)
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0
if "target" not in st.session_state:
    st.session_state.target = (300, 200)
if "current_radius" not in st.session_state:
    st.session_state.current_radius = target_radius
if "last_save" not in st.session_state:
    st.session_state.last_save = time.time()
if "last_pred" not in st.session_state:
    st.session_state.last_pred = 0
if "pred_window" not in st.session_state:
    st.session_state.pred_window = deque(maxlen=PRED_WINDOW_LEN)
if "running" not in st.session_state:
    st.session_state.running = False

# ---------------- UI layout ----------------
st.title("🖐️ CV Rehab Tracker — Adaptive Exercises")

col_left, col_right = st.columns([2,1])

# ---------------- Right Column: Controls + Prediction ----------------
with col_right:
    st.markdown("### ▶️ Session Controls")
    start_btn = st.button("Start Session", key="start_session")
    stop_btn = st.button("Stop & Save", key="stop_session")
    st.caption(f"Session CSV will be saved as: `{CSV_OUTPUT}`")
    
    st.markdown("### 🧠 Condition Prediction")
    pred_text = st.empty()
    pred_chart = st.empty()
    
    st.markdown("### 📊 Live Metrics")
    metric_speed = st.metric(label="Speed (px/s)", value="0")
    metric_tremor = st.metric(label="Tremor (px std)", value="0")
    metric_smooth = st.metric(label="Smoothness (jerk RMS)", value="0")
    
    st.markdown("---")
    adaptive_text = st.empty()

# ---------------- Left Column: Camera + Logs + Videos ----------------
with col_left:
    st.markdown("### 🎥 Live Camera Feed / Exercise")
    if show_video:
        frame_window = st.image([])
    else:
        st.info("Camera feed hidden — enable in sidebar to view live video.")
    
    st.markdown("### 📑 Recent Session Logs")
    log_table = st.dataframe(st.session_state.data.tail(8), use_container_width=True)
    
    st.markdown("### 📺 Video Exercise Library")
    # Video grid layout (2 per row)
    videos = [
    ("Move to Improve (Parkinson's)", "https://www.youtube.com/embed/jyOk-2DmVnU"),
    ("Better Every Day PD Exercises", "https://www.youtube.com/embed/cybmNjESPRg"),
    ("LSVT BIG Movement Session", "https://www.youtube.com/embed/pgtGOgVIhqc"),
    ("Brain Meditation (Kirtan Kriya)", "https://www.youtube.com/embed/pVE_PVGuXhM"),
    ("Birds Music Therapy", "https://www.youtube.com/embed/0-3OnQphZd8"),
    ("Ambient Sleep & Dementia Relax", "https://www.youtube.com/embed/CkmXEr8d_Uo")
]

for i in range(0, len(videos), 2):
    cols = st.columns(2)
    for j, col in enumerate(cols):
        if i + j < len(videos):
            name, url = videos[i + j]
            col.markdown(f"**{name}**")
            col.markdown(
                f"""
                <div style="text-align:center">
                  <iframe width="400" height="225"
                          src="{url}"
                          frameborder="0"
                          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                          allowfullscreen>
                  </iframe>
                </div>
                """,
                unsafe_allow_html=True
            )

# ---------------- Session control ----------------
if start_btn and not st.session_state.running:
    st.session_state.running = True
    st.success("✅ Session started")

if stop_btn and st.session_state.running:
    st.session_state.running = False
    try:
        st.session_state.data.to_csv(CSV_OUTPUT, index=False)
        st.success(f"💾 Saved session to {CSV_OUTPUT}")
    except Exception as e:
        st.error(f"❌ Failed to save: {e}")

# ---------------- Main camera loop ----------------
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.5) as hands:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from camera")
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            cx, cy = None, None
            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    xs = [p.x * w for p in lm.landmark]
                    ys = [p.y * h for p in lm.landmark]
                    cx = int(np.mean(xs))
                    cy = int(np.mean(ys))
                    mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                    break
            
            ts_now = time.time()
            if cx is not None and cy is not None:
                st.session_state.buf.append((cx, cy, ts_now))
            elif len(st.session_state.buf) > 0:
                last = st.session_state.buf[-1]
                st.session_state.buf.append((last[0], last[1], ts_now))
            
            speed, tremor, smoothness = compute_metrics(st.session_state.buf)
            
            # Adaptive difficulty
            if tremor > tremor_threshold or speed < speed_threshold:
                st.session_state.current_radius = min(250, st.session_state.current_radius + difficulty_step)
                adaptive_text.markdown(f"**Adapting:** Easier (radius={st.session_state.current_radius}px)")
            else:
                st.session_state.current_radius = max(10, st.session_state.current_radius - difficulty_step)
                adaptive_text.markdown(f"**Adapting:** Harder (radius={st.session_state.current_radius}px)")
            
            # Target hit logic
            tx, ty = st.session_state.target
            hit = False
            if cx is not None and cy is not None:
                if dist((cx, cy), (tx, ty)) <= st.session_state.current_radius:
                    hit = True
                    st.session_state.target = (np.random.randint(50, w-50), np.random.randint(50, h-50))
            
            # Draw overlays
            overlay = frame.copy()
            cv2.circle(overlay, (tx, ty), int(st.session_state.current_radius), (0, 255, 0), 2)
            if cx is not None and cy is not None:
                cv2.circle(overlay, (cx, cy), 8, (255, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            # Update metrics
            metric_speed.metric(label="Speed (px/s)", value=f"{speed:.1f}")
            metric_tremor.metric(label="Tremor (px std)", value=f"{tremor:.2f}")
            metric_smooth.metric(label="Smoothness (jerk RMS)", value=f"{smoothness:.4f}")
            
            # Append log
            new_row = {
                "timestamp": ts_now,
                "frame": st.session_state.frame_idx,
                "x": cx if cx is not None else np.nan,
                "y": cy if cy is not None else np.nan,
                "speed": speed,
                "tremor": tremor,
                "smoothness": smoothness,
                "target_x": tx,
                "target_y": ty,
                "target_radius": st.session_state.current_radius,
                "hit": int(hit)
            }
            st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_row])], ignore_index=True)
            st.session_state.frame_idx += 1
            
            # Show video
            if show_video:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_rgb)
            
            # Update log table
            log_table.dataframe(st.session_state.data.tail(8), use_container_width=True)
            
            # Auto-save
            if time.time() - st.session_state.last_save > save_every:
                st.session_state.data.to_csv(CSV_OUTPUT, index=False)
                st.session_state.last_save = time.time()
            
            # Live AI prediction
            if "rf" in models and time.time() - st.session_state.last_pred > PREDICT_INTERVAL:
                if len(st.session_state.data) > 20:
                    feats = extract_features(st.session_state.data)
                    X = pd.DataFrame([feats])
                    try:
                        X_scaled = models["scaler"].transform(X.values)
                    except Exception:
                        X_scaled = X.values
                    pred_encoded = models["rf"].predict(X_scaled)[0]
                    if "encoder" in models:
                        pred = models["encoder"].inverse_transform([pred_encoded])[0]
                        classes = models["encoder"].classes_
                    else:
                        pred = pred_encoded
                        classes = models["rf"].classes_
                    probs = models["rf"].predict_proba(X_scaled)[0]
                    st.session_state.last_pred = time.time()
                    st.session_state.pred_window.append(pred)
                    final_pred = max(set(st.session_state.pred_window), key=st.session_state.pred_window.count) \
                        if len(st.session_state.pred_window) == st.session_state.pred_window.maxlen else pred
                    prob_df = pd.DataFrame({"Condition": classes, "Probability": probs}).set_index("Condition")
                    pred_text.success(f"Predicted Condition: **{final_pred}**")
                    pred_chart.bar_chart(prob_df)
            
            time.sleep(1.0 / FPS)
    
    cap.release()
    st.session_state.running = False
else:
    st.info("Press 'Start session' to begin tracking. Allow camera access when prompted.")
    if not st.session_state.data.empty:
        log_table.dataframe(st.session_state.data.tail(8), use_container_width=True)
