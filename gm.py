# cv_rehab_games.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import math
import random
from collections import deque
from datetime import datetime

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="CV Rehab Games", layout="wide")

# ---------------------------
# Helper functions
# ---------------------------
def speak_js(text):
    """Use browser TTS via injected JS for quick feedback."""
    safe = str(text).replace('"', "'")
    st.markdown(f"""<script>
        (function() {{
            try {{
                const u = new SpeechSynthesisUtterance("{safe}");
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(u);
            }} catch(e){{ console.error(e); }}
        }})();
        </script>""", unsafe_allow_html=True)

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def smooth_point(prev, new, alpha=0.25):
    if prev is None:
        return new
    return (int(alpha*new[0] + (1-alpha)*prev[0]),
            int(alpha*new[1] + (1-alpha)*prev[1]))

def init_session_state():
    keys_defaults = {
        "running": False,
        "video_ready": False,
        "buf": deque(maxlen=60),
        "frame_idx": 0,
        "cursor": None,
        "cursor_prev": None,
        "data": pd.DataFrame(columns=["timestamp","frame","x","y","speed","tremor","smoothness","game","score","hit"]),
        "balloons": [],
        "balloon_score": 0,
        "strokes": [],   # paint strokes (list of points)
        "current_stroke": [],
        "stars": [],
        "star_score": 0,
        "simon_seq": [],
        "simon_stage": "idle",
        "simon_idx": 0,
        "calm_growth": 0.0,
    }
    for k, v in keys_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# initialize
init_session_state()

# ---------------------------
# Sidebar: settings & game select
# ---------------------------
with st.sidebar:
    st.title("🧠 Rehab Hub")
    game = st.selectbox("Choose game", ["Balloon Pop", "Paint & Draw", "Catch Stars", "Simon Says", "Calm Garden", "Video Exercises"])
    show_video = st.checkbox("Show camera feed", value=True)
    initial_radius = st.slider("Initial target radius (px)", 20, 120, 60)
    balloon_spawn_rate = st.slider("Balloons spawn/sec (approx)", 0.3, 3.0, 0.8)
    star_spawn_rate = st.slider("Stars spawn/sec (approx)", 0.3, 3.0, 0.8)
    debug_mode = st.checkbox("Show debug metrics", value=False)

# start/stop outside loop (unique keys)
col_start, col_stop = st.columns(2)
with col_start:
    if st.button("🚀 Start Session", key="start_btn"):
        st.session_state.running = True
        st.session_state.video_ready = False
        # reset some game states on new start
        st.session_state.frame_idx = 0
        st.session_state.buf.clear()
        st.session_state.cursor = None
        st.session_state.cursor_prev = None
        st.session_state.balloons = []
        st.session_state.balloon_score = 0
        st.session_state.strokes = []
        st.session_state.current_stroke = []
        st.session_state.stars = []
        st.session_state.star_score = 0
        st.session_state.simon_seq = []
        st.session_state.simon_stage = "idle"
        st.session_state.simon_idx = 0
        st.session_state.calm_growth = 0.0
        speak_js("Session started. Good luck!")

with col_stop:
    if st.button("⏹ Stop Session", key="stop_btn"):
        st.session_state.running = False
        # save CSV automatically
        try:
            fname = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            st.session_state.data.to_csv(fname, index=False)
            speak_js("Session stopped. Data saved.")
            st.success(f"Saved session to {fname}")
        except Exception as e:
            st.error(f"Failed to save: {e}")

# small top info
st.markdown(f"**Selected game:** {game}")

# left: camera / game
col_left, col_right = st.columns([2,1])
with col_left:
    st.subheader("🎥 Camera Feed & Game")
    frame_window = st.empty()

with col_right:
    st.subheader("📊 Live Metrics & Controls")
    speed_place = st.empty()
    tremor_place = st.empty()
    smooth_place = st.empty()
    st.markdown("---")
    st.write("Scores / Controls")
    if game == "Balloon Pop":
        st.write("Balloon score:", st.session_state.balloon_score)
    if game == "Paint & Draw":
        if st.button("Clear Drawing", key="clear_drawing"):
            st.session_state.strokes = []
            st.session_state.current_stroke = []
    if game == "Catch Stars":
        st.write("Star score:", st.session_state.star_score)
    if game == "Simon Says":
        if st.button("Generate Simon Sequence", key="gen_simon"):
            # generate sequence of positions (4 targets)
            seq_len = 4
            st.session_state.simon_seq = [random.randint(0,3) for _ in range(seq_len)]
            st.session_state.simon_stage = "show"
            st.session_state.simon_idx = 0
            st.session_state.simon_time = time.time()
            speak_js("New sequence generated.")
    if game == "Calm Garden":
        st.write(f"Growth: {st.session_state.calm_growth:.2f}")

    st.markdown("---")
    if st.button("Export full session CSV", key="export_btn"):
        fname = "session_export.csv"
        st.session_state.data.to_csv(fname, index=False)
        st.success(f"Exported to {fname}")

    st.markdown("---")
    st.write("Video exercise links")
    if game == "Video Exercises":
        st.video("https://www.youtube.com/watch?v=1sV6XNoyYo8")  # example
        st.video("https://www.youtube.com/watch?v=ET3x2R3a6sA")
        st.video("https://www.youtube.com/watch?v=TZ7dWpWsy4s")

# ---------------------------
# Mediapipe setup
# ---------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------------------------
# Helper: spawn balloon / star
# ---------------------------
def spawn_balloon(w, h, min_r=30, max_r=80):
    r = random.randint(min_r, max_r)
    x = random.randint(r+20, w - r - 20)
    y = random.randint(r+20, h - r - 20)
    return {"x": x, "y": y, "r": r, "created": time.time(), "popped": False}

def spawn_star(w, h):
    x = random.randint(30, w-30)
    y = -20
    speed = random.uniform(3.0, 8.0)
    return {"x": x, "y": y, "speed": speed, "caught": False}

# ---------------------------
# Main camera + game loop
# ---------------------------
if st.session_state.running and game != "Video Exercises":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        st.error("Unable to open camera. Check device and permissions.")
        st.session_state.running = False
    else:
        try:
            with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
                last_balloon_spawn = time.time()
                last_star_spawn = time.time()
                while st.session_state.running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed reading frame from camera.")
                        break

                    frame = cv2.flip(frame, 1)
                    h, w, _ = frame.shape
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(img_rgb)

                    cx, cy = None, None
                    if results.multi_hand_landmarks:
                        lm = results.multi_hand_landmarks[0]
                        # index finger tip landmark 8
                        ix = int(lm.landmark[8].x * w)
                        iy = int(lm.landmark[8].y * h)
                        # smoothing
                        prev = st.session_state.cursor_prev
                        sm = smooth_point(prev, (ix, iy), alpha=0.35)
                        st.session_state.cursor_prev = sm
                        st.session_state.cursor = sm
                        cx, cy = sm
                        mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                    else:
                        # if no hand found, keep previous cursor
                        cx, cy = st.session_state.cursor_prev if st.session_state.cursor_prev is not None else (None, None)

                    ts_now = time.time()
                    # append buffer for metrics
                    if cx is not None and cy is not None:
                        st.session_state.buf.append((cx, cy, ts_now))
                    else:
                        # append last known to preserve continuity
                        if len(st.session_state.buf) > 0:
                            lastpos = st.session_state.buf[-1]
                            st.session_state.buf.append((lastpos[0], lastpos[1], ts_now))

                    # compute basic metrics (speed, tremor, smoothness)
                    def compute_metrics_local(buf):
                        if len(buf) < 3:
                            return 0.0, 0.0, 0.0
                        arr = np.array([[p[0], p[1]] for p in buf])
                        times = np.array([p[2] for p in buf])
                        dt = np.diff(times)
                        dt[dt==0] = 1e-6
                        diffs = np.diff(arr, axis=0)
                        v = np.linalg.norm(diffs, axis=1) / dt
                        speed = float(np.mean(v)) if v.size>0 else 0.0
                        mean_pos = np.mean(arr, axis=0)
                        tremor = float(np.std(np.linalg.norm(arr - mean_pos, axis=1)))
                        smoothness = 0.0
                        if len(v) >= 3:
                            a = np.diff(v) / dt[1:]
                            j = np.diff(a) / dt[2:] if len(a) > 1 else np.array([])
                            smoothness = float(np.sqrt(np.mean(j**2))) if j.size>0 else 0.0
                        return speed, tremor, smoothness

                    speed, tremor, smoothness = compute_metrics_local(list(st.session_state.buf)[-30:])
                    speed_place.metric("⚡ Speed (px/s)", f"{speed:.1f}")
                    tremor_place.metric("🤲 Tremor (px std)", f"{tremor:.2f}")
                    smooth_place.metric("📈 Smoothness", f"{smoothness:.4f}")

                    # ------------------
                    # GAME LOGIC
                    # ------------------
                    hit = False
                    if game == "Balloon Pop":
                        # spawn balloons occasionally
                        if time.time() - last_balloon_spawn > (1.0 / balloon_spawn_rate):
                            st.session_state.balloons.append(spawn_balloon(w, h, min_r=30, max_r=80))
                            last_balloon_spawn = time.time()

                        # draw balloons & check pop
                        for b in st.session_state.balloons:
                            if b["popped"]:
                                continue
                            cv2.circle(frame, (int(b["x"]), int(b["y"])), int(b["r"]), (0, 0, 255), -1)
                            if cx is not None and cy is not None:
                                if dist((cx, cy), (b["x"], b["y"])) < b["r"]:
                                    b["popped"] = True
                                    st.session_state.balloon_score += 1
                                    hit = True
                                    speak_js("Pop")
                        # remove popped after small delay
                        st.session_state.balloons = [bb for bb in st.session_state.balloons if not bb.get("popped")]

                    elif game == "Paint & Draw":
                        # if index and thumb close -> pen down (optional)
                        # simple behavior: always paint while hand present
                        if cx is not None and cy is not None:
                            st.session_state.current_stroke.append((cx, cy))
                        else:
                            if len(st.session_state.current_stroke) > 0:
                                st.session_state.strokes.append(st.session_state.current_stroke.copy())
                                st.session_state.current_stroke = []

                        # draw strokes on frame
                        for stroke in st.session_state.strokes:
                            for i in range(1, len(stroke)):
                                cv2.line(frame, stroke[i-1], stroke[i], (0,255,0), 4)
                        # draw current stroke
                        for i in range(1, len(st.session_state.current_stroke)):
                            cv2.line(frame, st.session_state.current_stroke[i-1], st.session_state.current_stroke[i], (255,0,0), 4)

                    elif game == "Catch Stars":
                        # spawn stars
                        if time.time() - last_star_spawn > (1.0 / star_spawn_rate):
                            st.session_state.stars.append(spawn_star(w, h))
                            last_star_spawn = time.time()

                        # update stars
                        for s in st.session_state.stars:
                            s["y"] += s["speed"]
                            # draw star as small circle
                            cv2.circle(frame, (int(s["x"]), int(s["y"])), 12, (0,255,255), -1)
                            if cx is not None and cy is not None:
                                if dist((cx, cy), (s["x"], s["y"])) < 20 and not s.get("caught", False):
                                    s["caught"] = True
                                    st.session_state.star_score += 1
                                    hit = True
                                    speak_js("Nice catch")
                        # remove out-of-screen or caught
                        st.session_state.stars = [s for s in st.session_state.stars if (s["y"] < h+50 and not s.get("caught", False))]

                    elif game == "Simon Says":
                        # four quarter targets
                        targets = [
                            (int(w*0.25), int(h*0.25)),
                            (int(w*0.75), int(h*0.25)),
                            (int(w*0.25), int(h*0.75)),
                            (int(w*0.75), int(h*0.75)),
                        ]
                        # show sequence phase
                        if st.session_state.simon_stage == "show":
                            elapsed = time.time() - st.session_state.get("simon_time", 0)
                            # highlight each item for 0.8s sequentially
                            seq = st.session_state.simon_seq
                            for idx, t in enumerate(seq):
                                t_start = idx * 0.9
                                t_end = t_start + 0.8
                                color = (50,50,50)
                                if t_start <= elapsed < t_end:
                                    color = (0,255,0)
                                cv2.circle(frame, targets[t], 40, color, -1)
                            if elapsed > len(st.session_state.simon_seq) * 0.9:
                                st.session_state.simon_stage = "await"
                                st.session_state.simon_idx = 0
                                speak_js("Now your turn")
                        elif st.session_state.simon_stage in ("idle","await"):
                            # draw targets
                            for i, tpos in enumerate(targets):
                                cv2.circle(frame, tpos, 40, (100,100,100), -1)
                            # if player reaches current target in sequence
                            if st.session_state.simon_stage == "await" and cx is not None:
                                cur_target_idx = st.session_state.simon_seq[st.session_state.simon_idx]
                                cur_pos = targets[cur_target_idx]
                                cv2.circle(frame, cur_pos, 48, (0,255,255), 3)
                                if dist((cx,cy), cur_pos) < 45:
                                    # correct hit
                                    st.session_state.simon_idx += 1
                                    speak_js("Good")
                                    if st.session_state.simon_idx >= len(st.session_state.simon_seq):
                                        st.session_state.simon_stage = "idle"
                                        speak_js("Sequence complete")
                                    time.sleep(0.3)  # small debounce
                    elif game == "Calm Garden":
                        # encourage slow movement: lower speed increases growth
                        # baseline slow threshold ~ 50 px/s
                        if speed < 50:
                            st.session_state.calm_growth = min(1.0, st.session_state.calm_growth + 0.002)
                        else:
                            st.session_state.calm_growth = max(0.0, st.session_state.calm_growth - 0.001)
                        # draw a plant (circle) whose radius depends on growth
                        plant_x, plant_y = int(w*0.5), int(h*0.6)
                        radius = int(20 + st.session_state.calm_growth * 120)
                        cv2.circle(frame, (plant_x, plant_y), radius, (34,139,34), -1)

                    # overlay cursor
                    if cx is not None and cy is not None:
                        cv2.circle(frame, (cx, cy), 8, (255,0,0), -1)

                    # log row
                    new_row = {
                        "timestamp": ts_now,
                        "frame": st.session_state.frame_idx,
                        "x": cx if cx is not None else None,
                        "y": cy if cy is not None else None,
                        "speed": speed,
                        "tremor": tremor,
                        "smoothness": smoothness,
                        "game": game,
                        "score": st.session_state.get("balloon_score", 0) + st.session_state.get("star_score", 0),
                        "hit": int(hit)
                    }
                    st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_row])], ignore_index=True)
                    st.session_state.frame_idx += 1

                    # display frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_window.image(frame_rgb, use_container_width=True, channels="RGB")
                    st.session_state.video_ready = True

                    # small sleep to control FPS
                    time.sleep(1.0/20.0)

        except Exception as e:
            st.error(f"Camera loop error: {e}")
        finally:
            cap.release()
            st.session_state.running = False
            st.experimental_rerun()  # rerun UI so buttons return to normal

else:
    # not running
    if not st.session_state.video_ready:
        frame_window.markdown(
            "<div style='padding:30px; text-align:center; border:2px dashed #ddd; border-radius:10px;'>"
            "<b>Press 'Start Session' to begin.</b><br>Choose a game and allow camera permission.</div>",
            unsafe_allow_html=True
        )

# ---------------------------
# Post-session summary / show last rows
# ---------------------------
st.markdown("---")
st.subheader("Session log (last 10 rows)")
if not st.session_state.data.empty:
    st.dataframe(st.session_state.data.tail(10), use_container_width=True)
else:
    st.info("No data yet. Start a session and play a game to generate metrics.")
