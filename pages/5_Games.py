import streamlit as st
import random
import time
import base64
from streamlit_autorefresh import st_autorefresh

# --- PAGE CONFIG ---
st.set_page_config(page_title="Switch Challenge", layout="centered")

# --- BACKGROUND IMAGE ---
def get_base64(file_path: str):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

BG_PATH = "C:/Users/Kruthika/Downloads/game.jpeg"  # Update path
BG_IMAGE = get_base64(BG_PATH)

# --- STYLE ---
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: url("data:image/png;base64,{BG_IMAGE}") no-repeat center center fixed;
    background-size: cover;
}}
[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: fixed; inset: 0;
    background: rgba(5, 10, 25, 0.6);
    z-index: 0;
}}
.block-container {{ position: relative; z-index: 1; }}
.title {{
    font-size: 36px; font-weight: bold;
    text-align: center; color: #FFB74D;
    margin-bottom: 20px;
}}
.instructions {{
    background: rgba(0,0,0,0.7);
    padding: 20px; border-radius: 12px;
    margin-bottom: 20px;
    color: white; font-size: 16px;
    line-height: 1.5;
}}
.score-box {{
    background: #262730; padding: 6px 10px;
    border-radius: 8px; font-size: 14px;
    text-align: center; margin: 2px;
    display: inline-block; color: white;
}}
.shape-box {{
    background: #1E1E1E; padding: 8px;
    border-radius: 8px; font-size: 22px;
    font-weight: bold; text-align: center;
    margin: 2px; color: #FFD700;
}}
.output-box {{
    background: #292929; padding: 8px;
    border-radius: 8px; font-size: 22px;
    font-weight: bold; text-align: center;
    margin: 2px; color: #00BFFF;
}}
.input-text {{ font-size: 16px; color: #FFFFFF; margin-bottom: 4px; }}
</style>
""", unsafe_allow_html=True)

# --- GAME LOGIC ---
shapes = ["▲", "■", "●", "+"]
DIFFICULTY = {
    "Easy": {"time": 60, "rounds": 6},
    "Medium": {"time": 45, "rounds": 8},
    "Hard": {"time": 30, "rounds": 10},
}

def generate_challenge():
    input_seq = random.sample(shapes, len(shapes))
    output_seq = random.sample(input_seq, len(input_seq))
    answer = "".join(str(input_seq.index(s) + 1) for s in output_seq)
    return input_seq, output_seq, answer

def calculate_brain_score(score):
    if score >= 16:
        return "🧠 Genius Level Focus"
    elif score >= 12:
        return "💡 Strong Cognitive Skills"
    elif score >= 6:
        return "⚡ Average Mental Agility"
    else:
        return "🌱 Needs More Training"

# --- SESSION STATE INIT ---
for key, val in [
    ("started", False), ("difficulty", "Easy"),
    ("score", 0), ("round", 1),
    ("input_seq", None), ("user_answer", ""),
    ("start_time", None), ("input_key", ""),
    ("history", [])
]:
    if key not in st.session_state:
        st.session_state[key] = val

if st.session_state.input_seq is None:
    seq = generate_challenge()
    st.session_state.input_seq, st.session_state.output_seq, st.session_state.answer = seq

# --- TITLE ---
st.markdown('<div class="title">🎮 Switch Challenge</div>', unsafe_allow_html=True)

# --- INSTRUCTIONS (Before Game Starts) ---
if not st.session_state.started:
    st.markdown(
        """
        <div class="instructions">
        <h3>📘 How to Play</h3>
        <p>Here a few different symbols are displayed on both the top (Input) and bottom (Output) levels of the screen.</p>
        <p>🔹 Your task is to find the correct code of the output, based on the order of the input.</p>
        <p>🔹 The code is the position of each symbol in the input sequence.</p>
        <p>✅ Example:<br>
        Input: ● ■ ▲ + <br>
        Output: ▲ ■ + ● <br>
        Answer: 3241</p>
        <ul>
          <li>⏱ Time duration is based on difficulty (Easy: 60s, Medium: 45s, Hard: 30s)</li>
          <li>🎯 You need to enter the correct 4-digit code</li>
          <li>🏆 +3 points for correct, -1 point for wrong</li>
          <li>🎉 Every 4th round is a Bonus Round (+6 points)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True
    )

    # --- DIFFICULTY SELECTION ---
    st.session_state.difficulty = st.selectbox("🎯 Select Difficulty", list(DIFFICULTY.keys()))
    if st.button("🚀 Start Game", use_container_width=True):
        cfg = DIFFICULTY[st.session_state.difficulty]
        st.session_state.max_time = cfg["time"]
        st.session_state.max_rounds = cfg["rounds"]
        st.session_state.started = True
        st.session_state.start_time = time.time()
        st.session_state.input_seq, st.session_state.output_seq, st.session_state.answer = generate_challenge()
        st.session_state.user_answer = ""
        st.session_state.history = []
        st.session_state.input_key = str(time.time())
        st.session_state.score = 0
        st.session_state.round = 1
        st.rerun()
    st.stop()

# --- TIMER & END CONDITION ---
st_autorefresh(interval=1000, key="refresh")
elapsed = time.time() - st.session_state.start_time
time_left = max(st.session_state.max_time - int(elapsed), 0)

if time_left == 0 or st.session_state.round > st.session_state.max_rounds:
    score = st.session_state.score
    brain = calculate_brain_score(score)
    st.markdown(f"""
        <div style="background: linear-gradient(135deg,#ff416c,#ff4b2b);
         padding:20px;border-radius:15px;text-align:center;
         margin-top:20px;box-shadow:0px 4px 20px rgba(0,0,0,0.6);color:white;">
            <h2>🕹 Game Over</h2>
            <h3>🏆 Final Score: <span style="color:#FFD700">{score}</span></h3>
            <h3>🧠 Brain Fitness: <span style="color:#00FFAA">{brain}</span></h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📝 Round History")
    for e in st.session_state.history:
        st.markdown(f"- *Round {e['round']}*: {e['user_answer']} → {e['result']} (Correct: {e['correct_answer']})")

    if st.button("🔄 Restart Game", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
    st.stop()

# --- GAME UI ---
st.progress((st.session_state.max_time - time_left) / st.session_state.max_time)
cols = st.columns(3)
cols[0].markdown(f"<div class='score-box'>🏆 Score<br>{st.session_state.score}</div>", unsafe_allow_html=True)
cols[1].markdown(f"<div class='score-box'>⏱ Time Left<br>{time_left}s</div>", unsafe_allow_html=True)
cols[2].markdown(f"<div class='score-box'>🔁 Round<br>{st.session_state.round}/{st.session_state.max_rounds}</div>", unsafe_allow_html=True)

st.write("---")

# --- INPUT SEQUENCE DISPLAY ---
st.markdown("<div class='input-text'>💠 Input Sequence:</div>", unsafe_allow_html=True)
cols_in = st.columns(len(st.session_state.input_seq))
for i, s in enumerate(st.session_state.input_seq):
    cols_in[i].markdown(f"<div class='shape-box'>{s}</div>", unsafe_allow_html=True)

# --- OUTPUT SEQUENCE DISPLAY ---
st.markdown("<div class='input-text'>🔃 Output Sequence:</div>", unsafe_allow_html=True)
cols_out = st.columns(len(st.session_state.output_seq))
for i, s in enumerate(st.session_state.output_seq):
    cols_out[i].markdown(f"<div class='output-box'>{s}</div>", unsafe_allow_html=True)

st.write("---")
st.markdown("<div class='input-text'>✍ Enter 4-digit code:</div>", unsafe_allow_html=True)
inp = st.text_input("Code", value="", max_chars=4, key=st.session_state.input_key, label_visibility="collapsed")
if inp:
    st.session_state.user_answer = inp.strip()

# --- SUBMIT LOGIC ---
is_bonus = (st.session_state.round % 4 == 0)
if st.button("✅ Submit"):
    correct = st.session_state.user_answer == st.session_state.answer
    points = 6 if (correct and is_bonus) else (3 if correct else -1)

    if correct:
        st.session_state.score += points
        st.success(f"✅ Correct! +{points} points{' 🎉 Bonus!' if is_bonus else ''}")
        if is_bonus:
            st.balloons()
    else:
        st.session_state.score += points
        st.error(f"❌ Wrong! {points} point{'s' if points != -1 else ''} (Answer: {st.session_state.answer})")

    st.session_state.history.append({
        "round": st.session_state.round,
        "user_answer": st.session_state.user_answer,
        "correct_answer": st.session_state.answer,
        "result": "✅" if correct else "❌"
    })

    st.session_state.round += 1
    st.session_state.input_seq, st.session_state.output_seq, st.session_state.answer = generate_challenge()
    st.session_state.input_key = str(time.time())
    st.session_state.user_answer = ""
    st.rerun()