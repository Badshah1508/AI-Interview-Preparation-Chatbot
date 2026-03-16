import streamlit as st
import json
import random
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="AI Interview Chatbot", page_icon="👻", layout="wide")

st.title("AI Interview Preparation Assistant")

st.markdown("""
<style>

/* Adaptive Gradient Background */
.stApp {
    background: linear-gradient(
        135deg,
        rgba(125,185,232,0.25),
        rgba(255,255,255,0.15),
        rgba(200,220,255,0.25)
    );
}

/* Glass Card Container */
.chat-card {
    background: rgba(255,255,255,0.35);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 30px;
    border: 1px solid rgba(255,255,255,0.25);
}

/* Works well in dark mode */
@media (prefers-color-scheme: dark) {

.chat-card {
    background: rgba(30,30,30,0.55);
    border: 1px solid rgba(255,255,255,0.08);
}

}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    border-radius: 10px;
    padding: 10px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg,#667eea,#764ba2);
    border-radius: 10px;
    border: none;
    color: white;
    font-weight: 600;
}

/* Button hover */
.stButton>button:hover {
    background: linear-gradient(135deg,#764ba2,#667eea);
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Load Model (Cached)
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

with st.spinner("Loading AI model..."):
    model = load_model()

# -----------------------------
# Load Dataset (Cached)
# -----------------------------
@st.cache_data
def load_dataset():
    with open("interview_questions.json") as f:
        return json.load(f)

data = load_dataset()


# -----------------------------
# Initialize Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "topic" not in st.session_state:
    topic = random.choice(data["topics"])
    q = random.choice(topic["questions"])

    st.session_state.topic = topic["topic"]
    st.session_state.current_question = q["question"]
    st.session_state.correct_answer = q["answer"]

if "score" not in st.session_state:
    st.session_state.score = 0

if "total" not in st.session_state:
    st.session_state.total = 0


# -----------------------------
# Display Question
# -----------------------------
st.subheader(f"Topic: {st.session_state.topic}")
st.write("### Interview Question")
st.write(st.session_state.current_question)


# -----------------------------
# User Input
# -----------------------------
user_answer = st.chat_input("Type your answer here...")

if user_answer:

    st.session_state.messages.append(("user", user_answer))

    expected = st.session_state.correct_answer

    emb1 = model.encode(user_answer, convert_to_tensor=True)
    emb2 = model.encode(expected, convert_to_tensor=True)

    score = util.cos_sim(emb1, emb2).item()

    st.session_state.total += 1

    if score > 0.65:
        feedback = "Good answer!"
        st.session_state.score += 1
    elif score > 0.4:
        feedback = "Partially correct."
    else:
        feedback = "Incorrect answer."

    bot_message = f"""
{feedback}

Similarity Score: {round(score,2)}

Expected Answer:
{expected}
"""

    st.session_state.messages.append(("bot", bot_message))


# -----------------------------
# Chat History
# -----------------------------
for role, msg in st.session_state.messages:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)


# -----------------------------
# Next Question
# -----------------------------
if st.button("Next Question"):

    topic = random.choice(data["topics"])
    q = random.choice(topic["questions"])

    st.session_state.topic = topic["topic"]
    st.session_state.current_question = q["question"]
    st.session_state.correct_answer = q["answer"]

    st.rerun()


# -----------------------------
# Sidebar Scoreboard
# -----------------------------
st.sidebar.title("Interview Score")

st.sidebar.metric(
    label="Correct Answers",
    value=st.session_state.score
)

st.sidebar.metric(
    label="Total Questions",
    value=st.session_state.total
)