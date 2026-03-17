import streamlit as st
import json
import random
import pandas as pd
import requests  
from sentence_transformers import SentenceTransformer, util
from rag_utils import retrieve_answer
from mock_interviewer import generate_followup

st.set_page_config(page_title="AI Interview Assistant", page_icon="🤖", layout="wide")

# --------------------------------------------------
# UI Styling
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#e3f2fd,#ffffff,#f5f7fa);
}
@media (prefers-color-scheme: dark) {
.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}
}
.chat-card {
    background: rgba(255,255,255,0.4);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.3);
}
[data-testid="stChatMessage"] {
    border-radius: 10px;
    padding: 12px;
}
.stButton>button {
    background: linear-gradient(135deg,#667eea,#764ba2);
    border-radius: 10px;
    border: none;
    color: white;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.title("AI Interview Preparation Assistant")

# --------------------------------------------------
# OLLAMA FUNCTION (FAST MODEL)
# --------------------------------------------------

def ask_ollama(prompt, model="phi3"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        return response.json()["response"]
    except Exception as e:
        return f"Ollama error: {str(e)}"

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

@st.cache_data
def load_dataset():
    with open("interview_questions.json") as f:
        return json.load(f)

data = load_dataset()
topics = [t["topic"] for t in data["topics"]]

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------

st.sidebar.header("Interview Settings")

selected_topic = st.sidebar.selectbox(
    "Select Topic",
    topics,
    key="topic_selector"
)

# --------------------------------------------------
# Session State Initialization
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "score" not in st.session_state:
    st.session_state.score = 0

if "total" not in st.session_state:
    st.session_state.total = 0

if "topic_stats" not in st.session_state:
    st.session_state.topic_stats = {}

if "last_topic" not in st.session_state:
    st.session_state.last_topic = selected_topic

# --------------------------------------------------
# Topic Change Detection
# --------------------------------------------------

if st.session_state.last_topic != selected_topic:

    topic_data = next(t for t in data["topics"] if t["topic"] == selected_topic)
    q = random.choice(topic_data["questions"])

    st.session_state.topic = topic_data["topic"]
    st.session_state.current_question = q["question"]
    st.session_state.correct_answer = q["answer"]

    # cache embedding once
    st.session_state.expected_embedding = model.encode(q["answer"], convert_to_tensor=True)

    st.session_state.messages = []
    st.session_state.last_topic = selected_topic

# --------------------------------------------------
# Initialize Question
# --------------------------------------------------

if "current_question" not in st.session_state:

    topic_data = next(t for t in data["topics"] if t["topic"] == selected_topic)
    q = random.choice(topic_data["questions"])

    st.session_state.topic = topic_data["topic"]
    st.session_state.current_question = q["question"]
    st.session_state.correct_answer = q["answer"]

    st.session_state.expected_embedding = model.encode(q["answer"], convert_to_tensor=True)

# --------------------------------------------------
# Display Question
# --------------------------------------------------

with st.container():
    st.subheader(f"Topic: {st.session_state.topic}")
    st.markdown("### Interview Question")
    st.write(st.session_state.current_question)

# --------------------------------------------------
# Chat Input
# --------------------------------------------------

user_answer = st.chat_input("Type your answer here...")

if user_answer:

    st.session_state.messages.append(("user", user_answer))

    topic = st.session_state.topic

    if topic not in st.session_state.topic_stats:
        st.session_state.topic_stats[topic] = {"correct":0,"total":0}

    st.session_state.topic_stats[topic]["total"] += 1
    st.session_state.total += 1

    # -------------------------
    # Similarity (ONLY if real question)
    # -------------------------
    if st.session_state.correct_answer != "AI Generated":

        emb1 = model.encode(user_answer, convert_to_tensor=True)
        emb2 = st.session_state.expected_embedding

        similarity = util.cos_sim(emb1, emb2).item()

        if similarity > 0.7:
            feedback = "Excellent answer!"
            st.session_state.score += 1
            st.session_state.topic_stats[topic]["correct"] += 1
        elif similarity > 0.5:
            feedback = "Partially correct."
        else:
            feedback = "Incorrect answer."

    else:
        similarity = 0
        feedback = "AI-evaluated answer"

    # -------------------------
    # OLLAMA (WITH SPINNER)
    # -------------------------
    with st.spinner("Thinking... 🤖"):
        prompt = f"""
You are an interview expert.

Question: {st.session_state.current_question}
Candidate Answer: {user_answer}

Evaluate the answer like a real interviewer.

Return:
- Feedback (2–3 lines)
- Ideal Answer (short and clear)
- One follow-up question
- Score out of 10
"""
        llm_response = ask_ollama(prompt)

    # -------------------------
    # Extract follow-up
    # -------------------------
    followup_question = ""
    for line in llm_response.split("\n"):
        if "follow" in line.lower():
            followup_question = line
            break

    if followup_question:
        st.session_state.next_question = followup_question

    bot_message = f"""
{feedback}

Similarity Score: **{round(similarity,2)}**

AI Feedback:
{llm_response}
"""

    st.session_state.messages.append(("assistant", bot_message))

# --------------------------------------------------
# Chat History
# --------------------------------------------------

for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.write(msg)

# --------------------------------------------------
# Next Question
# --------------------------------------------------

if st.button("Next Question"):

    st.session_state.messages = []

    if "next_question" in st.session_state:

        st.session_state.current_question = st.session_state.next_question
        st.session_state.correct_answer = "AI Generated"

        del st.session_state.next_question

    else:
        topic_data = next(t for t in data["topics"] if t["topic"] == selected_topic)
        q = random.choice(topic_data["questions"])

        st.session_state.topic = topic_data["topic"]
        st.session_state.current_question = q["question"]
        st.session_state.correct_answer = q["answer"]

        st.session_state.expected_embedding = model.encode(q["answer"], convert_to_tensor=True)

    st.rerun()

# --------------------------------------------------
# Analytics
# --------------------------------------------------

st.sidebar.header("Interview Analytics")

accuracy = 0
if st.session_state.total > 0:
    accuracy = (st.session_state.score / st.session_state.total) * 100

st.sidebar.metric("Correct Answers", st.session_state.score)
st.sidebar.metric("Total Questions", st.session_state.total)
st.sidebar.metric("Accuracy", f"{round(accuracy,1)}%")

# --------------------------------------------------
# Topic Performance
# --------------------------------------------------

if st.session_state.topic_stats:

    chart_data = []

    for topic,stats in st.session_state.topic_stats.items():
        acc = (stats["correct"]/stats["total"])*100
        chart_data.append({"Topic":topic,"Accuracy":acc})

    df = pd.DataFrame(chart_data)

    st.sidebar.subheader("Topic Performance")
    st.sidebar.bar_chart(df.set_index("Topic"))

# --------------------------------------------------
# Weak / Strong
# --------------------------------------------------

weak_topics = []
strong_topics = []

for topic,stats in st.session_state.topic_stats.items():
    acc = stats["correct"]/stats["total"]

    if acc < 0.5:
        weak_topics.append(topic)
    elif acc > 0.8:
        strong_topics.append(topic)

if weak_topics:
    st.sidebar.subheader("Weak Areas")
    for t in weak_topics:
        st.sidebar.write(t)

if strong_topics:
    st.sidebar.subheader("Strengths")
    for t in strong_topics:
        st.sidebar.write(t)