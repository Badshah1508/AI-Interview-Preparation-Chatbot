import ollama

def generate_followup(question, answer):

    prompt = f"""
You are a technical interviewer.

Original question:
{question}

Candidate answer:
{answer}

Ask ONE follow-up interview question based on the answer.
Only return the question.
"""

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


# ---------- TEST CODE ----------
question = "What is Machine Learning?"
answer = "Machine learning allows computers to learn patterns from data."

followup = generate_followup(question, answer)

print("Follow-up question:", followup)