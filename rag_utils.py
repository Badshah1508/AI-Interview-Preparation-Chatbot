import faiss
import numpy as np
import json

index = faiss.read_index("interview_index.faiss")

with open("interview_questions.json") as f:
    data = json.load(f)

answers = []

for topic in data["topics"]:
    for q in topic["questions"]:
        answers.append(q["answer"])

def retrieve_answer(model, query):

    query_embedding = model.encode([query])

    D, I = index.search(np.array(query_embedding), k=1)

    return answers[I[0][0]]