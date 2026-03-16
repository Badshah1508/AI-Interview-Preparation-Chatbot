import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("interview_questions.json") as f:
    data = json.load(f)

documents = []
answers = []

for topic in data["topics"]:
    for q in topic["questions"]:
        documents.append(q["question"])
        answers.append(q["answer"])

embeddings = model.encode(documents)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, "interview_index.faiss")