import json
import random
from sentence_transformers import SentenceTransformer, util

# load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# load dataset
with open("interview_questions.json") as file:
    data = json.load(file)

print("AI Interviewer Started")
print("Type 'exit' to stop\n")

while True:

    # choose random topic
    topic = random.choice(data["topics"])

    # choose random question from that topic
    q = random.choice(topic["questions"])

    print("\nTopic:", topic["topic"])
    print("Bot:", q["question"])

    user_answer = input("Your answer: ")

    if user_answer.lower() == "exit":
        break

    expected_answer = q["answer"]

    # embeddings
    emb1 = model.encode(user_answer, convert_to_tensor=True)
    emb2 = model.encode(expected_answer, convert_to_tensor=True)

    score = util.cos_sim(emb1, emb2).item()

    print("\nSimilarity Score:", round(score,2))

    if score > 0.65:
        print("Good answer!")
    elif score > 0.4:
        print("Partially correct.")
    else:
        print("Incorrect answer.")

    print("\nExpected answer:")
    print(expected_answer)