import json
import random
from sentence_transformers import SentenceTransformer, util

# load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# load questions
with open("interview_questions.json") as file:
    data = json.load(file)

print("AI Interviewer Started")
print("Type 'exit' to stop\n")

while True:

    q = random.choice(data["questions"])

    print("\nBot:", q["question"])

    user_answer = input("Your answer: ")

    if user_answer.lower() == "exit":
        break

    expected_answer = q["answer"]

    # create embeddings
    emb1 = model.encode(user_answer, convert_to_tensor=True)
    emb2 = model.encode(expected_answer, convert_to_tensor=True)

    # compute similarity
    score = util.cos_sim(emb1, emb2)

    similarity = score.item()

    print("\nSimilarity Score:", round(similarity,2))

    if similarity > 0.65:
        print("Good answer!")
    elif similarity > 0.4:
        print("Partially correct answer.")
    else:
        print("Incorrect answer.")

    print("\nExpected answer:")
    print(expected_answer)