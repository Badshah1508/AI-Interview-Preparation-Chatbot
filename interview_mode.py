import json
import random

with open("interview_questions.json") as file:
    data = json.load(file)

print("AI Interviewer Mode Started")
print("Type 'exit' to stop\n")

while True:

    q = random.choice(data["questions"])

    print("\nBot:", q["question"])

    user = input("Your answer: ")

    if user.lower() == "exit":
        break

    print("\nExpected answer:")
    print(q["answer"])