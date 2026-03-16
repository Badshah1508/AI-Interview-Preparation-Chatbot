import json
import random
import pickle
import nltk
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')

with open("intents.json") as file:
    data = json.load(file)

patterns = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)

model = LogisticRegression()
model.fit(X, tags)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully!")