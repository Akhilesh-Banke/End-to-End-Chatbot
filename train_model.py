import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load intents
with open("data/intents.json", "r") as file:
    intents = json.load(file)["intents"]

patterns, tags = [], []
for intent in intents:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
clf = LogisticRegression(max_iter=200)
clf.fit(X, tags)

# Save model and vectorizer
pickle.dump((vectorizer, clf), open("chatbot_model.pkl", "wb"))
print("✅ Model trained and saved successfully!")
