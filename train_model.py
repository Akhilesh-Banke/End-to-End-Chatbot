from pathlib import Path
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "Data" / "intent.json"
MODEL_PATH = BASE / "chatbot_model.pkl"

# Load intents
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Could not find intents file at: {DATA_PATH}")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)["intents"]

patterns, tags = [], []
for intent in intents:
    for pattern in intent.get("patterns", []):
        patterns.append(pattern)
        tags.append(intent["tag"])

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
clf = LogisticRegression(max_iter=200)
clf.fit(X, tags)

# Save model and vectorizer
with open(MODEL_PATH, "wb") as f:
    pickle.dump((vectorizer, clf), f)

print(f"✅ Model trained and saved successfully at: {MODEL_PATH}")
