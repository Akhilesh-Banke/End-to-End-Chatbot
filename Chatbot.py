from pathlib import Path
import random
import json
import pickle

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "Data" / "intent.json"
MODEL_PATH = BASE / "chatbot_model.pkl"

# Load intents
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Could not find intents file at: {DATA_PATH}")
with open(DATA_PATH, "r", encoding="utf-8") as file:
    intents = json.load(file).get("intents", [])

# Load saved model and vectorizer
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found. Please run `python train_model.py` to create it at: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    vectorizer, clf = pickle.load(f)

def get_response(user_input):
    # basic guard
    if not user_input or not user_input.strip():
        return "Please type something so I can help."

    input_vec = vectorizer.transform([user_input])
    tag = clf.predict(input_vec)[0]

    # prefer exact matching intent
    for intent in intents:
        if intent.get("tag") == tag:
            return random.choice(intent.get("responses", ["Sorry, I don't have a response right now."]))

    # fallback: try unknown tag or default message
    for intent in intents:
        if intent.get("tag") == "unknown":
            return random.choice(intent.get("responses", ["I'm not sure about that. Could you rephrase?"]))

    # final fallback
    return "I'm not sure about that. Could you rephrase?"
