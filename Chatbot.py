import random
import json
import pickle
import os
from pathlib import Path
import numpy as np

# --- Base Directory ---
BASE = Path(__file__).resolve().parent

# --- Load Model and Vectorizer ---
model_path = BASE / "chatbot_model.pkl"
if not model_path.exists():
    raise FileNotFoundError("Model file not found. Please run train_model.py first.")

with open(model_path, "rb") as f:
    vectorizer, clf = pickle.load(f)

# --- Load Intents File ---
intents_path = BASE / "Data" / "intent.json"
if not intents_path.exists():
    raise FileNotFoundError("Intents file not found. Please check the path: data/intents.json")

with open(intents_path, "r", encoding="utf-8") as file:
    intents = json.load(file)["intents"]

# --- Response Function ---
def get_response(user_input):
    """Generate a chatbot response with fallback handling and confidence threshold."""
    input_vec = vectorizer.transform([user_input])
    probs = clf.predict_proba(input_vec)[0]
    tag_index = np.argmax(probs)
    tag = clf.classes_[tag_index]
    confidence = probs[tag_index]

    # Confidence threshold (tune this for accuracy)
    if confidence < 0.4:
        tag = "unknown"

    # Match response
    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    # Fallback if not matched
    for intent in intents:
        if intent["tag"] == "unknown":
            return random.choice(intent["responses"])

    # Ultimate fallback (should not occur)
    return "I'm not sure I understand that."
