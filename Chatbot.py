import random
import json
import pickle

# Load saved model and vectorizer
vectorizer, clf = pickle.load(open("chatbot_model.pkl", "rb"))

# Load intents
with open("data/intents.json", "r") as file:
    intents = json.load(file)["intents"]

def get_response(user_input):
    input_vec = vectorizer.transform([user_input])
    tag = clf.predict(input_vec)[0]

    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    # fallback if not found
    for intent in intents:
        if intent["tag"] == "unknown":
            return random.choice(intent["responses"])
