import random
import json
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import nltk

# Ensure nltk data is available
nltk.download("punkt")
nltk.download("wordnet")

# Setup paths
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "Data"
MODEL_PATH = BASE / "chatbot_model.h5"
PICKLE_PATH = BASE / "chatbot_model.pkl"

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load saved model and data
model = load_model(MODEL_PATH)
with open(PICKLE_PATH, "rb") as f:
    data = pickle.load(f)

words = data["words"]
classes = data["classes"]
intents = data["intents"]

# Helper functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        # fallback if no intent found
        for intent in intents_json["intents"]:
            if intent["tag"] == "unknown":
                return random.choice(intent["responses"])
        return "I'm sorry, I didn't understand that."

    tag = intents_list[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm not sure I understand. Could you please rephrase?"

# Streamlit UI
st.set_page_config(page_title="AI Chatbot", page_icon="💬", layout="centered")

st.title("🤖 AI Chatbot Assistant")
st.write("Chat with your intelligent assistant trained on custom intents!")

# Chat input
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type your message here...")

if user_input:
    # Predict and get response
    ints = predict_class(user_input)
    res = get_response(ints, intents)

    # Add to chat history
    st.session_state.chat_history.append({"user": user_input, "bot": res})

# Display chat
for chat in st.session_state.chat_history:
    st.chat_message("user").write(chat["user"])
    st.chat_message("assistant").write(chat["bot"])
