#  End-to-End Chatbot using Python and Streamlit

An **AI powered chatbot** built using **TensorFlow**, **Keras**, **NLTK**, and **Streamlit**.  
This project demonstrates how to train a deep learning model on conversational intents and deploy it as an interactive chatbot web app.

---
##  Project Overview

The **End-to-End Chatbot System** is a complete conversational AI pipeline built from scratch — starting from raw text data to a fully functional web-based chatbot interface.

This project demonstrates the full journey of building an intelligent chatbot — from dataset creation, text preprocessing, and model training, to a user-friendly deployment with Streamlit.

**User Input → NLP Preprocessing → Model Prediction → Intent Detection → Response Selection → Display in Streamlit UI**

---
##  Features

- **Intent-based chatbot** using Neural Networks (Keras + TensorFlow)
- **Natural language understanding** with tokenization & lemmatization (NLTK)
- **Persistent trained model** — train once, reuse anytime
- **Streamlit-based interactive web UI**
- **Easily extendable intents dataset (`intents.json`)**

---

##  Project Structure

```text
End-to-End-Chatbot/
│
├── .idea/ # IDE configuration files (optional)
├── Data/
│     └── intents.json # Contains training data
├── pycache/ # Auto-generated Python cache files
│
├── Chatbot.py # Core chatbot logic (loads model, predicts responses)
├── train_model.py # Script to train the chatbot model
├── app.py # Streamlit web app for interactive chat interface
│
├── chatbot_model.h5 # Trained deep learning model
├── chatbot_model.pkl # Pickle file (tokenizer/label encoder)
│
├── requirements.txt # Python dependencies
├── package.json # (Optional) For npm or Streamlit Cloud setup
└── README.md # Project documentation

```

## Installation
Clone my repository:

```bash
https://github.com/Akhilesh-Banke/End-to-End-Chatbot.git
```

## Run the Chatbot App
```bash
streamlit run app.py
```

Visit the browser at: https://end-to-end-chatbot-akhilesh.streamlit.app
To Chat with my small AI assistant!
