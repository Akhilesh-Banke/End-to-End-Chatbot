#  End-to-End Chatbot using Python and Streamlit

An **AI-powered chatbot** built using Python, NLP (TF-IDF + Logistic Regression), and Streamlit.  
This chatbot can understand user queries, classify intent, and respond intelligently — all within a clean web interface.

---

##  Project Overview

An **End-to-End Chatbot** refers to a conversational AI system that can handle complete conversations from start to finish — without human intervention.

This chatbot:
- Learns from pre-defined **intents and responses**.
- Uses **TF-IDF Vectorization** and **Logistic Regression** for intent classification.
- Provides a **Streamlit-based web UI** for real-time interaction.
- Can be extended with APIs, databases, or even transformer-based models like GPT.

---

##  Project Structure

```text
end-to-end-chatbot/
│
├── data/
│   └── intents.json
│
├── model/
│   └── train_model.py
│
├── chatbot.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
