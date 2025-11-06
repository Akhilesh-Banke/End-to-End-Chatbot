import streamlit as st
from Chatbot import get_response

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

st.title("🤖 End-to-End Chatbot")
st.markdown("Welcome! Type a message below to start chatting.")

# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:", key="input")

if user_input:
    response = get_response(user_input)
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", response))

# Display chat history
for sender, msg in st.session_state.history:
    if sender == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")
