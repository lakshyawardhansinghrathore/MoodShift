# app.py

import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from datetime import datetime
import time

# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Mood AI Chatbot",
    page_icon="🤖",
    layout="wide",
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

.stChatMessage {
    border-radius: 15px;
    padding: 10px;
}

.user-msg {
    background-color: #1e293b;
    padding: 12px;
    border-radius: 12px;
    color: white;
}

.bot-msg {
    background-color: #334155;
    padding: 12px;
    border-radius: 12px;
    color: white;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #38bdf8;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
}

.sidebar-title {
    font-size: 24px;
    font-weight: bold;
    color: #38bdf8;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown('<div class="title">🤖 Mood AI Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Chat with AI in different moods and personalities</div>',
    unsafe_allow_html=True
)

# =========================
# SIDEBAR
# =========================
with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">⚙️ Settings</div>',
        unsafe_allow_html=True
    )

    mood = st.selectbox(
        "Choose AI Mood",
        [
            "😡 Angry",
            "😂 Funny",
            "😢 Sad",
            "🧠 Motivational",
            "❤️ Romantic",
            "👨‍🏫 Teacher",
            "💻 Coding Expert"
        ]
    )

    temperature = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=1.5,
        value=0.9,
        step=0.1
    )

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    dark_mode = st.toggle("🌙 Dark Mode", value=True)

# =========================
# SYSTEM PROMPTS
# =========================
mood_prompts = {
    "😡 Angry":
        "You are an angry AI assistant. Reply aggressively and impatiently.",
    
    "😂 Funny":
        "You are a hilarious AI assistant. Use jokes, memes, and funny replies.",
    
    "😢 Sad":
        "You are a sad emotional AI assistant. Reply emotionally and sadly.",
    
    "🧠 Motivational":
        "You are a motivational coach AI. Inspire and motivate the user.",
    
    "❤️ Romantic":
        "You are a romantic AI assistant. Reply sweetly and affectionately.",
    
    "👨‍🏫 Teacher":
        "You are a helpful teacher AI. Explain concepts clearly and simply.",
    
    "💻 Coding Expert":
        "You are a senior software engineer AI. Help with coding professionally."
}

# =========================
# MODEL
# =========================
model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=temperature
)

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content=mood_prompts[mood])
    ]

# =========================
# UPDATE SYSTEM PROMPT
# =========================
st.session_state.chat_history[0] = SystemMessage(
    content=mood_prompts[mood]
)

# =========================
# DISPLAY CHAT HISTORY
# =========================
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-msg">{msg["content"]}</div>',
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                f'<div class="bot-msg">{msg["content"]}</div>',
                unsafe_allow_html=True
            )

# =========================
# CHAT INPUT
# =========================
user_input = st.chat_input("Type your message...")

if user_input:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    st.session_state.chat_history.append(
        HumanMessage(content=user_input)
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(
            f'<div class="user-msg">{user_input}</div>',
            unsafe_allow_html=True
        )

    # AI RESPONSE
    with st.chat_message("assistant"):

        message_placeholder = st.empty()

        full_response = ""

        try:
            response = model.invoke(
                st.session_state.chat_history
            )

            bot_response = response.content

            # Typing animation
            for char in bot_response:
                full_response += char
                time.sleep(0.01)

                message_placeholder.markdown(
                    f'<div class="bot-msg">{full_response}▌</div>',
                    unsafe_allow_html=True
                )

            message_placeholder.markdown(
                f'<div class="bot-msg">{full_response}</div>',
                unsafe_allow_html=True
            )

            # Save AI response
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

            st.session_state.chat_history.append(
                AIMessage(content=full_response)
            )

        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# FOOTER
# =========================
st.divider()

current_time = datetime.now().strftime("%I:%M %p")

st.caption(f"⏰ Current Time: {current_time}")
st.caption("Built with Streamlit + LangChain + Mistral AI")