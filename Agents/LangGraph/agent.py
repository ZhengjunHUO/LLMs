import streamlit as st
import json
import os
import uuid
from datetime import datetime

def save_conversations():
    try:
        with open("conversations.json", "w") as f:
            json.dump(st.session_state.conversations, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save conversations: {e}")

def load_conversations():
    try:
        if os.path.exists("conversations.json"):
            with open("conversations.json", "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}
    return {}

def save_current_conversation():
    if st.session_state.current_chat_id:
        title = "New Chat"
        if st.session_state.messages:
            first_message = st.session_state.messages[0]["content"]
            title = first_message[:50] + "..." if len(first_message) > 50 else first_message
        
        st.session_state.conversations[st.session_state.current_chat_id] = {
            "title": title,
            "messages": st.session_state.messages,
            "created_at": datetime.now().isoformat()
        }
        
        # Persist to disk
        save_conversations()

def initialize_session_state():
    if "initialized" not in st.session_state:
        st.session_state.conversations = load_conversations()
        st.session_state.current_chat_id = None
        st.session_state.messages = []
        st.session_state.initialized = True

initialize_session_state()

st.set_page_config(
    page_title="Chat Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.header("Chat History")
    if st.button("+ New Chat", use_container_width=True):
        new_chat_id = str(uuid.uuid4())
        st.session_state.current_chat_id = new_chat_id
        st.session_state.messages = []
        st.rerun()
    
    # Display chat history
    for chat_id, chat_data in st.session_state.conversations.items():
        chat_title = chat_data.get("title", "New Chat")
        if st.button(
            chat_title, 
            key=f"chat_{chat_id}",
            use_container_width=True,
            type="primary" if chat_id == st.session_state.current_chat_id else "secondary"
        ):
            st.session_state.current_chat_id = chat_id
            st.session_state.messages = chat_data.get("messages", [])
            st.rerun()

st.header("Chat Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            #response = process_with_langgraph(prompt)
            response = "bla bla bla"
            st.write(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    save_current_conversation()
