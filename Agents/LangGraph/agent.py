import streamlit as st
import asyncio
from typing import AsyncGenerator
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
        
        save_conversations()

def initialize_session_state():
    if "initialized" not in st.session_state:
        st.session_state.conversations = load_conversations()
        st.session_state.current_chat_id = None
        st.session_state.messages = []
        st.session_state.initialized = True

class ChatBot:
    def __init__(self):
        self.graph = self._create_graph()

    def _create_graph(self):
        """Create the LangGraph instance"""
        from typing import Annotated
        from typing_extensions import TypedDict
        from langgraph.graph import StateGraph, START, END
        from langgraph.graph.message import add_messages
        from langgraph.prebuilt import ToolNode, tools_condition
        from langchain_ollama import ChatOllama
        from langchain_tavily import TavilySearch

        class State(TypedDict):
            messages: Annotated[list, add_messages]

        def chatbot(state: State):
            return { "messages": [llm_with_tools.invoke(state["messages"])]}

        llm = ChatOllama(model="gpt-oss:20b", temperature=0)
        search_tool = TavilySearch(
            max_results=5,
            topic="general",
        )
        tools = [search_tool]

        llm_with_tools = llm.bind_tools(tools)
        tools_node = ToolNode(tools=tools)

        builder = StateGraph(State)
        builder.add_node("chatbot", chatbot)
        builder.add_node("tools", tools_node)
        builder.add_edge(START, "chatbot")
        builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )
        builder.add_edge("tools", "chatbot")
        return builder.compile()

    async def async_run(self, question: str) -> AsyncGenerator[str, None]:
        async for event in self.graph.astream({
            "messages": [
                {"role": "user", "content": question}
            ]
        }):
            print(f"[DEBUG] Event: {event}")
            if "chatbot" in event:
                node_output = event["chatbot"]

                if "messages" in node_output and node_output["messages"]:
                    latest_message = node_output["messages"][-1]

                    if hasattr(latest_message, 'content') and latest_message.content:
                        yield latest_message.content

@st.cache_resource
def get_chatbot():
    """Create and cache the chatbot instance"""
    return ChatBot()

# def process_with_langgraph(user_input):
#     chatbot = get_chatbot()
#     return chatbot.async_run(user_input)

def stream_to_ui(user_input: str):
    async def collect_stream():
        chatbot = get_chatbot()
        full_response = ""
        placeholder = st.empty()

        async for chunk in chatbot.async_run(user_input):
            full_response += chunk
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)
        return full_response

    return asyncio.run(collect_stream())

initialize_session_state()

st.set_page_config(
    page_title="Chat Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.header("Chat Assistant")

with st.sidebar:
    st.header("Chat History")
    if st.button("+ New Chat", use_container_width=True):
        new_chat_id = str(uuid.uuid4())
        st.session_state.current_chat_id = new_chat_id
        st.session_state.messages = []
        st.rerun()
    
    # Display chat history in side bar
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

if st.sidebar.button("🗑️ Delete Chat", disabled=not st.session_state.current_chat_id):
    if st.session_state.current_chat_id in st.session_state.conversations:
        del st.session_state.conversations[st.session_state.current_chat_id]
        st.session_state.current_chat_id = None
        st.session_state.messages = []
        save_conversations()
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Type your message..."):
    # When the app is up for the the first time, the id is None
    if st.session_state.current_chat_id is None:
        st.session_state.current_chat_id = str(uuid.uuid4())

    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        # with st.spinner("Thinking..."):
        #     #response = process_with_langgraph(prompt)
        #     response = "bla bla bla"
        #     st.write(response)

        # placeholder = st.empty()
        # response = ""
        # for chunk in process_with_langgraph(prompt):  # streaming
        #     response += chunk
        #     placeholder.markdown(response + "▌")
        # placeholder.markdown(response)

        response = stream_to_ui(prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})

    save_current_conversation()
