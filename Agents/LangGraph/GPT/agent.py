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
        from typing import Annotated, Dict, Any, Optional
        from typing_extensions import TypedDict
        from langgraph.graph import StateGraph, START, END
        from langgraph.graph.message import add_messages
        from langgraph.prebuilt import ToolNode, tools_condition
        from langchain.tools import Tool
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from langchain_tavily import TavilySearch
        from langchain_core.embeddings import Embeddings
        from langchain_core.tools import BaseTool
        from pydantic import Field
        import subprocess
        import shlex
        import json
        import asyncpg
        import asyncio

        class State(TypedDict):
            messages: Annotated[list, add_messages]
            context: Optional[str]

        class VectorSearchTool(BaseTool):
            """Custom tool to search vector database"""
            name: str = "vector_search"
            description: str = "Search for relevant documents using semantic similarity. Use this when you need to find information from the knowledge base."

            # Declare fields with Pydantic
            db_config: Dict[str, Any] = Field(...)
            embeddings_model: Embeddings = Field(...)

            def __init__(self, db_config: Dict[str, Any], embeddings_model: Embeddings):
                super().__init__(
                    db_config = db_config,
                    embeddings_model = embeddings_model
                )

            async def _arun(self, query: str, top_k: int = 5, threshold: float = 0.7) -> str:
                """Async implementation for vector search"""
                try:
                    # Generate embedding for the query
                    query_embedding = await self.embeddings_model.aembed_query(query)
                    query_embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

                    # Connect to PostgreSQL
                    conn = await asyncpg.connect(**self.db_config)

                    # Perform vector similarity search
                    search_query = """
                    SELECT
                        id,
                        content,
                        metadata,
                        1 - (embedding <=> $1::vector) as similarity_score
                    FROM documents
                    WHERE 1 - (embedding <=> $1::vector) > $2
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3;
                    """

                    results = await conn.fetch(
                        search_query,
                        query_embedding_str,
                        threshold,
                        top_k
                    )

                    await conn.close()

                    if not results:
                        return "No relevant documents found."

                    # Format results
                    formatted_results = []
                    for row in results:
                        formatted_results.append({
                            "id": row["id"],
                            "content": row["content"],
                            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                            "similarity_score": float(row["similarity_score"])
                        })

                    # Create a readable summary
                    context = "\n\n".join([
                        f"Document {i+1} (Score: {result['similarity_score']:.3f}):\n{result['content']}"
                        for i, result in enumerate(formatted_results)
                    ])

                    print(f"[DEBUG] Get context: {context}")

                    return context

                except Exception as e:
                    return f"Error searching vector database: {str(e)}"

            def _run(self, query: str, top_k: int = 5, threshold: float = 0.7) -> str:
                return asyncio.run(self._arun(query, top_k, threshold))

        async def search_knowledge_base(state: State):
            """Node that searches the vector database"""
            last_message = state["messages"][-1]

            db_config = {
                "host": "localhost",
                "port": 5432,
                "database": "langgraph",
                "user": "postgres",
                "password": "admin"
            }

            embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
            vector_tool = VectorSearchTool(db_config, embeddings)

            # Search for relevant context
            context = await vector_tool._arun(last_message.content)

            return {"context": context}

        def chatbot(state: State):
            last_message = state["messages"][-1]
            context = state.get("context", "")
            if context and context != "No relevant documents found.":
                prompt = f"""Based on the knowledge base search, please answer the user's question.

Context:
{context}

Question: {last_message.content}

Please provide a helpful and accurate answer based on the context provided."""
                state["messages"][-1].content = prompt
            return { "messages": [llm_with_tools.invoke(state["messages"])]}

        ALLOWED_COMMANDS = {
            'pwd', 'ls', 'cat', 'echo', 'cp', 'mv', 'mkdir', 'touch', 'grep', 'find',
            'head', 'tail', 'wc', 'sort', 'date', 'curl', 'wget', 'du', 'df', 'python',
            'uname', 'ps', "free", "nvidia-smi"
        }

        def is_command_allowed(command: str) -> bool:
            cmd_parts = shlex.split(command)
            base_command = cmd_parts[0].split('/')[-1]  # Get command name
            return base_command in ALLOWED_COMMANDS

        def shell_command_tool(command: str) -> dict:
            """
            Executes a Linux shell command within a secure, isolated workspace.

            This tool is essential for tasks involving file system inspection,
            data manipulation, and system interaction. Use it when you need to:
            - List, read, or search for files (e.g., 'ls -l', 'cat my_file.txt', 'grep "error" logs/').
            - Check system status or network information (e.g., 'df -h', 'curl ifconfig.me').
            - Perform simple data processing with command-line tools like 'awk' or 'sed'.

            The command runs in the predefined workspace: /safe/agent/workdir.
            Do NOT use this for long-running processes or interactive sessions like 'ssh' or 'vim'.
            """
            if not is_command_allowed(command):
                return {"error": f"Command {command} is not allowed"}

            try:
                full_command = f"sudo -u agentexecutor /bin/bash -c 'cd /safe/agent/workdir && {command}'"
                print(f"[DEBUG] full_command: {full_command}")

                result = subprocess.run(
                    full_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    # cwd="/safe/agent/workdir",
                    # cwd=working_dir,
                    check=True,
                )

                return {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode,
                    "success": result.returncode == 0
                }
            except subprocess.TimeoutExpired:
                return {"error": "Command timed out"}
            except subprocess.CalledProcessError as e:
                return f"Error: Command failed with exit code {e.returncode}.\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
            except Exception as e:
                return {"error": f"Execution failed: {str(e)}"}

        # TODO: fallback to ddg ?
        search_tool = TavilySearch(
            max_results=5,
            topic="general",
        )
        shell_tool = Tool(
            name="shell_command",
            description="""Executes a Linux shell command within a secure, isolated workspace.
            This tool is essential for tasks involving file system inspection,
            data manipulation, and system interaction. Use it when you need to:
            - List, read, or search for files (e.g., 'ls -l', 'cat my_file.txt', 'grep "error" logs/').
            - Check system status or network information (e.g., 'df -h', 'curl ifconfig.me').
            - Perform simple data processing with command-line tools like 'awk' or 'sed'.
            The command runs in the predefined workspace: /safe/agent/workdir.
            Do NOT use this for long-running processes or interactive sessions like 'ssh' or 'vim'.""",
            func=shell_command_tool
        )
        tools = [search_tool, shell_tool]

        llm = ChatOllama(model="gpt-oss:20b", temperature=0)
        llm_with_tools = llm.bind_tools(tools)
        tools_node = ToolNode(tools=tools)

        builder = StateGraph(State)
        builder.add_node("search_knowledge_base", search_knowledge_base)
        builder.add_node("chatbot", chatbot)
        builder.add_node("tools", tools_node)
        builder.add_edge(START, "search_knowledge_base")
        builder.add_edge("search_knowledge_base", "chatbot")
        builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )
        builder.add_edge("tools", "chatbot")
        return builder.compile()

    async def async_run(self, messages: list) -> AsyncGenerator[str, None]:
        async for event in self.graph.astream({
            "messages": messages
        }):
            print(f"[DEBUG] Event: {event}")
            # {'chatbot': {'messages': [AIMessage(content='...', additional_kwargs={}, response_metadata={'model': 'gpt-oss:20b', 'created_at': '2025-08-29T10:34:54.83054935Z', 'done': True, 'done_reason': 'stop', 'total_duration': 5755868475, 'load_duration': 3876618943, 'prompt_eval_count': 1338, 'prompt_eval_duration': 1104694397, 'eval_count': 85, 'eval_duration': 769915896, 'model_name': 'gpt-oss:20b'}, id='run--8f10d57d-7aee-49c7-9de8-a2f66da4583b-0', usage_metadata={'input_tokens': 1338, 'output_tokens': 85, 'total_tokens': 1423})]}}
            if "chatbot" in event:
                node_output = event["chatbot"]

                if "messages" in node_output and node_output["messages"]:
                    latest_message = node_output["messages"][-1]

                    if hasattr(latest_message, 'content') and latest_message.content:
                        # yield latest_message.content

                        content = latest_message.content
                        sentences = content.split('. ')
                        for i, sentence in enumerate(sentences):
                            if i < len(sentences) - 1:
                                yield sentence + ". "
                            else:
                                yield sentence
                            await asyncio.sleep(0.1)

@st.cache_resource
def get_chatbot():
    """Create and cache the chatbot instance"""
    return ChatBot()

def stream_to_ui():
    async def collect_stream():
        chatbot = get_chatbot()
        full_response = ""
        placeholder = st.empty()

        async for chunk in chatbot.async_run(st.session_state.messages):
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

        response = stream_to_ui()

    st.session_state.messages.append({"role": "assistant", "content": response})

    save_current_conversation()
