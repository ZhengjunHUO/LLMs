import asyncpg
import numpy as np
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import Field
import json

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
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
        import asyncio
        return asyncio.run(self._arun(query, top_k, threshold))

async def search_knowledge_base(state: AgentState) -> AgentState:
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

async def generate_response(state: AgentState) -> AgentState:
    """Node that generates response using retrieved context"""
    last_message = state["messages"][-1]
    context = state.get("context", "")
    
    # Create a prompt with context
    if context and context != "No relevant documents found.":
        prompt = f"""Based on the knowledge base search, please answer the user's question.

Context:
{context}

Question: {last_message.content}

Please provide a helpful and accurate answer based on the context provided."""
    else:
        prompt = f"I couldn't find relevant information in the knowledge base. Question: {last_message.content}"
    
    # Here you would typically use your LLM to generate a response
    # For this example, we'll create a simple response
    response = AIMessage(content=prompt)
    
    return {"messages": [response]}

# Helper function to add documents to the vector database
async def add_document_to_vectordb(
    content: str, 
    metadata: Dict[str, Any],
    db_config: Dict[str, Any],
    embeddings_model: Embeddings,
    doc_id: Optional[str] = None
):
    """Add a document to the vector database"""
    
    # Generate embedding
    embedding = await embeddings_model.aembed_query(content)
    embedding_str = '[' + ','.join(map(str, embedding)) + ']'

    # Connect to database
    conn = await asyncpg.connect(**db_config)
    
    # Insert document
    insert_query = """
    INSERT INTO documents (id, content, metadata, embedding)
    VALUES ($1, $2, $3, $4::vector)
    ON CONFLICT (id) DO UPDATE SET
        content = EXCLUDED.content,
        metadata = EXCLUDED.metadata,
        embedding = EXCLUDED.embedding;
    """
    
    await conn.execute(
        insert_query,
        doc_id or f"doc_{hash(content)}",
        content,
        json.dumps(metadata),
        embedding_str
    )
    
    await conn.close()

# Create the LangGraph workflow
def create_vector_search_graph():
    """Create a LangGraph that uses vector database for RAG"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("search_knowledge", search_knowledge_base)
    workflow.add_node("generate_response", generate_response)
    
    # Define the flow
    workflow.set_entry_point("search_knowledge")
    workflow.add_edge("search_knowledge", "generate_response")
    workflow.add_edge("generate_response", END)
    
    return workflow.compile()

async def main():
    # # Database configuration
    # db_config = {
    #     "host": "localhost",
    #     "port": 5432,
    #     "database": "langgraph",
    #     "user": "postgres",
    #     "password": "admin"
    # }
    
    # # Initialize embeddings
    # embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
    
    # # Add some sample documents (run once to populate)
    # sample_docs = [
    #     {
    #         "content": "PostgreSQL is a powerful, open source object-relational database system with over 35 years of active development.",
    #         "metadata": {"source": "postgresql_docs", "topic": "database"}
    #     },
    #     {
    #         "content": "LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows.",
    #         "metadata": {"source": "langgraph_docs", "topic": "ai_framework"}
    #     }
    # ]
    
    # for doc in sample_docs:
    #     await add_document_to_vectordb(
    #         doc["content"], 
    #         doc["metadata"], 
    #         db_config, 
    #         embeddings
    #     )
    
    # Create and run the graph
    app = create_vector_search_graph()
    
    # Example query
    initial_state = {
        "messages": [HumanMessage(content="What is PostgreSQL?")],
        "context": None
    }
    
    result = await app.ainvoke(initial_state)
    print("Final response:", result["messages"][-1].content)

# Database setup SQL
"""
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster vector similarity search
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents USING hnsw (embedding vector_cosine_ops);

-- Create index on metadata for filtering
CREATE INDEX IF NOT EXISTS documents_metadata_idx 
ON documents USING gin (metadata);
"""

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())