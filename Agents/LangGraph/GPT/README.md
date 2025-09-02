## Prerequis
```sh
# Prepare shell tool user & dir
sudo adduser --system --shell /bin/bash --group --no-create-home agentexecutor
sudo mkdir -p /safe/agent/workdir
sudo chown agentexecutor: /safe/agent/workdir
sudo chmod 700 /safe/agent/workdir
# Add one line in sudoer
# ++ user_exec_script ALL=(agentexecutor) NOPASSWD: ALL

# Prepare search tool API token
export TAVILY_API_KEY=tvly-foobar

# Underlying LLM
ollama pull nomic-embed-text:v1.5
ollama pull gpt-oss:20b
```

### For RAG
```
docker run --name pgvector -p 5432:5432 -e POSTGRES_PASSWORD=admin -e POSTGRES_DB=langgraph -d ankane/pgvector
psql -h 127.0.0.1 -U postgres -d langgraph

CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS documents_metadata_idx ON documents USING gin (metadata);
```

## Run
```
export TAVILY_API_KEY=tvly-foobar
docker start pgvector
streamlit run agent.py
```
