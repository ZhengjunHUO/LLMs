## Prerequis pgvector
```sh
docker run --name pgvector -p 5432:5432 -e POSTGRES_PASSWORD=admin -d ankane/pgvector
psql -h 127.0.0.1 -U postgres -d langgraph
```
### for nomic-embed-text:v1.5, dimension == 768
```
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
