# Context7 Architecture Analysis: From Raw Docs to Relevant AI Context

This document analyzes how Context7 processes raw documentation and provides highly relevant context to AI assistants.

## Overview

Context7's architecture separates into **two key systems**:

1. **Backend (Private)**: Crawling, parsing, indexing, and ranking engine
2. **Frontend (This Repo)**: MCP server, SDK, and tools that serve the processed documentation

**Note**: The backend systems (crawler, parser, indexing, and ranking algorithms) are **not in this repository**. This analysis focuses on what we can observe from the client-side implementation.

---

## The Two-Phase System

### Phase 1: Library Resolution (Search & Match)

**Endpoint**: `GET /v2/libs/search`

**Parameters**:
- `query`: User's question/task (e.g., "I need to build authentication with JWT")
- `libraryName`: Library to search for (e.g., "express", "react")

**What Happens**:

1. **Backend Ranking** (Private Logic):
   - The API receives both the `query` and `libraryName`
   - Returns libraries sorted by relevance to the user's query
   - Uses multiple scoring factors (described below)

2. **Response Format**:
```typescript
{
  results: [
    {
      id: "/facebook/react",              // Context7 library ID
      title: "React",                     // Display name
      description: "A JavaScript library for building user interfaces",
      totalSnippets: 15234,               // Number of indexed code examples
      trustScore: 8.5,                    // Source reputation (0-10)
      benchmarkScore: 95,                 // Quality indicator (0-100)
      versions: ["v18.2.0", "v17.0.2"],   // Available versions
      state: "finalized"                  // Processing status
    }
    // ... more results
  ]
}
```

3. **Client-Side Interpretation** (`packages/sdk/src/utils/format.ts`):
```typescript
// Trust score mapping
trustScore >= 7  → "High" reputation
trustScore >= 4  → "Medium" reputation
trustScore < 4   → "Low" reputation
```

**Key Insight**: The `query` parameter is **critical** - it's not just finding libraries by name, but **ranking them by relevance to what the user is trying to accomplish**.

---

### Phase 2: Context Retrieval (Query & Rank Documentation)

**Endpoint**: `GET /v2/context`

**Parameters**:
- `query`: Specific question (e.g., "How to use useEffect cleanup functions")
- `libraryId`: Resolved library ID (e.g., "/facebook/react")
- `type`: Response format ("json" or "txt")

**What Happens**:

1. **Backend Semantic Ranking** (Private Logic):
   - Takes the user's natural language query
   - Searches through indexed documentation snippets
   - **Ranks and reranks** results by relevance to the query
   - Returns the most relevant snippets (not all documentation)

2. **Response Format** (JSON type):
```typescript
{
  codeSnippets: [
    {
      codeTitle: "Using the Effect Hook",
      codeDescription: "Effects may optionally specify how to clean up after them",
      codeLanguage: "javascript",
      codeList: [
        {
          language: "javascript",
          code: "useEffect(() => {\n  // subscription logic\n  return () => {\n    // cleanup\n  };\n});"
        }
      ],
      codeId: "react-hooks-effect-cleanup",
      codeTokens: 145
    }
  ],
  infoSnippets: [
    {
      content: "# Cleaning up an effect\n\nSome effects need to specify...",
      breadcrumb: "React > Hooks > useEffect",
      pageId: "react-docs-hooks-effect",
      contentTokens: 423
    }
  ]
}
```

3. **Client-Side Formatting** (`packages/sdk/src/utils/format.ts`):

The SDK transforms API responses into structured Documentation objects:

```typescript
// Code snippets become:
{
  title: "Using the Effect Hook",
  content: "Effects may optionally specify...\n\n```javascript\nuseEffect(...);\n```",
  source: "react-hooks-effect-cleanup"
}

// Info snippets become:
{
  title: "React > Hooks > useEffect",
  content: "# Cleaning up an effect\n\nSome effects need to specify...",
  source: "react-docs-hooks-effect"
}
```

**Text Format**: When `type: "txt"` is requested, the backend returns pre-formatted markdown text optimized for LLM consumption.

---

## The Ranking & Relevance System

### Library Ranking Factors

From the MCP tool descriptions and code structure, we can infer the ranking considers:

1. **Name Similarity**: Exact matches prioritized
2. **Query Relevance**: Description matches user's intent
3. **Documentation Coverage**: Higher `totalSnippets` = more comprehensive
4. **Source Reputation**: `trustScore` indicates authority
   - High (≥7): Official docs, authoritative sources
   - Medium (4-7): Community docs, established sources
   - Low (<4): Less verified sources
5. **Quality Indicator**: `benchmarkScore` (0-100) - quality metric
6. **Version Availability**: Supports version-specific documentation

### Documentation Ranking (The "Secret Sauce")

**What We Know**:

1. **Semantic Search**: The backend performs semantic/vector search on documentation
   - Not simple keyword matching
   - Understanding user intent from natural language queries

2. **Intelligent Reranking**: Results are sorted by relevance
   - Most relevant snippets first
   - Balances code examples vs. explanatory text
   - Considers token limits (note `codeTokens` and `contentTokens` fields)

3. **Snippet Types**:
   - **Code Snippets**: Executable examples with descriptions
   - **Info Snippets**: Explanatory documentation with breadcrumb context

**What We Don't See** (Private Backend):
- Vector embedding models used
- Reranking algorithms
- How documentation is chunked and indexed
- Query understanding and expansion logic
- Token limit optimization strategies

---

## How AI Assistants Use This System

### MCP Server Flow (`packages/mcp/src/index.ts`)

The MCP server provides two tools with detailed instructions:

1. **`resolve-library-id`** Tool:
```
Description: "Resolves a package/product name to a Context7-compatible library ID"

Instructions to AI:
- Call this BEFORE query-docs (unless user provides explicit ID)
- Analyze query to understand what library user needs
- Select most relevant match based on:
  * Name similarity (exact matches prioritized)
  * Description relevance to query intent
  * Documentation coverage (higher snippet counts)
  * Source reputation (High/Medium preferred)
  * Benchmark score (higher is better)
- Limit: Max 3 calls per question
```

2. **`query-docs`** Tool:
```
Description: "Retrieves and queries up-to-date documentation"

Instructions to AI:
- Must call resolve-library-id first
- Be specific in queries (good: "How to set up JWT auth in Express", bad: "auth")
- Limit: Max 3 calls per question
```

### AI SDK Agent Flow (`packages/tools-ai-sdk/src/agents/context7.ts`)

The agent follows a **strict multi-step workflow**:

```typescript
AGENT_PROMPT = `
CRITICAL WORKFLOW - YOU MUST FOLLOW THESE STEPS:

Step 1: ALWAYS call 'resolveLibraryId' with library name
   - Extract main library name from user query
   - Review ALL search results returned

Step 2: Analyze results and select BEST library ID based on:
   - Official sources (e.g., /reactjs/react.dev for React)
   - Name similarity
   - Description relevance
   - Source reputation (High/Medium is better)
   - Code snippet coverage (higher is better)
   - Benchmark score (higher is better)

Step 3: Call 'queryDocs' with selected library ID and user's query
   - Use exact library ID from resolveLibraryId
   - Include user's original question

Step 4: Provide clear answer with code examples

IMPORTANT:
- Do NOT skip resolveLibraryId
- Do not call either tool more than 3 times per question
- Always cite which library ID you used
`
```

---

## Network & Performance Optimizations

### HTTP Client (`packages/sdk/src/http/index.ts`)

**Retry Logic**:
```typescript
{
  attempts: 5,                                    // Retry up to 5 times
  backoff: (retryCount) => Math.exp(retryCount) * 50  // Exponential backoff
}
```

**Cache Settings**:
- Default: `"no-store"` (always fetch fresh)
- Configurable per client

**Response Headers** (for text format):
```typescript
{
  "x-context7-page": "1",
  "x-context7-limit": "10",
  "x-context7-total-pages": "5",
  "x-context7-has-next": "true",
  "x-context7-total-tokens": "12450"  // Token count for context management
}
```

### Telemetry & Tracking

The MCP server sends telemetry headers (`packages/mcp/src/lib/encryption.ts`):

```typescript
{
  "X-Context7-Source": "mcp-server",
  "X-Context7-Server-Version": "2.1.0",
  "X-Context7-Client-IDE": "Cursor",              // Extracted from User-Agent
  "X-Context7-Client-Version": "2.2.44",
  "X-Context7-Transport": "stdio" | "http",
  "mcp-client-ip": "<encrypted-ip>",             // AES-256-CBC encrypted
  "Authorization": "Bearer <api-key>"
}
```

**Privacy**: Client IPs are encrypted with AES-256-CBC before transmission.

---

## Quality Signals & Trust Indicators

### Trust Score System

The trust score (0-10) indicates source authority:

```typescript
function getTrustScoreLabel(score?: number) {
  if (score >= 7) return "High";     // Official docs, verified sources
  if (score >= 4) return "Medium";   // Community docs, established
  return "Low";                       // Less verified
}
```

**Likely Factors** (inferred, not documented):
- Official documentation sites
- GitHub stars/activity
- Community consensus
- Documentation freshness
- Source maintainability

### Benchmark Score

Quality indicator (0-100) representing:
- Documentation completeness
- Code example quality
- Structure and organization
- Search performance

**Note**: Exact calculation is private backend logic.

---

## Token Management

### Why Token Counts Matter

Each snippet includes token counts:
- `codeTokens`: Tokens in code example
- `contentTokens`: Tokens in documentation text

**Purpose**:
- Help AI assistants manage context windows
- Allow selective inclusion of snippets
- Enable pagination for large result sets

### Response Format Selection

**JSON Format** (`type: "json"`):
- Structured data with separate code/info snippets
- Client can filter/format as needed
- Best for programmatic processing

**Text Format** (`type: "txt"`):
- Pre-formatted markdown string
- Optimized for direct LLM consumption
- Includes separators and structure
- Best for simple AI tool integration

---

## Key Architectural Insights

### 1. Query-Driven Design

Every API call takes a `query` parameter representing **user intent**, not just keywords. This enables:
- Semantic matching beyond keyword search
- Relevance ranking based on task context
- Better results for natural language questions

### 2. Two-Stage Resolution

The system doesn't just search documentation:
1. **First**: Find the right library (with relevance ranking)
2. **Second**: Find the right docs in that library (with query-specific ranking)

This two-stage approach provides better precision than flat search.

### 3. LLM-Optimized Responses

Responses are designed for LLM consumption:
- Clear structured format
- Breadcrumb context for info snippets
- Code blocks with language tags
- Source citations for verification
- Token counts for context management

### 4. Workflow Enforcement

The MCP tools and agent prompts **enforce a specific workflow**:
- Must resolve library ID first (prevents hallucinated library IDs)
- Limit tool calls to 3 per question (prevents loops)
- Require specific, detailed queries (improves results)

### 5. Quality Over Quantity

Rather than returning all documentation:
- Returns **most relevant** snippets
- Balances code examples with explanations
- Considers source reputation
- Provides version-specific docs

---

## What We Don't See (Private Backend)

The following are implemented server-side and not in this repo:

1. **Crawling Engine**:
   - How documentation sites are discovered and crawled
   - Update frequency and change detection
   - Handling different documentation formats

2. **Parsing & Indexing**:
   - Documentation parsing logic
   - Code extraction and classification
   - Chunk size and overlap strategies
   - Vector embedding generation

3. **Ranking Algorithms**:
   - Semantic search model architecture
   - Reranking strategies
   - Query understanding and expansion
   - Trust score calculation
   - Benchmark score calculation

4. **Infrastructure**:
   - Database schema and storage
   - Vector database implementation
   - Caching layers
   - Rate limiting logic

---

## Conclusion

Context7's effectiveness comes from:

1. **Query-Aware Ranking**: Using user intent to rank both libraries and documentation
2. **Quality Signals**: Trust scores and benchmark scores guide selection
3. **Structured Workflow**: Enforced two-stage process prevents common AI mistakes
4. **LLM Optimization**: Responses formatted specifically for AI consumption
5. **Token Management**: Providing counts helps AIs manage context windows

The **private backend** handles the heavy lifting of crawling, parsing, and intelligent ranking. The **public client** (this repo) provides clean interfaces and enforces best practices through structured prompts and tool descriptions.

The "secret sauce" is likely in:
- The semantic search and reranking algorithms (server-side)
- How documentation is chunked and indexed (server-side)
- The trust scoring system (server-side)
- Query understanding and expansion (server-side)

What we see in this repo is the **clever integration layer** that guides AI assistants to use these powerful backend capabilities effectively.


---
  Key Findings: How Context7 Achieves High Relevance

  The Two-Phase System

  1. Library Resolution (/v2/libs/search):
    - Takes both libraryName AND the user's query (their actual question/task)
    - Backend ranks libraries by relevance to what the user is trying to 
  accomplish
    - Returns scored results with:
        - trustScore (0-10): Source reputation/authority
      - benchmarkScore (0-100): Quality indicator
      - totalSnippets: Documentation coverage
      - versions: Available versions
  2. Context Retrieval (/v2/context):
    - Takes user's specific query + resolved libraryId
    - Backend performs semantic search and intelligent reranking
    - Returns only the most relevant snippets (not all docs)
    - Separates into codeSnippets (executable examples) and infoSnippets
  (explanations)

  The "Secret Sauce" (Based on Code Analysis)

  What We Can Observe:

  1. Query-Driven Architecture: Every API call includes the user's natural
  language query, enabling semantic matching beyond keywords
  2. Multi-Factor Ranking:
  Library Selection Based On:
  - Name similarity (exact matches prioritized)
  - Query relevance (description matches intent)
  - Documentation coverage (totalSnippets count)
  - Source reputation (trustScore: High ≥7, Medium ≥4)
  - Quality indicator (benchmarkScore 0-100)
  3. Structured Workflow Enforcement: The MCP tools force AI assistants to:
    - Always resolve library ID first (prevents hallucination)
    - Limit to 3 tool calls per question (prevents loops)
    - Use specific, detailed queries (improves results)
  4. Token-Aware Responses: Each snippet includes codeTokens and contentTokens
  to help AI manage context windows
  5. LLM-Optimized Formatting:
    - Pre-formatted markdown for direct consumption
    - Breadcrumb context for navigation
    - Code blocks with language tags
    - Source citations for verification

  What's Private (Backend):

  The actual ranking algorithms, vector embeddings, document parsing, crawling
  logic, and trust score calculations are server-side and not in this
  repository. The client-side code shows us the interface and scoring factors,
  but not the implementation.

  Key Architectural Insights

  - Two-stage resolution prevents poor results from flat search
  - Query context flows through every API call for relevance
  - Quality signals (trust/benchmark scores) guide selection
  - Strict prompts in MCP tools enforce best practices
  - Retry logic (5 attempts, exponential backoff) ensures reliability

  The genius is in combining server-side semantic ranking with client-side
  workflow enforcement - the backend does intelligent ranking while the client
  ensures AI assistants use it correctly.
