# PRD / Technical Spec: CoreRAG (SOTA Agentic RAG Engine)

## 1. Executive Summary
**Objective:** Architect a low-latency, hallucination-resistant RAG microservice to query dense AI Systems literature.
**Target Audience:** Open-source portfolio piece demonstrating SOTA AI Systems engineering for MS admissions and IIT Delhi BTP pivot (targeting Prof. Sougata Mukherjea or similar labs).
**Core Value Proposition:** Solves the precision/recall trade-off using a Two-Stage Retrieval pipeline (Bi-Encoder + Cross-Encoder) and mitigates hallucinations via deterministic LangGraph Reflection loops.

## 2. System Architecture & Flow
The system operates as an asynchronous FastAPI microservice, moving beyond standard LangChain wrappers to implement 2026 production patterns.

**A. Pre-Retrieval (Ingestion Pipeline)**
1.  **Contextual Chunking:** Before embedding, an LLM prepends a 1-2 sentence global summary of the parent document to every chunk.
2.  **Propositional Extraction:** Complex chunks are optionally broken down into atomic facts (propositions) to improve dense vector matching.

**B. Query Pipeline (State Machine)**
1.  **Request Layer:** Client submits a query via POST `/query`.
2.  **Semantic Cache:** Query is embedded. If cosine similarity against the `RedisVL` cache is $\ge 0.90$, return cached answer instantly (bypass LLM/Retrieval).
3.  **Agentic Routing (LangGraph State Machine):** 
    *   *Note: Explicitly avoids ReAct/Reflexion paradigms to prevent infinite loops and latency spikes.*
    *   The Router node evaluates the query to determine if it requires decomposition (Multi-Hop) or standard retrieval.
4.  **Two-Stage Retrieval:**
    *   *Stage 1 (High Recall, Fast):* Qdrant executes Hybrid Search (Dense Vectors + BM25) with Reciprocal Rank Fusion (RRF) to pull $K=100$ candidate chunks.
    *   *Stage 2 (High Precision, Compute Heavy):* A Cross-Encoder (`GTE-reranker-modernbert-base`) reranks the 100 candidates by running full attention across the `(query + document)` pairs, returning the absolute top $N=5$.
5.  **Reflection (Grader Node):** A fast LLM evaluates the top 5 chunks. If `IsRel` (Relevance) is low, it rewrites the query and loops back to Stage 1 (max 2 retries).
6.  **Generation:** Groq LLM synthesizes the final answer. Payload is streamed back via Server-Sent Events (SSE) and written to the cache.

## 3. Technical Stack
| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Backend API** | FastAPI (Python) | Asynchronous request handling. |
| **Orchestration** | LangGraph | Hardcoded state machine control flow (Router $\rightarrow$ Retriever $\rightarrow$ Grader). |
| **Vector DB** | Qdrant | Natively supports Dense + Sparse (BM25). |
| **Stage 1 (Bi-Encoder)** | OpenAI `text-embedding-3-small` | Fast, cost-effective semantic representation. |
| **Stage 2 (Cross-Encoder)** | `Alibaba-NLP/gte-reranker-modernbert-base` | SOTA latency/accuracy for reranking candidates. |
| **LLM (Agents/Reflection)**| OpenAI `gpt-4o-mini` | Fast, cheap function calling for the Router/Grader nodes. |
| **LLM (Generation)** | Groq `llama3-70b-8192` | Ultra-low latency token generation. |
| **Dataset** | ArXiv API (cs.DC, cs.AR) | 150 AI Systems papers. |
| **Evaluation** | RAGAS | Quantitative benchmarking (Faithfulness, Precision). |

## 4. Prioritized Implementation Milestones

### Phase 1: Advanced Data Ingestion (P0)
*   [ ] Fetch 150 PDFs from ArXiv API.
*   [ ] Implement Contextual Chunking script (passing document text to `gpt-4o-mini` to generate chunk prefixes).
*   [ ] Index embeddings and BM25 payload into Qdrant.

### Phase 2: Two-Stage Retrieval (P0)
*   [ ] Implement Qdrant Hybrid Search endpoint ($K=100$).
*   [ ] Integrate `sentence-transformers` Cross-Encoder to rerank to $N=5$.
*   [ ] Implement Redis score caching for the Cross-Encoder `(query, doc_id)` pairs to reduce latency on repeated sub-queries.

### Phase 3: LangGraph Orchestration & Caching (P0)
*   [ ] Define the state graph schema (Router $\rightarrow$ Retriever $\rightarrow$ Grader $\rightarrow$ Generator).
*   [ ] Implement the Reflection Grader node logic and max-retry circuit breaker.
*   [ ] Implement the Semantic Cache interceptor layer before the LangGraph invocation.

### Phase 4: GraphRAG Expansion (P1 - For Prof. Mukherjea pitch)
*   [ ] Spin up Neo4j container.
*   [ ] Implement entity/relationship extraction during ingestion to build a parallel Knowledge Graph.
*   [ ] Add tool logic to the Router node to choose between Qdrant (Factual queries) and Neo4j (Global trend queries).

## 5. Repository Structure
```text
corerag/
├── api/                    
│   ├── main.py             
│   └── routes.py           
├── core/                   
│   ├── agents/             # LangGraph state machine (router.py, grader.py)
│   ├── retrieval.py        # Qdrant Hybrid Search logic
│   ├── reranker.py         # Cross-Encoder (ModernBERT) implementation
│   ├── cache.py            # RedisVL semantic cache logic
│   └── llm.py              
├── data/                   
│   ├── fetch_arxiv.py      
│   └── contextual_index.py # Contextual chunking and DB upsert logic
├── eval/                   
│   └── benchmark.py        # Latency & RAGAS tracking script
├── .env.example            
├── docker-compose.yml      # Multi-container orchestration (FastAPI, Qdrant, Redis)
└── README.md