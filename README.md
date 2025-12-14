# Retrieval-Augmented-Resume-Intelligence-Platform
---
- This project implements a resume intelligence system using Python, Streamlit, LangChain, ChromaDB, and Ollama to enable structured, retrieval-grounded question-answering over PDF resumes.
---

## Scope

**Solved**
- Grounded question answering over single PDF resumes
- Persistent semantic retrieval without re-embedding
- Explicit rejection of insufficient-context queries

**Not Solved**
- Cross-document reasoning
- Large-scale search
- Learned reranking or adaptive retrieval
---

## Stack

| Component | Choice |
|---------|-------|
| Language | Python |
| UI | Streamlit |
| RAG Framework | LangChain |
| Vector Store | ChromaDB |
| Embeddings | Ollama (`nomic-embed-text`) |
| LLM | Ollama (`llama3`) |
| PDF Parsing | PyMuPDF |

---

## Architecture
DF
→ Parse (PyMuPDF)
→ Chunk + Section Tagging
→ Embed (Ollama)
→ Persist (ChromaDB)
→ Retrieve (Similarity + Threshold)
→ Answer (Context-Constrained LLM)

---
---
## Core Behaviors

- **Persistent Indexing**  
  Embeddings are stored on disk; repeated queries do not trigger recomputation.

- **Document Isolation**  
  Each resume is hashed (SHA-256) to prevent duplicate ingestion and retrieval leakage.

- **Resume-Aware Chunking**  
  Chunks are tagged with semantic sections (experience, skills, education).

- **Thresholded Retrieval**  
  Queries below a similarity threshold are rejected instead of answered.

- **Constrained Generation**  
  LLM responses are limited strictly to retrieved context.

---
---
## Snippet
<img width="737" height="740" alt="image" src="https://github.com/user-attachments/assets/a8562111-1be6-4cbb-801c-b7d9a1f89f40" />
---
