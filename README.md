# Local RAG

A **local-first Retrieval-Augmented Generation system** for PDF documents.  
Ask questions about your document library and get cited answers — no cloud APIs required.

---

## Features

- **Structured PDF ingestion** via [Docling](https://github.com/DS4SD/docling) — text, tables, figures, equations
- **Hybrid retrieval** — dense vector search (HNSW/COSINE) fused with BM25 sparse search
- **LLM reranking** — Ollama reorders the top candidates before answering
- **Metadata filtering** — filter by year, chunk type, or paper at query time
- **Citation-grounded answers** — every claim is linked to `[paper_id, Section, Page]`
- **Fully local** — Milvus (Docker), sentence-transformers, Ollama; no external API calls

---

## Architecture

### Ingestion

```
PDF Folder
  → Docling parsing (title, authors, year, structure)
  → Structure-aware chunking (text / table / figure / equation)
  → sentence-transformers embeddings  (BAAI/bge-base-en-v1.5, 768-dim)
  → Milvus vector store  (HNSW index, COSINE metric)
  → BM25 index rebuild  (rank-bm25, saved to data/bm25_index.pkl)
```

### Query

```
Question
  → embed (dense vector)
  → HybridSearcher
      ├── Milvus ANN search  → TOP_K_VECTOR candidates + cosine scores
      └── BM25 search        → TOP_K_BM25  candidates + BM25 scores
          ↓  min-max normalise · fuse (0.6 × vec + 0.4 × BM25)
          → TOP_K_RERANK merged candidates
  → LLM Reranker  (single Ollama call)
  → TOP_K_FINAL chunks
  → ChatEngine  (Ollama, rolling history)
  → Answer with inline citations
```

---

## Tech Stack

| Concern | Library / Tool |
|---|---|
| PDF Parsing | [Docling](https://github.com/DS4SD/docling) |
| Vector DB | [Milvus standalone](https://milvus.io/) (Docker) |
| Embeddings | [sentence-transformers](https://www.sbert.net/) — `BAAI/bge-base-en-v1.5` |
| Sparse Retrieval | [rank-bm25](https://github.com/dorianbrown/rank_bm25) |
| LLM Runtime | [Ollama](https://ollama.ai/) — `llama3.1` default |
| Language | Python 3.11 |

---

## Quick Start

### Prerequisites

- Docker + Docker Compose
- Python 3.11
- [Ollama](https://ollama.ai/) with at least one model pulled

```bash
ollama pull llama3.1
```

### 1. Clone and install

```bash
git clone git@github.com:domingoran/academic_rag.git
cd academic_rag

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Milvus

```bash
docker compose up -d
```

> **WSL2 users:** volumes are stored at `~/milvus-volumes/` (WSL home).  
> Do **not** change this to `/mnt/c/` — MinIO requires fast local storage.

### 3. Add documents

**Option A — download from arXiv automatically:**

```bash
# 10 most recent q-bio.PE papers (default)
python download_papers.py

# Custom category / count
python download_papers.py --category cs.LG --n 20
```

**Option B — drop PDFs manually** into `data/papers/`.  
The filename stem becomes the `paper_id` (e.g. `attention_is_all_you_need.pdf`).

### 4. Ingest

```bash
python main.py --ingest
```

Already-ingested documents are skipped. The BM25 index is rebuilt automatically after ingestion.

### 5. Chat

```bash
python main.py
```

---

## Command-line flags

Run `python main.py [flag]`:

| Flag | Description |
|---|---|
| _(none)_ | Start the interactive chat |
| `--ingest` | Ingest all PDFs from `data/papers/` into the vector store |
| `--ingest-file <PDF>` | Ingest a single PDF file into the vector store |
| `--chat` | Start the interactive chat after ingestion (use with `--ingest`) |
| `--eval-generate` | Generate an evaluation Q&A dataset from the indexed chunks |
| `--manual-generator` | Run clustering + seed selection, then print each chunk group with the system prompt for manual Q&A generation via an external model |
| `--eval-score` | Score retrieval quality against `data/eval_dataset.json` (Recall@K, MRR) **and** generate LLM answers for comparison |
| `--eval-retrieve` | Score retrieval quality (Recall@K, MRR) without LLM answer generation |

---

## In-chat commands

| Command | Description |
|---|---|
| `/reset` | Clear conversation history |
| `/count` | Show number of indexed chunks |
| `/chunk <chunk_id>` | Show the full stored content + metadata of a single chunk |
| `/bm25` | Rebuild BM25 index from current Milvus data |
| `/filter <expr>` | Set a Milvus scalar filter (shown in prompt while active) |
| `/filter clear` | Remove the active filter |
| `/delete <paper_id>` | Delete all chunks for one paper, then rebuild BM25 (**asks for confirmation**) |
| `/delete` | Wipe the entire vector store and BM25 index (**asks for confirmation**) |
| `/quit` | Exit (also: `exit`, `q`) |

### Filter examples

```
/filter year >= 2022
/filter chunk_type == "table"
/filter paper_id == "2312.00752"
/filter year >= 2020 && chunk_type == "text"
```

---

## Configuration

All tuneable parameters live in `config.py` — never hard-code them in module files.

```python
# Retrieval
TOP_K_VECTOR         = 10     # dense candidates from Milvus
TOP_K_BM25           = 10     # sparse candidates from BM25
TOP_K_RERANK         = 20     # merged candidates passed to the reranker
TOP_K_FINAL          = 5      # chunks shown to the LLM

# Hybrid fusion weights (must sum to 1.0)
HYBRID_VECTOR_WEIGHT = 0.6
HYBRID_BM25_WEIGHT   = 0.4

# Set to False to skip LLM reranking (faster, lower quality)
RERANKER_ENABLED     = True

# Swap embedding model here (update EMBEDDING_DIM accordingly)
EMBEDDING_MODEL      = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM        = 768

# Swap LLM here (any model pulled in Ollama)
OLLAMA_CHAT_MODEL    = "llama3.1"
```

### A/B testing

| Goal | Change in `config.py` |
|---|---|
| Reproduce Phase 1 (pure vector) | `HYBRID_VECTOR_WEIGHT=1.0, HYBRID_BM25_WEIGHT=0.0` |
| Pure BM25 baseline | `HYBRID_VECTOR_WEIGHT=0.0, HYBRID_BM25_WEIGHT=1.0` |
| Disable reranker | `RERANKER_ENABLED=False` |

---

## Project Structure

```
academic_rag/
│
├── data/
│   ├── papers/              ← place PDF files here (gitignored)
│   ├── metadata.json        ← arXiv metadata (download_papers.py output)
│   └── bm25_index.pkl       ← BM25 index (auto-generated, gitignored)
│
├── ingestion/
│   ├── docling_parser.py    ← PDF → ParsedDocument (title, year, authors)
│   ├── chunker.py           ← structure-aware chunking
│   └── metadata_builder.py  ← UUID assignment, empty-chunk cleanup
│
├── retrieval/
│   ├── vector_store.py      ← Milvus MilvusClient wrapper
│   ├── bm25_index.py        ← BM25Okapi + disk persistence
│   ├── hybrid_search.py     ← score fusion (dense + sparse)
│   └── reranker.py          ← LLM reranker (Ollama)
│
├── embeddings/
│   └── embedder.py          ← sentence-transformers wrapper
│
├── llm/
│   ├── ollama_client.py     ← Ollama SDK wrapper
│   └── chat_engine.py       ← prompt builder, rolling history, citations
│
├── core/
│   ├── schemas.py           ← Pydantic models (RawElement, Chunk, …)
│   └── pipeline.py          ← RAGPipeline orchestrator (lazy init)
│
├── config.py                ← all tuneable settings
├── main.py                  ← CLI entry point
├── download_papers.py       ← arXiv batch downloader
├── docker-compose.yml       ← Milvus standalone stack
└── requirements.txt
```

---

## Roadmap

### ✅ Phase 1 — Core pipeline
PDF ingestion · structure-aware chunking · embeddings · Milvus · vector search · CLI chat · arXiv downloader

### ✅ Phase 2 — Hybrid retrieval
BM25 index · weighted score fusion · LLM reranker · metadata filtering · author extraction · `/delete` reset command

### 🔲 Phase 3 — Quality & scale
Improved figure/table/equation handling · citation quality · chunk size experiments · quantitative eval (Recall@K, MRR) · FastAPI layer

---

## Known Issues

- **WSL2 / C: drive space** — if Milvus crashes with *"minimum free drive threshold"*, volumes landed on `/mnt/c/`. Fix: `docker compose down && rm -rf ./volumes && docker compose up -d`.
- **Author extraction** — heuristic-based; works well for standard arXiv PDFs, less reliable for scanned or unconventional layouts.
- **Reranker with small models** — models ≤ 7B may produce malformed rankings; the parser falls back to the original order so retrieval never silently fails.
- **First run** — Docling downloads its layout/OCR models on first use (~few minutes, then cached). The BM25 index is also built from Milvus on startup if the `.pkl` file is absent.

---

## License

MIT
