## 1. Project Overview

This project is a **local-first Retrieval-Augmented Generation (RAG) system** for academic PDF papers.

It enables:
- Ingestion of academic papers from a local folder
- Structured parsing of PDFs (text, tables, figures, equations, authors)
- Hybrid retrieval (BM25 + vector search with weighted score fusion)
- LLM-based reranking via Ollama
- Metadata filtering (year, chunk_type, paper_id) at query time
- Chat-based interaction with citations
- Modular LLM backend (Ollama-based models)

The system is designed to start as a local research tool and later evolve into an API-based service.

---

## 2. High-Level Architecture

### Ingestion Pipeline

```
PDF Folder
↓
Docling Parsing  (ingestion/docling_parser.py)   ← title, year, authors (Phase 2)
↓
Structured Document Representation  (core/schemas.py → ParsedDocument)
↓
Structure-aware + token-budget Chunking  (ingestion/chunker.py)
↓
Metadata Enrichment  (ingestion/metadata_builder.py)
↓
Embedding Generation — sentence-transformers  (embeddings/embedder.py)
↓
Milvus Vector Store + flush  (retrieval/vector_store.py)
↓
BM25 Index rebuild + save to disk  (retrieval/bm25_index.py)
```

### Query Pipeline (Phase 2)

```
User question
↓
Embedder.embed_query()
↓
HybridSearcher  (retrieval/hybrid_search.py)
  ├── VectorStore.search_with_scores()   → TOP_K_VECTOR candidates + cosine scores
  └── BM25Index.query()                 → TOP_K_BM25  candidates + BM25 scores
      ↓  min-max normalise each list
      ↓  fuse: 0.6 × vec_norm + 0.4 × bm25_norm
      → TOP_K_RERANK merged candidates
↓
Reranker  (retrieval/reranker.py)        → single Ollama generate() call
↓
TOP_K_FINAL chunks
↓
ChatEngine  (llm/chat_engine.py)         → answer with inline citations
```

---

## 3. Tech Stack

### Core Components

| Concern | Library / Tool | Notes |
|---|---|---|
| PDF Parsing | **Docling** | `DocumentConverter`, iterates `DocItemLabel` items |
| Chunking | **custom** | structure-aware + word-count token budget; no LangChain splitter |
| Vector DB | **Milvus** standalone | Docker Compose; `MilvusClient` API (pymilvus ≥ 2.4) |
| Sparse Retrieval | **rank-bm25** | `BM25Okapi`; persisted to `data/bm25_index.pkl` |
| Embeddings | **sentence-transformers** | `BAAI/bge-base-en-v1.5` default; configurable |
| LLM Runtime | **Ollama** | `llama3.1` default; any pulled model works |
| Language | Python 3.11 | WSL2, `.venv` managed by uv |

---

## 4. Data Model

Each paper is decomposed into **structured chunks** (`core/schemas.py`).

### Schemas

```python
# Raw structural element from Docling — input to the chunker
class RawElement(BaseModel):
    label: str          # DocItemLabel value: 'text', 'section_header', 'table', …
    text: str | None
    markdown: str | None    # table markdown rendering
    caption: str | None     # figure / table caption
    page: int = 0
    level: int = 0

# Document-level metadata + ordered elements — output of the parser
class ParsedDocument(BaseModel):
    paper_id: str           # PDF filename stem (e.g. "2605.22665")
    title: str
    authors: list[str]      # best-effort heuristic extraction (Phase 2)
    year: int
    elements: list[RawElement]

# Per-chunk provenance stored in Milvus scalar fields
class ChunkMetadata(BaseModel):
    page: int = 0
    figure_id: str | None = None
    table_id: str | None = None
    equation_id: str | None = None

# The core unit of retrieval — one Milvus entity
class Chunk(BaseModel):
    chunk_id: str           # UUID
    paper_id: str
    title: str
    authors: list[str]
    year: int
    section: str
    chunk_type: str         # text | table | figure | equation
    content: str
    metadata: ChunkMetadata
    embedding: list[float] | None   # populated by embedder
```

---

## 5. PDF Ingestion Pipeline

### Step 1 — PDF Loading

* Load PDFs from `data/papers/`
* `paper_id` = filename stem (e.g. `2605.22665.pdf` → `paper_id = "2605.22665"`)

### Step 2 — Docling Parsing (`ingestion/docling_parser.py`)

`parse_pdf(pdf_path) → ParsedDocument`

* Converts PDF with `DocumentConverter`
* Iterates `doc.iterate_items()` — handles `TextItem`, `TableItem`, `PictureItem`
* Extracts title from first `DocItemLabel.TITLE` item
* Infers year with regex from filename then first 2000 chars of text
* Tables exported via `item.export_to_markdown(doc)`
* Figures use `item.caption_text(doc)` (caption-only; no OCR)
* **Author extraction (Phase 2):** scans TEXT elements between the TITLE and the first SECTION_HEADER on page 1/2; a block is accepted as author-like if it has ≥ 2 capitalised words, no sentence-ending punctuation, no affiliation keywords (university, department, @, …). Best-effort — works well for standard arXiv PDFs.

### Step 3 — Chunking (`ingestion/chunker.py`)

`chunk_document(parsed_doc) → list[Chunk]`

* **Section headers** → flush text buffer, update current section label
* **Text / list_item / paragraph** → accumulate in buffer
* **Table** → flush buffer, then one standalone chunk (markdown + caption)
* **Figure** → one chunk if caption exists; else skipped
* **Formula** → one standalone equation chunk
* Long text buffers are word-split with overlap (`CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP_TOKENS`)
* Max content stored in Milvus VARCHAR: **8 000 chars**

### Step 4 — Metadata Enrichment (`ingestion/metadata_builder.py`)

* Assigns UUIDs to any chunk missing a `chunk_id`
* Drops empty / whitespace-only chunks

---

## 6. Embedding Layer (`embeddings/embedder.py`)

### Model

Default: `BAAI/bge-base-en-v1.5` (768-dim, instruction-tuned for retrieval)

Configurable via `config.py`:
```python
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM   = 768
EMBEDDING_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
```

Set `EMBEDDING_QUERY_INSTRUCTION = ""` when switching to a non-BGE model.

### Rules

* Document passages are encoded **as-is** (no instruction prefix)
* Queries are encoded with the instruction prefix prepended
* All vectors are L2-normalised (`normalize_embeddings=True`)
* Only `content` field is embedded; metadata is not

---

## 7. Vector Store — Milvus (`retrieval/vector_store.py`)

### API

Uses **`MilvusClient`** (pymilvus ≥ 2.4 new API — not the deprecated ORM `Collection`/`connections`).

### Collection: `academic_chunks`

| Field | Type | Length |
|---|---|---|
| `chunk_id` | VARCHAR (PK) | 64 |
| `paper_id` | VARCHAR | 256 |
| `title` | VARCHAR | 512 |
| `authors` | VARCHAR | 512 |
| `year` | INT64 | — |
| `section` | VARCHAR | 512 |
| `chunk_type` | VARCHAR | 32 |
| `content` | VARCHAR | 8192 |
| `page` | INT64 | — |
| `embedding` | FLOAT_VECTOR | 768 |

### Index

HNSW, metric COSINE, M=16, efConstruction=128.

### Key methods

| Method | Description |
|---|---|
| `insert(chunks)` | Insert + **flush** immediately (prevents count() returning 0) |
| `has_paper(paper_id)` | Skip re-ingestion check |
| `search(query_vec, top_k, expr)` | Returns `List[Chunk]` |
| `search_with_scores(query_vec, top_k, expr)` | Returns `List[(Chunk, float)]` with cosine scores — used by HybridSearcher |
| `fetch_all_content()` | Returns all `(chunk_id, content)` pairs, paginated — used to rebuild BM25 |
| `fetch_by_ids(ids)` | Materialise BM25 hits not in vector results |
| `count()` | Uses `query(count(*))` — accurate even before auto-flush |
| `reset()` | Drop + recreate collection (empty) |

### ⚠️ Flush note

`insert()` calls `flush()` immediately after every write. Without this, `count()` and `get_collection_stats()` return 0 because Milvus only counts sealed (flushed) segments. Do **not** remove the flush call.

---

## 8. BM25 Index — Sparse Retrieval (`retrieval/bm25_index.py`)

`BM25Index` wraps `rank_bm25.BM25Okapi` with chunk-id mapping and disk persistence.

### Lifecycle

* **Built** from `(chunk_id, content)` pairs via `build(items)`
* **Saved** to `data/bm25_index.pkl` (pickle) via `save()`
* **Auto-loaded** on pipeline startup; rebuilt from Milvus if file is missing
* **Rebuilt** automatically after every `ingest_folder()` run
* **Manually rebuilt** via the `/bm25` CLI command

### Tokenisation

Simple whitespace split (lowercased). Good enough for English academic text. Can be replaced with a proper tokeniser in Phase 3.

### Query

`query(text, top_k) → List[(chunk_id, bm25_score)]` — returns only chunks with score > 0.

---

## 9. Hybrid Retrieval (`retrieval/hybrid_search.py`)

`HybridSearcher` fuses dense and sparse results via **min-max normalised weighted sum**.

### Score fusion

1. Fetch `TOP_K_VECTOR` results from Milvus with cosine scores.
2. Fetch `TOP_K_BM25` results from BM25 with raw scores.
3. Each list is min-max normalised independently to [0, 1].
4. Chunks missing from one list receive 0 in that component.
5. `fused = HYBRID_VECTOR_WEIGHT × vec_norm + HYBRID_BM25_WEIGHT × bm25_norm`
6. Sort descending, return top `TOP_K_RERANK`.

### Fallback

If the BM25 index is not built yet, `HybridSearcher` falls back to pure vector search transparently.

### Metadata filtering

Pass a Milvus boolean expression as `expr` to pre-filter candidates:
```python
searcher.search(vec, text, expr='year >= 2022 && chunk_type == "text"')
```

---

## 10. Reranking Layer (`retrieval/reranker.py`)

`Reranker` uses a single `OllamaClient.generate()` call to reorder up to `TOP_K_RERANK` candidates.

### Prompt strategy

Sends the query + truncated passage previews (400 chars each) and asks the LLM for a comma-separated ranking of passage numbers (most relevant first).

### Robustness

`_parse_ranking()` handles:
* Perfect LLM responses → exact permutation
* Partial responses → valid indices placed first, missing ones appended in original order
* Gibberish / no numbers → original order preserved

### Toggle

Set `RERANKER_ENABLED = False` in `config.py` to skip reranking (faster, lower quality).

---

## 11. Chat Generation Layer (`llm/`)

### `llm/ollama_client.py` — `OllamaClient`

* Wraps the Ollama Python SDK
* `chat(messages)` — multi-turn with history
* `generate(prompt)` — one-shot (used by Reranker)
* `is_available()` — checks Ollama server + model presence

### `llm/chat_engine.py` — `ChatEngine`

* Keeps a rolling 6-turn history (bare queries only, not full context)
* `answer(query, chunks)` — injects retrieved chunks as numbered context block
* Citation format enforced in system prompt: `[paper_id, Section: …, Page: …]`
* LLM is instructed to answer **only from provided context**

---

## 12. Pipeline Orchestrator (`core/pipeline.py`)

`RAGPipeline` — owns all components with lazy init.

### Ingestion
```python
pipeline.ingest_pdf(pdf_path)    # single file
pipeline.ingest_folder()          # all PDFs in data/papers/ → auto-rebuilds BM25
```

### Query
```python
answer, chunks = pipeline.query("What is self-attention?")
answer, chunks = pipeline.query("results table", expr='chunk_type == "table"')
```

### Management
```python
pipeline.rebuild_bm25()   # force BM25 rebuild from current Milvus data
pipeline.full_reset()     # wipe Milvus collection + delete BM25 file + reset state
```

### Component init order (lazy)

`embedder` → `vector_store` → `bm25_index` (load or build) → `hybrid_searcher` → `ollama_client` → `reranker` → `chat_engine`

---

## 13. Citation System

Each chunk retains `paper_id`, `section`, `page` (scalar Milvus fields).

LLM is instructed via system prompt:
```
[paper_id, Section: <section>, Page: <page>]
```

---

## 14. Project Structure

```
project/
│
├── data/
│   ├── papers/          ← PDF files (one per paper; filename = paper_id)
│   ├── metadata.json    ← arXiv metadata (populated by download_papers.py)
│   └── bm25_index.pkl   ← BM25 index (auto-generated; gitignored)
│
├── ingestion/
│   ├── docling_parser.py    ✅ Phase 1+2 (+ author extraction)
│   ├── chunker.py           ✅ Phase 1
│   └── metadata_builder.py  ✅ Phase 1
│
├── retrieval/
│   ├── vector_store.py      ✅ Phase 1+2 (+ scores, fetch_all, fetch_by_ids, reset)
│   ├── bm25_index.py        ✅ Phase 2
│   ├── hybrid_search.py     ✅ Phase 2
│   └── reranker.py          ✅ Phase 2
│
├── embeddings/
│   └── embedder.py          ✅ Phase 1
│
├── llm/
│   ├── ollama_client.py     ✅ Phase 1
│   └── chat_engine.py       ✅ Phase 1
│
├── core/
│   ├── schemas.py           ✅ Phase 1
│   └── pipeline.py          ✅ Phase 1+2
│
├── config.py                ✅ Phase 1+2 — all tuneable settings here
├── main.py                  ✅ Phase 1+2 — CLI entry point
├── download_papers.py       ✅ arXiv downloader utility
└── docker-compose.yml       ✅ Milvus standalone stack
```

---

## 15. Milvus Docker Compose

Milvus standalone stack: **etcd + minio + milvus-standalone**.

### ⚠️ WSL2 Volume Location

Volumes are stored at **`~/milvus-volumes/`** (WSL home directory — `/home/<user>/`).  
**Do NOT change this to a path under `/mnt/c/`** — the Windows C: drive does not have enough free space, and MinIO will panic with a "minimum free drive threshold" error.

### Commands

```bash
# Start
docker compose up -d

# Stop (keeps volumes)
docker compose down

# Full reset (deletes all indexed data)
docker compose down && rm -rf ~/milvus-volumes
```

---

## 16. MVP Roadmap

### Phase 1 — ✅ COMPLETE

* [x] PDF ingestion (Docling)
* [x] Structure-aware chunking
* [x] Embeddings (sentence-transformers, bge-base-en-v1.5)
* [x] Milvus storage (MilvusClient)
* [x] Basic vector search (COSINE/HNSW)
* [x] Interactive CLI chat with citations
* [x] arXiv paper downloader (`download_papers.py`)

### Phase 2 — ✅ COMPLETE

* [x] BM25 index (`retrieval/bm25_index.py`) — rank-bm25, disk persistence
* [x] Hybrid retrieval with weighted score fusion (`retrieval/hybrid_search.py`)
* [x] Metadata filtering (year, chunk_type, paper_id) via `/filter` CLI command
* [x] LLM-based reranker (`retrieval/reranker.py`) — single Ollama call, robust parser
* [x] Author extraction from parsed PDFs (heuristic, best-effort)
* [x] `search_with_scores()`, `fetch_all_content()`, `fetch_by_ids()` in VectorStore
* [x] Milvus flush-on-insert + `count(*)` query fix
* [x] `/delete` command to wipe and restart from scratch
* [x] `/bm25` command to manually rebuild the sparse index

### Phase 3

* [ ] Improved figure/table/equation handling
* [ ] Citation quality improvements
* [ ] Chunk size / overlap optimisation experiments
* [ ] Quantitative retrieval eval (Recall@K, MRR) with a ground-truth query set
* [ ] API layer (FastAPI)

---

## 17. Key Design Principles

* local-first architecture (no cloud APIs required)
* reproducible pipelines
* modular retrieval components (Phase 2 components are drop-in replaceable)
* experiment-friendly LLM switching (change `OLLAMA_CHAT_MODEL` in `config.py`)
* strong metadata tracking (every chunk knows its source)
* config-driven A/B testing: toggle hybrid weights and reranker in `config.py`

---

## 18. Configuration (`config.py`)

All tuneable parameters live in `config.py`. **Never hard-code these in module files.**

```python
# Paths
DATA_DIR        = BASE_DIR / "data" / "papers"
BM25_INDEX_PATH = BASE_DIR / "data" / "bm25_index.pkl"

# Embedding — swap model here, update EMBEDDING_DIM accordingly
EMBEDDING_MODEL             = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM               = 768
EMBEDDING_BATCH_SIZE        = 32
EMBEDDING_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Milvus
MILVUS_HOST       = "localhost"
MILVUS_PORT       = 19530
MILVUS_COLLECTION = "academic_chunks"

# Ollama
OLLAMA_BASE_URL   = "http://localhost:11434"
OLLAMA_CHAT_MODEL = "llama3.1"

# Chunking
CHUNK_MAX_TOKENS     = 512
CHUNK_OVERLAP_TOKENS = 50

# Retrieval — candidate counts
TOP_K_VECTOR = 10    # dense candidates from Milvus
TOP_K_BM25   = 10    # sparse candidates from BM25
TOP_K_RERANK = 20    # merged candidates passed to reranker
TOP_K_FINAL  = 5     # top chunks shown to the LLM

# Hybrid fusion weights (must sum to 1.0)
HYBRID_VECTOR_WEIGHT = 0.6
HYBRID_BM25_WEIGHT   = 0.4

# Set to False to skip LLM reranking (faster, lower quality)
RERANKER_ENABLED = True
```

### A/B testing via config

| Goal | Change |
|---|---|
| Pure vector (Phase 1 behaviour) | `HYBRID_VECTOR_WEIGHT=1.0, HYBRID_BM25_WEIGHT=0.0` |
| Pure BM25 | `HYBRID_VECTOR_WEIGHT=0.0, HYBRID_BM25_WEIGHT=1.0` |
| Disable reranker | `RERANKER_ENABLED=False` |
| More LLM context | Increase `TOP_K_FINAL` |

---

## 19. Dependency Management (uv)

This project uses **uv** as the package manager.

### Current `requirements.txt`

```
pymilvus
rank-bm25
ollama
pydantic
numpy
tqdm
docling
sentence-transformers
```

### Activate environment

```bash
source .venv/bin/activate   # WSL / Linux
```

### Install / update deps

```bash
pip install -r requirements.txt
```

---

## 20. CLI Usage

### Step 1 — Start Milvus

```bash
docker compose up -d
```

### Step 2 — (Optional) Download papers from arXiv

```bash
# 10 most recent q-bio.PE papers (default)
python download_papers.py

# Custom category / count
python download_papers.py --category cs.LG --n 20
```

Output: `data/papers/*.pdf` + `data/metadata.json`

### Step 3 — Ingest papers

```bash
python main.py --ingest
```

* Already-ingested papers are skipped automatically.
* The BM25 index is rebuilt and saved automatically at the end of ingestion.

### Step 4 — Chat

```bash
python main.py
```

In-chat commands:

| Command | Action |
|---|---|
| `/reset` | Clear conversation history |
| `/count` | Show number of indexed chunks |
| `/bm25` | Rebuild BM25 index from current Milvus data |
| `/filter <expr>` | Set a Milvus scalar filter (shown in prompt while active) |
| `/filter clear` | Remove the active filter |
| `/delete` | Wipe the vector store + BM25 index (asks for confirmation) |
| `/quit` | Exit (also: `exit`, `q`) |

### Filter expression examples

```
/filter year >= 2022
/filter chunk_type == "table"
/filter paper_id == "2605.22665"
/filter year >= 2020 && chunk_type == "text"
```

### Ingest + chat in one shot

```bash
python main.py --ingest --chat
```

---

## 21. Git Repository

**Remote:** `git@github.com:domingoran/academic_rag.git`  
**Branches:** `main` (stable releases) · `dev` (active development)

```bash
# Day-to-day work happens on dev
git checkout dev
git add -A && git commit -m "your message" && git push

# Merge dev → main when a phase is complete
git checkout main && git merge dev && git push
git checkout dev   # switch back to keep working

# Clone fresh (gets main by default; switch to dev to develop)
git clone git@github.com:domingoran/academic_rag.git
git checkout dev
```

**Not tracked** (see `.gitignore`): `.venv/`, `data/papers/*`, `data/bm25_index.pkl`, `data/metadata.json`, `__pycache__/`.

---

## 22. Known Issues / Notes

- **C: drive space**: Docker volumes must stay on the WSL filesystem (`~/milvus-volumes/`). If Milvus crashes with `Storage backend has reached its minimum free drive threshold`, the volumes ended up on `/mnt/c/`. Fix: `docker compose down && rm -rf ./volumes && docker compose up -d` (the compose file already points to `~/milvus-volumes/`).
- **pymilvus deprecation warnings**: The ORM-style `Collection`/`connections` API is deprecated in pymilvus 3.x. The codebase already uses `MilvusClient` — ignore any warnings from third-party libs.
- **Author extraction**: Best-effort heuristic — works well for standard arXiv PDFs (names on page 1 between title and Abstract). May miss or over-include text for non-standard layouts. Already-ingested chunks with empty authors need a `/delete` + re-ingest to pick up the new extraction.
- **BM25 index cold start**: On first startup after the Phase 2 update (or if `data/bm25_index.pkl` is missing), the pipeline fetches all content from Milvus and builds the index automatically. This adds a few seconds to startup but is silent.
- **Reranker with small models**: Small Ollama models (≤ 7B) may produce malformed rankings. The fallback always returns a valid order so retrieval never silently fails — but quality may vary.
- **Docling model download**: First run of `parse_pdf()` downloads Docling's internal ML models (layout detection, OCR). This takes a few minutes on first use and is cached afterwards.
