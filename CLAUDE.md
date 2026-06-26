## 1. Project Overview

This project is a **local-first Retrieval-Augmented Generation (RAG) system** for academic PDF papers.

It enables:
- Ingestion of academic papers from a local folder
- Structured parsing of PDFs (text, tables, figures, equations, authors)
- Hybrid retrieval (BM25 + vector search with weighted score fusion)
- Cross-encoder reranking via BAAI/bge-reranker-v2-m3
- Metadata filtering (year, chunk_type, paper_id) at query time
- Chat-based interaction with citations
- Modular LLM backend (Ollama-based models)
- Automated RAG evaluation dataset generation via embedding clustering and LLM-based Q&A synthesis

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

### Evaluation Dataset Generation Pipeline (Phase 4a)

```
All embeddings fetched from Milvus  (vector_store.fetch_all_embeddings())
↓
K-Means clustering — K chosen via silhouette-score sweep over [EVAL_CLUSTER_K_MIN, EVAL_CLUSTER_K_MAX]
↓
Stratified seed sampling — proportional to each paper_id's chunk count; one seed per cluster slot
↓
Nearest-neighbour group assembly — EVAL_CHUNKS_PER_GROUP-1 neighbours per seed (cosine, within cluster, in-memory numpy)
↓
Chunk content fetched from Milvus  (vector_store.fetch_by_ids())
↓
OllamaClient.generate() — EVAL_QUESTIONS_PER_GROUP Q&A pairs per group (easy / medium / hard)
                           LLM also returns relevant_chunk_ids for each Q&A pair
↓
Accumulate until EVAL_NUM_QUESTIONS total; write data/eval_dataset.json
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
Reranker  (retrieval/reranker.py)        → cross-encoder scoring (bge-reranker-v2-m3)
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
| Clustering | **scikit-learn** | `KMeans` + `silhouette_score` for eval dataset generation |
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
* **Table** → flush buffer → standalone chunk with smart context stitching (Phase 3)
* **Figure** → standalone chunk; no-caption figures kept when context exists (Phase 3)
* **Formula** → standalone equation chunk with context stitching (Phase 3)
* Long text buffers are word-split with overlap (`CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP_TOKENS`)
* Max content stored in Milvus VARCHAR: **8 000 chars**

#### Phase 3 context stitching

Tables, figures, and equations alone embed poorly — their meaning lives in the surrounding prose. The chunker now stitches `_CONTEXT_WORDS = 50` words of immediately preceding and following text into every non-text chunk:

```text
[...attention is computed as a scaled dot-product of queries and keys...]

Table: Comparison of model variants on WMT14 EN-DE

| Model         | BLEU |
|---------------|------|
| Base          | 27.3 |
| + larger FFN  | 28.4 |
[... 3 rows truncated]

[...bold entries indicate statistical significance (p < 0.05)...]
```

* **Preceding context** — last 50 words from the accumulated text buffer (taken before flush)
* **Following context** — next 50 words scanned ahead in the element list (stops at section headers)
* **Table truncation** — Markdown header row is always kept; data rows are trimmed to fit `_MAX_TABLE_MARKDOWN_CHARS = 6 000` with a `[... N rows truncated]` note
* **No-caption figures** — a `"Figure (no caption)"` chunk is still created if pre/following context exists; only truly isolated caption-less figures are dropped
* **Equation buffer** — the text buffer is **not** flushed for equations (they are inline); surrounding text flow continues uninterrupted

#### Sequential element IDs (Phase 3)

`ChunkMetadata.table_id`, `figure_id`, and `equation_id` are now populated with sequential per-document IDs (`tbl-1`, `fig-2`, `eq-3`, …). These enable targeted metadata queries once a `/filter` expression based on these fields is added.

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
| `fetch_all_embeddings()` | Returns all `{"chunk_id", "paper_id", "embedding"}` dicts, paginated — used by EvalDatasetGenerator |
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

`Reranker` uses `sentence_transformers.CrossEncoder` with `BAAI/bge-reranker-v2-m3` to score each (query, passage) pair and return candidates sorted by descending relevance score.

### Scoring strategy

Calls `model.predict([(query, passage), ...])` in a single batch — no LLM call, no prompt parsing.
Model weights are downloaded from HuggingFace on first use (~568 MB) and cached automatically.

### Robustness

Falls back to the original order on any exception so retrieval never silently degrades.

### Toggle

Set `RERANKER_ENABLED = False` in `config.py` to skip reranking (faster, lower quality).

---

## 11. Chat Generation Layer (`llm/`)

### `llm/ollama_client.py` — `OllamaClient`

* Wraps the Ollama Python SDK
* `chat(messages)` — multi-turn with history
* `generate(prompt)` — one-shot
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
chunks = pipeline.retrieve("What is self-attention?", top_k=5)  # retrieval only, no LLM call
```

### Management
```python
pipeline.rebuild_bm25()   # force BM25 rebuild from current Milvus data
pipeline.full_reset()     # wipe Milvus collection + delete BM25 file + reset state
```

### Component init order (lazy)

`embedder` → `vector_store` → `bm25_index` (load or build) → `hybrid_searcher` → `reranker` → `ollama_client` → `chat_engine`

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
│   ├── papers/             ← PDF files (one per paper; filename = paper_id)
│   ├── metadata.json       ← arXiv metadata (populated by download_papers.py)
│   ├── bm25_index.pkl      ← BM25 index (auto-generated; gitignored)
│   ├── eval_dataset.json   ← Q&A evaluation dataset (auto-generated; gitignored)
│   └── eval_results.json   ← retrieval scoring results (auto-generated; gitignored)
│
├── ingestion/
│   ├── docling_parser.py    ✅ Phase 1+2
│   ├── chunker.py           ✅ Phase 1+3
│   └── metadata_builder.py  ✅ Phase 1
│
├── retrieval/
│   ├── vector_store.py      ✅ Phase 1+2+4a
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
│   └── pipeline.py          ✅ Phase 1+2+4b (+ retrieve())
│
├── evaluation/
│   ├── eval_dataset_generator.py  ✅ Phase 4a — clustering + Q&A generation
│   └── eval_scorer.py             ✅ Phase 4b — Recall@K + MRR scoring
│
├── notebooks/               ← Jupyter notebooks (gitignored)
│   ├── clustering_analysis.ipynb    — interactive cluster/t-SNE exploration
│   └── llm_pair_generation.ipynb   — interactive Q&A generation with Ollama
│
├── config.py                ✅ Phase 1+2+4a+4b — all tuneable settings here
├── main.py                  ✅ Phase 1+2+4a+4b — CLI entry point
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
* [x] Reranker (`retrieval/reranker.py`) — originally Ollama LLM; replaced in Phase 3 with `BAAI/bge-reranker-v2-m3` cross-encoder
* [x] Author extraction from parsed PDFs (heuristic, best-effort)
* [x] `search_with_scores()`, `fetch_all_content()`, `fetch_by_ids()` in VectorStore
* [x] Milvus flush-on-insert + `count(*)` query fix
* [x] `/delete` command to wipe and restart from scratch
* [x] `/bm25` command to manually rebuild the sparse index

### Phase 3 — 🔲 IN PROGRESS

* [x] Improved figure/table/equation handling — context stitching, no-caption figures, smart table truncation, element IDs (`chunker.py`)
* [ ] Citation quality improvements
* [ ] Chunk size / overlap optimisation experiments
* [ ] API layer (FastAPI)

### Phase 4a — ✅ COMPLETE: Evaluation Dataset Generation

* [x] `VectorStore.fetch_all_embeddings()` — paginated fetch of all `(chunk_id, paper_id, embedding)` records
* [x] K-Means clustering with silhouette-score sweep to choose K (`sklearn.cluster.KMeans`)
* [x] Stratified seed sampling proportional to each paper's chunk share
* [x] Nearest-neighbour group assembly within each cluster (in-memory cosine, numpy)
* [x] LLM Q&A generation — EVAL_QUESTIONS_PER_GROUP pairs per group, easy/medium/hard, with `relevant_chunk_ids`
* [x] JSON output to `data/eval_dataset.json`
* [x] `--eval-generate` CLI flag in `main.py`
* [x] All hyperparameters in `config.py`
* [x] `scikit-learn` and `plotly` added to `requirements.txt`
* [x] Interactive notebooks in `notebooks/` for clustering exploration and Q&A generation

### Phase 4b — ✅ COMPLETE: Retrieval Evaluation Scoring

* [x] Load `data/eval_dataset.json`
* [x] `pipeline.retrieve(question, top_k)` — retrieval-only path in `core/pipeline.py` (skips LLM generation)
* [x] Compare retrieved chunk_ids against `relevant_chunk_ids` from the dataset
* [x] Compute Recall@K (K=1, 3, 5) and MRR per question; break down by difficulty
* [x] Output `data/eval_results.json` with per-question and aggregate scores
* [x] `--eval-score` CLI flag in `main.py`

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

# Cross-encoder reranker model
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Set to False to skip reranking (faster, lower quality)
RERANKER_ENABLED = True

# Evaluation — dataset generation (Phase 4a) and scoring (Phase 4b)
EVAL_DATASET_PATH        = BASE_DIR / "data" / "eval_dataset.json"
EVAL_RESULTS_PATH        = BASE_DIR / "data" / "eval_results.json"
EVAL_NUM_QUESTIONS       = 50    # total Q&A pairs to generate
EVAL_CLUSTER_K_MIN       = 5     # min K for silhouette-score sweep
EVAL_CLUSTER_K_MAX       = 30    # max K for silhouette-score sweep
EVAL_CHUNKS_PER_GROUP    = 4     # chunks per LLM call (1 seed + 3 neighbours)
EVAL_QUESTIONS_PER_GROUP = 3     # Q&A pairs generated per chunk group
EVAL_LLM_MODEL           = OLLAMA_CHAT_MODEL   # inherits chat model; override here if needed
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
scikit-learn
plotly
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

### Step 5 — Generate evaluation dataset (Phase 4a)

```bash
python main.py --eval-generate
```

* Requires at least one paper already ingested in Milvus.
* Clusters all embeddings, samples stratified seed chunks, calls the LLM to generate Q&A pairs.
* Writes `data/eval_dataset.json`. Controlled by `EVAL_NUM_QUESTIONS` in `config.py`.
* Alternatively run interactively via `notebooks/clustering_analysis.ipynb` + `notebooks/llm_pair_generation.ipynb`.

### Step 6 — Score retrieval quality (Phase 4b)

```bash
python main.py --eval-score
```

* Requires `data/eval_dataset.json` (produced by Step 5).
* Runs `pipeline.retrieve()` on every question, computes Recall@1/3/5 and MRR.
* Prints a summary table and writes `data/eval_results.json`.

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

**Not tracked** (see `.gitignore`): `.venv/`, `data/papers/*`, `data/bm25_index.pkl`, `data/metadata.json`, `data/eval_dataset.json`, `data/eval_results.json`, `notebooks/`, `__pycache__/`.

---

## 22. Known Issues / Notes

- **C: drive space**: Docker volumes must stay on the WSL filesystem (`~/milvus-volumes/`). If Milvus crashes with `Storage backend has reached its minimum free drive threshold`, the volumes ended up on `/mnt/c/`. Fix: `docker compose down && rm -rf ./volumes && docker compose up -d` (the compose file already points to `~/milvus-volumes/`).
- **pymilvus deprecation warnings**: The ORM-style `Collection`/`connections` API is deprecated in pymilvus 3.x. The codebase already uses `MilvusClient` — ignore any warnings from third-party libs.
- **Author extraction**: Best-effort heuristic — works well for standard arXiv PDFs (names on page 1 between title and Abstract). May miss or over-include text for non-standard layouts. Already-ingested chunks with empty authors need a `/delete` + re-ingest to pick up the new extraction.
- **BM25 index cold start**: On first startup after the Phase 2 update (or if `data/bm25_index.pkl` is missing), the pipeline fetches all content from Milvus and builds the index automatically. This adds a few seconds to startup but is silent.
- **Reranker model download**: First run loads `BAAI/bge-reranker-v2-m3` from HuggingFace (~568 MB). Download is automatic and cached; subsequent startups are instant.
- **Docling model download**: First run of `parse_pdf()` downloads Docling's internal ML models (layout detection, OCR). This takes a few minutes on first use and is cached afterwards.
- **Notebooks kernel**: The `notebooks/` Jupyter notebooks require `sys.path.insert(0, '..')` (already present) to resolve project imports from their subdirectory. Launch Jupyter from the project root.
- **pymilvus + pyarrow conflict**: Calling `DataFrame.to_parquet()` in the same kernel as pymilvus fails with an extension-type registration error. Use CSV instead (already done in the notebooks).

---

## 23. Evaluation Dataset Generation (`evaluation/eval_dataset_generator.py`)

### Overview

`EvalDatasetGenerator` produces a labelled Q&A dataset by mining the existing Milvus collection. It requires no hand-labelling — the LLM creates questions from real indexed content. The dataset is used in Phase 4b to measure retrieval quality (Recall@K, MRR).

### Class API

```python
generator = EvalDatasetGenerator(vector_store: VectorStore, ollama_client: OllamaClient)
summary = generator.generate(output_path: Path = EVAL_DATASET_PATH) -> dict
# Returns {"total_questions": N, "cluster_k": K, "output_path": "..."}
```

Instantiation does not fetch data. `generate()` runs the full pipeline and writes the JSON file.

### Step 1 — Fetch all embeddings

Call `vector_store.fetch_all_embeddings()` to retrieve every chunk's `chunk_id`, `paper_id`, and `embedding` as a list of dicts. This method must use the same pagination pattern as `fetch_all_content()` (page through all segments using `query()` with `output_fields=["chunk_id", "paper_id", "embedding"]`).

Build two parallel numpy arrays:
- `embeddings`: shape `(N, EMBEDDING_DIM)` — float32
- `chunk_ids`: list of N strings (same order as rows)
- `paper_ids`: list of N strings (same order as rows)

If N < `EVAL_CLUSTER_K_MIN * EVAL_CHUNKS_PER_GROUP`, raise a `ValueError` with a message telling the user to ingest more papers first.

### Step 2 — K-Means clustering with silhouette refinement

Determine optimal K:
1. Initial heuristic: `k_init = max(EVAL_CLUSTER_K_MIN, min(EVAL_CLUSTER_K_MAX, int(sqrt(N / 2))))`
2. If N > 5000, subsample 5000 random rows for the silhouette sweep (use these for scoring only; full N is always clustered at the chosen K).
3. Sweep K from `EVAL_CLUSTER_K_MIN` to `min(EVAL_CLUSTER_K_MAX, N // EVAL_CHUNKS_PER_GROUP)` inclusive.
4. For each K: fit `KMeans(n_clusters=K, n_init=10, random_state=42)`, compute `silhouette_score` on the subsample.
5. Choose K with the highest silhouette score.
6. Fit the final `KMeans` on all N embeddings with the chosen K and store `cluster_labels` (length N array).

Print chosen K and silhouette score to stdout.

### Step 3 — Stratified seed sampling

Determine how many seed chunks to draw: `n_seeds = ceil(EVAL_NUM_QUESTIONS / EVAL_QUESTIONS_PER_GROUP)`.

Stratify by `paper_id`:
1. Count chunks per paper; compute each paper's fractional share of N.
2. Allocate `round(share * n_seeds)` seed slots to each paper; adjust rounding so total = `n_seeds`.
3. For each paper, sample its allocated slots by choosing indices uniformly at random from that paper's chunks, **without replacement** (or with replacement if the paper has fewer chunks than its slot count).

The result is a list of `n_seeds` chunk indices (row positions in the `embeddings` array).

### Step 4 — Nearest-neighbour group assembly

For each seed index:
1. Find all other indices that share the same cluster label.
2. Compute cosine similarity between the seed embedding and all same-cluster embeddings using numpy dot product (vectors are already L2-normalised from the embedder, so `dot` equals cosine similarity).
3. Take the top `EVAL_CHUNKS_PER_GROUP - 1` by similarity (excluding the seed itself).
4. Group = [seed_chunk_id] + [neighbour_chunk_id, …]; deduplicate by chunk_id.

### Step 5 — Fetch chunk content

For each group, call `vector_store.fetch_by_ids(group_chunk_ids)` to materialise the full `Chunk` objects (including `content`, `paper_id`, `section`, `page`).

### Step 6 — LLM Q&A generation

Build a prompt from the group's chunks and call `ollama_client.generate(prompt)`. Parse the response as JSON.

#### Prompt template

```
You are an expert researcher creating evaluation questions for a RAG system.

Below are {n} text passages from academic papers. Each passage is identified by its chunk_id.

{for i, chunk in enumerate(chunks, 1)}
[{i}] chunk_id: {chunk.chunk_id}
Paper: {chunk.title} ({chunk.year})
Section: {chunk.section}
---
{chunk.content}
{end for}

Generate exactly {EVAL_QUESTIONS_PER_GROUP} question-answer pairs based ONLY on the information in these passages.

Requirements:
- Vary difficulty: include "easy" (single-passage factual recall), "medium" (requires connecting two passages), and "hard" (requires inference, comparison, or critical thinking).
- Ground every answer strictly in the passages — do not hallucinate.
- For each Q&A pair, list the chunk_id(s) from the passages above that are needed to answer the question.

Return a JSON array only — no markdown, no extra text:
[
  {{
    "question": "...",
    "answer": "...",
    "difficulty": "easy|medium|hard",
    "relevant_chunk_ids": ["<chunk_id>", ...]
  }}
]
```

#### Parsing and error handling

- Parse the LLM response with `json.loads()`.
- If parsing fails or the response does not match the expected schema, log a warning (`logging.warning`) and skip the group — do not raise.
- Validate that each item has all four keys; drop any malformed items.
- Accept partial results from a group (e.g., if 2 of 3 items parsed).

### Step 7 — Accumulate and write output

Collect generated items until `total_collected >= EVAL_NUM_QUESTIONS`. Stop early once the target is reached (do not process remaining groups).

Assign a `question_id` (UUID4) to each item. Add `cluster_id` and `source_paper_ids` derived from the group's chunk objects.

Write `data/eval_dataset.json`:

```json
{
  "generated_at": "<ISO-8601 UTC timestamp>",
  "model": "<EVAL_LLM_MODEL>",
  "total_questions": 50,
  "cluster_k": 12,
  "items": [
    {
      "question_id": "<uuid4>",
      "question": "...",
      "answer": "...",
      "difficulty": "easy|medium|hard",
      "relevant_chunk_ids": ["<chunk_id>", ...],
      "source_paper_ids": ["<paper_id>", ...],
      "cluster_id": 3
    }
  ]
}
```

Print a summary to stdout: total questions generated, K chosen, output path.

### CLI integration (`main.py`)

Add `--eval-generate` flag to the `argparse` parser. When set:
1. Initialise `vector_store` (no need to start the full pipeline — only `vector_store` and `ollama_client` are needed).
2. Instantiate `EvalDatasetGenerator(vector_store, ollama_client)`.
3. Call `generator.generate()` and print the returned summary.
4. Exit.

### Dependencies added

Add `scikit-learn` to `requirements.txt`. No other new dependencies.

### Phase 4b scorer (`evaluation/eval_scorer.py`)

`EvalScorer(pipeline).score()` loads `eval_dataset.json`, calls `pipeline.retrieve(question)` for each item (no LLM call), compares retrieved chunk_ids against `relevant_chunk_ids`, and writes `eval_results.json` with per-question Recall@1/3/5, MRR, and a breakdown by difficulty.
