"""
Central configuration for the RAG Study pipeline.
All tuneable parameters live here — change values here, not in module code.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "papers"

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
# Any HuggingFace sentence-transformers model can be swapped in here.
# bge-base-en-v1.5 works well for English academic text (768-dim).
EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM: int = 768
EMBEDDING_BATCH_SIZE: int = 32

# Prefix prepended to *queries* only (not to document passages).
# bge-base-en-v1.5 is instruction-tuned for retrieval with this prefix.
# Set to "" if you switch to a model that doesn't use instructions.
EMBEDDING_QUERY_INSTRUCTION: str = "Represent this sentence for searching relevant passages: "

# ---------------------------------------------------------------------------
# Milvus
# ---------------------------------------------------------------------------
MILVUS_HOST: str = "localhost"
MILVUS_PORT: int = 19530
MILVUS_COLLECTION: str = "academic_chunks"

# ---------------------------------------------------------------------------
# Ollama / LLM
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_CHAT_MODEL: str = "llama3.1"  # change to any model pulled in Ollama

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# Approximate token budget per chunk (1 token ≈ 0.75 words in English).
CHUNK_MAX_TOKENS: int = 512
CHUNK_OVERLAP_TOKENS: int = 50

# Phase 3 — context stitching for non-text chunks (tables, figures, equations).
# Words of surrounding prose stitched in before and after each non-text element.
CONTEXT_WORDS: int = 50
# Table Markdown is capped at this many chars before context is appended,
# ensuring context snippets are never silently cut by the Milvus VARCHAR limit.
MAX_TABLE_MARKDOWN_CHARS: int = 6_000

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K_VECTOR: int = 10   # dense candidates fetched from Milvus
TOP_K_BM25: int = 10     # sparse candidates fetched from BM25
TOP_K_RERANK: int = 20   # merged candidates passed into the reranker
TOP_K_FINAL: int = 5     # top chunks shown to the LLM

# Hybrid fusion weights (must sum to 1.0)
HYBRID_VECTOR_WEIGHT: float = 0.6
HYBRID_BM25_WEIGHT: float = 0.4

# BM25 index on-disk location (rebuilt automatically after ingestion)
BM25_INDEX_PATH: Path = BASE_DIR / "data" / "bm25_index.pkl"

# Set to False to skip the LLM reranking step (faster but lower quality)
RERANKER_ENABLED: bool = True
