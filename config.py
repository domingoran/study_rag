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
#
# Boundary-aware splitting (improvement #1): text buffers are packed sentence
# by sentence. A chunk grows until it reaches CHUNK_IDEAL_TOKENS, then closes at
# the next paragraph boundary; if none arrives before CHUNK_MAX_TOKENS it closes
# at the current sentence boundary instead. This avoids cutting mid-sentence.
CHUNK_MIN_TOKENS: int = 250      # below this a trailing chunk is acceptable but not forced
CHUNK_IDEAL_TOKENS: int = 450    # target size; chunk starts looking for a boundary here
CHUNK_MAX_TOKENS: int = 512      # hard ceiling; force a sentence-boundary cut at/above this
CHUNK_OVERLAP_TOKENS: int = 50   # carried over as whole trailing sentences (semantic overlap)

# Phase 3 — context stitching for non-text chunks (tables, figures, equations).
# Words of surrounding prose stitched in before and after each non-text element.
CONTEXT_WORDS: int = 50
# Table Markdown is capped at this many chars before context is appended,
# ensuring context snippets are never silently cut by the Milvus VARCHAR limit.
MAX_TABLE_MARKDOWN_CHARS: int = 6_000

# Abbreviations that end in a period but do NOT end a sentence. The boundary-aware
# splitter (#1) checks only the last token of each candidate sentence against this
# set (one lowercased lookup per sentence — negligible cost) and merges the split
# back when it matches, so "et al. Smith (2017)" stays one sentence. Lowercase,
# trailing dot included.
SENTENCE_ABBREVIATIONS: frozenset[str] = frozenset({
    "al.", "e.g.", "i.e.", "cf.", "etc.", "vs.", "viz.", "no.", "nos.",
    "fig.", "figs.", "eq.", "eqs.", "ref.", "refs.", "sec.", "secs.",
    "ch.", "pp.", "approx.", "resp.", "dr.", "prof.", "mr.", "mrs.", "ms.",
    "st.", "ca.", "vol.", "ed.", "eds.", "tab.", "thm.", "def.", "prop.",
})

# Improvement #15 — reference/bibliography sections are retrieval noise and are
# dropped during chunking. A section header whose text matches this pattern (and
# everything under it, until the next non-matching header) is skipped entirely.
EXCLUDE_SECTION_RE: str = r"^\s*(references|bibliography|works\s+cited)\b"

# ---------------------------------------------------------------------------
# Breadcrumb embedding context (improvement #4) — [BREADCRUMB]
# ---------------------------------------------------------------------------
# When True, a "Paper: … / Section: …" header is prepended to each chunk *only
# for embedding*. It is NOT stored in `content`, so it never appears in the
# LLM's citation context or the BM25 index — it only steers the dense vector.
# Grep for "[BREADCRUMB]" to find every line involved in this feature.
EMBED_BREADCRUMB: bool = True

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K_VECTOR: int = 20   # dense candidates fetched from Milvus
TOP_K_BM25: int = 20     # sparse candidates fetched from BM25
TOP_K_RERANK: int = 10   # merged candidates passed into the reranker
TOP_K_FINAL: int = 5     # top chunks shown to the LLM

# Hybrid fusion weights (must sum to 1.0)
HYBRID_VECTOR_WEIGHT: float = 0.6
HYBRID_BM25_WEIGHT: float = 0.4

# BM25 index on-disk location (rebuilt automatically after ingestion)
BM25_INDEX_PATH: Path = BASE_DIR / "data" / "bm25_index.pkl"

# Cross-encoder reranker model from HuggingFace (sentence-transformers CrossEncoder).
RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

# Set to False to skip the reranking step (faster but lower quality)
RERANKER_ENABLED: bool = True

# ---------------------------------------------------------------------------
# Evaluation — dataset generation (Phase 4a)
# ---------------------------------------------------------------------------
EVAL_DATASET_PATH: Path = BASE_DIR / "data" / "eval_dataset.json"
EVAL_RESULTS_PATH: Path = BASE_DIR / "data" / "eval_results.json"
EVAL_NUM_QUESTIONS: int = 50        # total Q&A pairs to generate
EVAL_CLUSTER_K_MIN: int = 5         # min K for silhouette-score sweep
EVAL_CLUSTER_K_MAX: int = 35        # max K for silhouette-score sweep
EVAL_CHUNKS_PER_GROUP: int = 4      # chunks per LLM call (1 seed + 3 neighbours)
EVAL_QUESTIONS_PER_GROUP: int = 3   # Q&A pairs generated per chunk group
EVAL_LLM_MODEL: str = OLLAMA_CHAT_MODEL  # override here to use a different model
