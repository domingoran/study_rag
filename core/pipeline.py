"""
Orchestrates the full RAG pipeline.

Ingestion path:
    PDF  →  parse  →  chunk  →  metadata  →  embed  →  Milvus
    (after each ingest session)  →  rebuild BM25 index

Query path (Phase 2):
    question
      → embed (dense)
      → HybridSearcher  [vector + BM25 fusion]  → TOP_K_RERANK candidates
      → Reranker        [LLM reorder]            → TOP_K_FINAL chunks
      → ChatEngine      [Ollama answer]           → answer + sources
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

import config
from core.schemas import Chunk
from embeddings.embedder import Embedder
from ingestion.chunker import build_embedding_text, chunk_document
from ingestion.docling_parser import parse_pdf
from ingestion.metadata_builder import build_metadata
from llm.chat_engine import ChatEngine
from llm.ollama_client import OllamaClient
from retrieval.bm25_index import BM25Index
from retrieval.hybrid_search import HybridSearcher
from retrieval.reranker import Reranker
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Top-level object that owns all pipeline components.

    Components are initialised lazily (``_init_*``) so that starting the
    chat mode doesn't block on loading the embedding model if Milvus has
    already been populated.

    Typical usage::

        pipeline = RAGPipeline()
        pipeline.ingest_folder(config.DATA_DIR)
        answer, chunks = pipeline.query("What is self-attention?")
    """

    def __init__(self) -> None:
        self._embedder: Optional[Embedder] = None
        self._vector_store: Optional[VectorStore] = None
        self._bm25_index: Optional[BM25Index] = None
        self._hybrid_searcher: Optional[HybridSearcher] = None
        self._reranker: Optional[Reranker] = None
        self._chat_engine: Optional[ChatEngine] = None
        self._ollama_client: Optional[OllamaClient] = None

    # ------------------------------------------------------------------ #
    # Lazy initialisation                                                  #
    # ------------------------------------------------------------------ #

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore()
        return self._vector_store

    @property
    def bm25_index(self) -> BM25Index:
        if self._bm25_index is None:
            self._bm25_index = BM25Index()
            # Try to load a previously saved index; build from Milvus if missing
            if not self._bm25_index.load():
                self._rebuild_bm25(silent=True)
        return self._bm25_index

    @property
    def hybrid_searcher(self) -> HybridSearcher:
        if self._hybrid_searcher is None:
            self._hybrid_searcher = HybridSearcher(self.vector_store, self.bm25_index)
        return self._hybrid_searcher

    @property
    def ollama_client(self) -> OllamaClient:
        if self._ollama_client is None:
            self._ollama_client = OllamaClient()
        return self._ollama_client

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    @property
    def chat_engine(self) -> ChatEngine:
        if self._chat_engine is None:
            self._chat_engine = ChatEngine(self.ollama_client)
        return self._chat_engine

    # ------------------------------------------------------------------ #
    # Ingestion                                                            #
    # ------------------------------------------------------------------ #

    def ingest_pdf(self, pdf_path: Path) -> int:
        """
        Run the full ingestion pipeline for a single PDF.

        Skips the file if its paper_id already exists in Milvus.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Number of chunks inserted (0 if skipped or empty).
        """
        pdf_path = Path(pdf_path)
        paper_id = pdf_path.stem

        if self.vector_store.has_paper(paper_id):
            logger.info("Skipping '%s' — already in vector store.", paper_id)
            print(f"  ↷ Skipping '{pdf_path.name}' (already ingested)")
            return 0

        # 1. Parse
        print(f"  📄 Parsing   {pdf_path.name} …")
        parsed_doc = parse_pdf(pdf_path)

        # 2. Chunk
        print(f"  ✂  Chunking  …")
        chunks = chunk_document(parsed_doc)

        # 3. Metadata enrichment / cleanup
        chunks = build_metadata(chunks)

        if not chunks:
            logger.warning("No chunks produced for '%s' — skipping.", paper_id)
            print(f"  ⚠  No chunks produced for '{pdf_path.name}'")
            return 0

        # 4. Embed
        # [BREADCRUMB] (#4) embed a "Paper / Section" header + content, but keep
        # chunk.content (stored in Milvus / BM25 / LLM context) breadcrumb-free.
        print(f"  🔢 Embedding {len(chunks)} chunks …")
        texts = [build_embedding_text(c) for c in chunks]
        embeddings = self.embedder.embed_passages(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # 5. Store
        print(f"  💾 Storing   {len(chunks)} chunks in Milvus …")
        self.vector_store.insert(chunks)

        print(f"  ✓  Done — {len(chunks)} chunks indexed.")
        return len(chunks)

    def ingest_folder(self, folder: Path | None = None) -> None:
        """
        Ingest all PDF files found in *folder* (default: config.DATA_DIR).

        Rebuilds the BM25 index after all new papers have been inserted.

        Args:
            folder: Directory to scan for PDF files.
        """
        folder = Path(folder or config.DATA_DIR)
        pdf_files = sorted(folder.glob("*.pdf"))

        if not pdf_files:
            print(f"\n⚠  No PDF files found in {folder}\n"
                  "   Drop some papers into data/papers/ and re-run with --ingest")
            return

        print(f"\nFound {len(pdf_files)} PDF file(s) in {folder}\n")
        total_chunks = 0
        for pdf_path in pdf_files:
            print(f"→  {pdf_path.name}")
            try:
                n = self.ingest_pdf(pdf_path)
                total_chunks += n
            except Exception as exc:
                logger.exception("Failed to ingest '%s'", pdf_path.name)
                print(f"  ✗  Error: {exc}")

        print(f"\n✅ Ingestion complete — {total_chunks} new chunks stored.")

        if total_chunks > 0:
            print("\n🔄 Rebuilding BM25 index …")
            self._rebuild_bm25(silent=False)

    # ------------------------------------------------------------------ #
    # BM25 index management                                               #
    # ------------------------------------------------------------------ #

    def _rebuild_bm25(self, silent: bool = False) -> None:
        """
        Fetch all chunk content from Milvus and rebuild the BM25 index.

        Args:
            silent: If True, suppress progress prints (used on startup).
        """
        if not silent:
            print("  Fetching all chunk content from Milvus …")

        items = self.vector_store.fetch_all_content()

        if not items:
            logger.info("No chunks in Milvus yet — BM25 index not built.")
            return

        if self._bm25_index is None:
            self._bm25_index = BM25Index()

        self._bm25_index.build(items)
        self._bm25_index.save()

        if not silent:
            print(f"  ✓  BM25 index saved ({len(items)} chunks).")

        # Invalidate the hybrid searcher so it picks up the fresh index
        self._hybrid_searcher = None

    def rebuild_bm25(self) -> None:
        """Public method — triggered by the /bm25 CLI command."""
        print("\n🔄 Rebuilding BM25 index …")
        self._rebuild_bm25(silent=False)
        print()

    def full_reset(self) -> None:
        """
        Wipe all indexed data and reset every in-memory component.

        Actions performed:
          1. Drop and recreate the Milvus collection (empty).
          2. Delete the BM25 index file from disk.
          3. Reset all lazy-loaded components so they reinitialise cleanly
             on the next query.
        """
        # 1. Wipe Milvus collection
        self.vector_store.reset()

        # 2. Delete BM25 index file
        bm25_path = config.BM25_INDEX_PATH
        if bm25_path.exists():
            bm25_path.unlink()
            logger.info("BM25 index file deleted: %s", bm25_path)

        # 3. Reset retrieval components (forces re-init on next use).
        #    _embedder, _ollama_client and _chat_engine are kept alive —
        #    they hold no indexed data and are expensive to reload.
        self._bm25_index      = None
        self._hybrid_searcher = None
        self._reranker        = None
        logger.info("Pipeline full reset complete.")

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        question: str,
        top_k: int = config.TOP_K_FINAL,
        expr: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Embed → hybrid search → rerank, returning top-K chunks without LLM generation.
        Used by the evaluation scorer to avoid paying for an LLM call per question.
        """
        query_vec = self.embedder.embed_query(question)
        candidates = self.hybrid_searcher.search(
            query_embedding=query_vec,
            query_text=question,
            top_k=config.TOP_K_RERANK,
            expr=expr,
        )
        if not candidates:
            return []
        if config.RERANKER_ENABLED and len(candidates) > top_k:
            candidates = self.reranker.rerank(question, candidates)
        return candidates[:top_k]

    def query(
        self,
        question: str,
        top_k: int = config.TOP_K_FINAL,
        expr: Optional[str] = None,
    ) -> Tuple[str, List[Chunk]]:
        """
        Answer a question using Phase 2 hybrid retrieval + LLM reranking.

        Pipeline:
          1. Embed the question (dense vector).
          2. HybridSearcher fuses vector + BM25 → up to TOP_K_RERANK candidates.
          3. Reranker reorders with LLM → top TOP_K_FINAL chunks.
          4. ChatEngine generates the answer.

        Args:
            question: User's natural-language question.
            top_k:    Number of chunks to pass to the LLM.
            expr:     Optional Milvus scalar filter expression.

        Returns:
            (answer_text, retrieved_chunks)
        """
        # 1. Embed
        query_vec = self.embedder.embed_query(question)

        # 2. Hybrid retrieval (vector + BM25)
        candidates = self.hybrid_searcher.search(
            query_embedding=query_vec,
            query_text=question,
            top_k=config.TOP_K_RERANK,
            expr=expr,
        )

        if not candidates:
            return (
                "I could not find relevant information in the indexed papers. "
                "Make sure you have run the ingestion pipeline first (--ingest).",
                [],
            )

        # 3. LLM reranking
        if config.RERANKER_ENABLED and len(candidates) > top_k:
            candidates = self.reranker.rerank(question, candidates)

        # Clip to the final top-K
        chunks = candidates[:top_k]

        # 4. Generate answer
        answer = self.chat_engine.answer(question, chunks)
        return answer, chunks
