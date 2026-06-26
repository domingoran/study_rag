"""
Milvus vector store wrapper — uses the modern MilvusClient API (pymilvus ≥ 2.4).

Handles:
  • Connection management
  • Collection creation (custom schema + HNSW index)
  • Insert
  • Similarity search
  • Paper existence check (to skip re-ingestion)
"""
from __future__ import annotations

import logging
from typing import List, Optional

from pymilvus import DataType, MilvusClient

import config
from core.schemas import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)

# Maximum VARCHAR lengths
_LEN_CHUNK_ID   = 64
_LEN_PAPER_ID   = 256
_LEN_TITLE      = 512
_LEN_AUTHORS    = 512
_LEN_SECTION    = 512
_LEN_CHUNK_TYPE = 32
_LEN_CONTENT    = 8192


class VectorStore:
    """
    Manages the Milvus `academic_chunks` collection via MilvusClient.

    Usage::

        store = VectorStore()
        store.insert(chunks)
        results = store.search(query_embedding, top_k=5)
    """

    def __init__(self) -> None:
        # Use 127.0.0.1 explicitly to avoid IPv6 resolution issues in WSL2
        uri = f"http://127.0.0.1:{config.MILVUS_PORT}"
        logger.info("Connecting to Milvus at %s", uri)
        self._client = MilvusClient(uri=uri)
        self._ensure_collection()

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def _ensure_collection(self) -> None:
        if self._client.has_collection(config.MILVUS_COLLECTION):
            logger.info("Collection '%s' already exists.", config.MILVUS_COLLECTION)
            return

        logger.info("Creating collection '%s'.", config.MILVUS_COLLECTION)

        schema = self._client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field("chunk_id",   DataType.VARCHAR, is_primary=True, max_length=_LEN_CHUNK_ID)
        schema.add_field("paper_id",   DataType.VARCHAR, max_length=_LEN_PAPER_ID)
        schema.add_field("title",      DataType.VARCHAR, max_length=_LEN_TITLE)
        schema.add_field("authors",    DataType.VARCHAR, max_length=_LEN_AUTHORS)
        schema.add_field("year",       DataType.INT64)
        schema.add_field("section",    DataType.VARCHAR, max_length=_LEN_SECTION)
        schema.add_field("chunk_type", DataType.VARCHAR, max_length=_LEN_CHUNK_TYPE)
        schema.add_field("content",    DataType.VARCHAR, max_length=_LEN_CONTENT)
        schema.add_field("page",       DataType.INT64)
        schema.add_field("embedding",  DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIM)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            metric_type="COSINE",
            index_type="HNSW",
            params={"M": 16, "efConstruction": 128},
        )

        self._client.create_collection(
            collection_name=config.MILVUS_COLLECTION,
            schema=schema,
            index_params=index_params,
        )
        logger.info("Collection created and indexed.")

    # ------------------------------------------------------------------ #
    # Write                                                                #
    # ------------------------------------------------------------------ #

    def has_paper(self, paper_id: str) -> bool:
        """Return True if any chunks for *paper_id* already exist."""
        results = self._client.query(
            collection_name=config.MILVUS_COLLECTION,
            filter=f'paper_id == "{paper_id}"',
            output_fields=["chunk_id"],
            limit=1,
        )
        return len(results) > 0

    def insert(self, chunks: List[Chunk]) -> None:
        """
        Insert a list of embedded Chunks.

        Args:
            chunks: Chunks with .embedding populated.

        Raises:
            ValueError: if any chunk is missing its embedding.
        """
        if not chunks:
            logger.warning("insert() called with empty list.")
            return

        missing = [c.chunk_id for c in chunks if c.embedding is None]
        if missing:
            raise ValueError(f"{len(missing)} chunk(s) missing embedding: {missing[:3]}")

        data = [
            {
                "chunk_id":   c.chunk_id[:_LEN_CHUNK_ID - 1],
                "paper_id":   c.paper_id[:_LEN_PAPER_ID - 1],
                "title":      c.title[:_LEN_TITLE - 1],
                "authors":    ", ".join(c.authors)[:_LEN_AUTHORS - 1],
                "year":       c.year,
                "section":    c.section[:_LEN_SECTION - 1],
                "chunk_type": c.chunk_type[:_LEN_CHUNK_TYPE - 1],
                "content":    c.content[:_LEN_CONTENT - 1],
                "page":       c.metadata.page,
                "embedding":  c.embedding,
            }
            for c in chunks
        ]

        self._client.insert(
            collection_name=config.MILVUS_COLLECTION,
            data=data,
        )
        # Flush seals the write buffer so the data is immediately visible
        # to count() and get_collection_stats() — without this Milvus may
        # report 0 rows until it auto-flushes in the background.
        self._client.flush(collection_name=config.MILVUS_COLLECTION)
        logger.info("Inserted and flushed %d chunks into '%s'.", len(chunks), config.MILVUS_COLLECTION)

    # ------------------------------------------------------------------ #
    # Read                                                                 #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _row_to_chunk(chunk_id: str, entity: dict) -> Chunk:
        """Reconstruct a Chunk from a Milvus hit/row entity dict."""
        authors_str = entity.get("authors") or ""
        authors = [a.strip() for a in authors_str.split(",") if a.strip()]
        return Chunk(
            chunk_id=chunk_id,
            paper_id=entity.get("paper_id", ""),
            title=entity.get("title", ""),
            authors=authors,
            year=entity.get("year", 0),
            section=entity.get("section", ""),
            chunk_type=entity.get("chunk_type", "text"),
            content=entity.get("content", ""),
            metadata=ChunkMetadata(page=entity.get("page", 0)),
        )

    # ------------------------------------------------------------------ #
    # Read                                                                 #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query_embedding: List[float],
        top_k: int = config.TOP_K_FINAL,
        expr: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Nearest-neighbour search.

        Args:
            query_embedding: Query vector (length == EMBEDDING_DIM).
            top_k:           Number of results.
            expr:            Optional Milvus boolean filter, e.g.
                             'year >= 2020 && chunk_type == "text"'

        Returns:
            List of Chunk objects reconstructed from stored fields.
        """
        return [c for c, _ in self.search_with_scores(query_embedding, top_k, expr)]

    def search_with_scores(
        self,
        query_embedding: List[float],
        top_k: int = config.TOP_K_VECTOR,
        expr: Optional[str] = None,
    ) -> List[tuple]:
        """
        Like search() but also returns the raw cosine similarity score.

        Returns:
            List of (Chunk, float) tuples sorted by descending score.
        """
        output_fields = [
            "chunk_id", "paper_id", "title", "authors",
            "year", "section", "chunk_type", "content", "page",
        ]

        results = self._client.search(
            collection_name=config.MILVUS_COLLECTION,
            data=[query_embedding],
            anns_field="embedding",
            limit=top_k,
            filter=expr,
            output_fields=output_fields,
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        )

        pairs = []
        for hit in results[0]:
            chunk = self._row_to_chunk(hit["id"], hit["entity"])
            score = float(hit.get("distance", 0.0))
            pairs.append((chunk, score))

        return pairs

    def fetch_all_content(self) -> List[tuple]:
        """
        Fetch (chunk_id, content) for every entity in the collection.

        Used to build / rebuild the BM25 index.  Paginates internally so
        collections of any size are handled safely.

        Returns:
            List of (chunk_id, content) string pairs.
        """
        items: List[tuple] = []
        batch_size = 1000

        iterator = self._client.query_iterator(
            collection_name=config.MILVUS_COLLECTION,
            filter='chunk_id != ""',
            output_fields=["chunk_id", "content"],
            batch_size=batch_size,
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for row in batch:
                    items.append((row["chunk_id"], row["content"]))
        finally:
            iterator.close()

        logger.info("fetch_all_content: returned %d items.", len(items))
        return items

    def fetch_all_embeddings(self) -> List[dict]:
        """
        Fetch chunk_id, paper_id, and embedding for every entity in the collection.

        Used by EvalDatasetGenerator for clustering.  Paginates internally.

        Returns:
            List of {"chunk_id": str, "paper_id": str, "embedding": list[float]}.
        """
        items: List[dict] = []
        batch_size = 1000

        iterator = self._client.query_iterator(
            collection_name=config.MILVUS_COLLECTION,
            filter='chunk_id != ""',
            output_fields=["chunk_id", "paper_id", "embedding"],
            batch_size=batch_size,
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for row in batch:
                    items.append({
                        "chunk_id": row["chunk_id"],
                        "paper_id": row["paper_id"],
                        "embedding": row["embedding"],
                    })
        finally:
            iterator.close()

        logger.info("fetch_all_embeddings: returned %d items.", len(items))
        return items

    def fetch_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """
        Fetch full Chunk objects for a list of chunk_ids.

        Used by HybridSearcher to materialise BM25 hits that weren't
        returned by the vector search.

        Args:
            chunk_ids: List of chunk_id strings to fetch.

        Returns:
            List of Chunk objects (order is not guaranteed).
        """
        if not chunk_ids:
            return []

        ids_csv = ", ".join(f'"{cid}"' for cid in chunk_ids)
        filter_expr = f"chunk_id in [{ids_csv}]"

        output_fields = [
            "chunk_id", "paper_id", "title", "authors",
            "year", "section", "chunk_type", "content", "page",
        ]

        rows = self._client.query(
            collection_name=config.MILVUS_COLLECTION,
            filter=filter_expr,
            output_fields=output_fields,
            limit=len(chunk_ids),
        )

        return [self._row_to_chunk(row["chunk_id"], row) for row in rows]

    def delete_paper(self, paper_id: str) -> int:
        expr = f'paper_id == "{paper_id}"'
        result = self._client.delete(
            collection_name=config.MILVUS_COLLECTION,
            filter=expr,
        )
        deleted = result.get("delete_count", 0) if isinstance(result, dict) else result
        logger.info("Deleted %s chunks for paper_id='%s'.", deleted, paper_id)
        return deleted

    def reset(self) -> None:
        """
        Drop and recreate the collection, wiping all indexed data.

        After this call the collection exists but is empty, ready for
        fresh ingestion.
        """
        if self._client.has_collection(config.MILVUS_COLLECTION):
            self._client.drop_collection(config.MILVUS_COLLECTION)
            logger.info("Collection '%s' dropped.", config.MILVUS_COLLECTION)
        self._ensure_collection()
        logger.info("Collection '%s' recreated (empty).", config.MILVUS_COLLECTION)

    def count(self) -> int:
        """
        Return total number of entities in the collection.

        Uses query(count(*)) rather than get_collection_stats() because
        the stats endpoint only counts sealed (flushed) segments and can
        return 0 for data that was just inserted but not yet auto-flushed.
        """
        result = self._client.query(
            collection_name=config.MILVUS_COLLECTION,
            filter='chunk_id != ""',
            output_fields=["count(*)"],
        )
        return int(result[0]["count(*)"]) if result else 0
