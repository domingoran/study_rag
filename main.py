"""
Entry point for the RAG Study CLI.

Usage
-----
Ingest all PDFs in data/papers/:
    python main.py --ingest

Start the interactive chat:
    python main.py

Ingest then immediately start chat:
    python main.py --ingest --chat
"""
from __future__ import annotations

import argparse
import logging
from typing import Optional

import config
from core.pipeline import RAGPipeline
from evaluation.eval_dataset_generator import EvalDatasetGenerator
from evaluation.eval_scorer import EvalScorer

# ---------------------------------------------------------------------------
# Logging setup — INFO for our code, WARNING for noisy third-party libs
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s  %(message)s",
)
logging.getLogger("rag_study").setLevel(logging.INFO)

BANNER = """\
╔══════════════════════════════════════════╗
║       RAG Study — Academic Paper Q&A    ║
╚══════════════════════════════════════════╝
Model : {model}
Store : {host}:{port}  collection={collection}

Commands:
  /reset          — clear conversation history
  /count          — show number of indexed chunks
  /chunk <id>     — retrieve and display a chunk by its chunk_id
  /bm25           — rebuild BM25 index from current Milvus data
  /filter <expr>  — set a Milvus scalar filter  (e.g. year >= 2022)
  /filter clear   — remove the current filter
  /delete <paper>  — delete all chunks for a paper and rebuild BM25
  /delete         — wipe the entire vector store and BM25 index
  /quit           — exit  (also: exit, q)
"""


def run_eval_score(pipeline: RAGPipeline, retrieve_only: bool = False) -> None:
    label = "Retrieval-Only Evaluation" if retrieve_only else "Retrieval Evaluation Scoring"
    print(f"\n=== {label} ===")
    scorer = EvalScorer(pipeline)
    scorer.score(retrieve_only=retrieve_only)


def run_eval_generate(pipeline: RAGPipeline) -> None:
    print("\n=== Evaluation Dataset Generation ===")
    generator = EvalDatasetGenerator(pipeline.vector_store, pipeline.ollama_client)
    summary = generator.generate()
    print(
        f"\nSummary:\n"
        f"  Questions : {summary['total_questions']}\n"
        f"  Clusters  : {summary['cluster_k']}\n"
        f"  Output    : {summary['output_path']}"
    )


def run_manual_generator(pipeline: RAGPipeline) -> None:
    print("\n=== Manual Evaluation Dataset Generator ===")
    generator = EvalDatasetGenerator(pipeline.vector_store, pipeline.ollama_client)
    generator.manual_generate()


def run_ingest(pipeline: RAGPipeline) -> None:
    print("\n=== Ingestion Mode ===")
    pipeline.ingest_folder(config.DATA_DIR)


def run_chat(pipeline: RAGPipeline) -> None:
    print(BANNER.format(
        model=config.OLLAMA_CHAT_MODEL,
        host=config.MILVUS_HOST,
        port=config.MILVUS_PORT,
        collection=config.MILVUS_COLLECTION,
    ))

    # Warn if Ollama model is unreachable
    if not pipeline.ollama_client.is_available():
        print(
            f"⚠  Warning: Ollama model '{config.OLLAMA_CHAT_MODEL}' not found.\n"
            f"   Run:  ollama pull {config.OLLAMA_CHAT_MODEL}\n"
        )

    # Active Milvus scalar filter (set via /filter)
    active_filter: Optional[str] = None

    while True:
        # Show filter hint in prompt when a filter is active
        prompt = f"[filter: {active_filter}] You: " if active_filter else "You: "

        try:
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # ----------------------------------------------------------------
        # Built-in commands
        # ----------------------------------------------------------------
        if user_input.lower() in ("/quit", "quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input.lower() == "/reset":
            pipeline.chat_engine.reset()
            print("🔄 Conversation history cleared.\n")
            continue

        if user_input.lower() == "/count":
            n = pipeline.vector_store.count()
            print(f"📦 {n} chunks currently in the vector store.\n")
            continue

        if user_input.lower().startswith("/chunk"):
            arg = user_input[len("/chunk"):].strip()
            if not arg:
                print("Usage: /chunk <chunk_id>\n")
                continue
            results = pipeline.vector_store.fetch_by_ids([arg])
            if not results:
                print(f"⚠  No chunk found with id '{arg}'\n")
                continue
            chunk = results[0]
            authors_str = ", ".join(chunk.authors) if chunk.authors else "—"
            print(
                f"\n{'─' * 60}\n"
                f"  chunk_id   : {chunk.chunk_id}\n"
                f"  paper_id   : {chunk.paper_id}\n"
                f"  title      : {chunk.title}\n"
                f"  authors    : {authors_str}\n"
                f"  year       : {chunk.year}\n"
                f"  section    : {chunk.section or '—'}\n"
                f"  chunk_type : {chunk.chunk_type}\n"
                f"  page       : {chunk.metadata.page}\n"
                f"{'─' * 60}\n"
                f"{chunk.content}\n"
                f"{'─' * 60}\n"
            )
            continue

        if user_input.lower() == "/bm25":
            pipeline.rebuild_bm25()
            # Invalidate hybrid_searcher cache so it picks up fresh index
            pipeline._hybrid_searcher = None
            continue

        if user_input.lower().startswith("/filter"):
            arg = user_input[len("/filter"):].strip()
            if not arg or arg.lower() == "clear":
                active_filter = None
                print("🔍 Filter cleared.\n")
            else:
                active_filter = arg
                print(f"🔍 Filter set: {active_filter}\n")
            continue

        if user_input.lower().startswith("/delete "):
            paper_id = user_input[len("/delete "):].strip()
            if not paper_id:
                print("Usage: /delete <paper_id>\n")
                continue
            try:
                confirm = input(
                    f"⚠️  Delete all chunks for paper '{paper_id}'? Type 'yes' to confirm: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n   Aborted.\n")
                continue
            if confirm == "yes":
                deleted = pipeline.vector_store.delete_paper(paper_id)
                print(f"🗑️  Deleted {deleted} chunks. Rebuilding BM25 index …")
                pipeline.rebuild_bm25()
                pipeline._hybrid_searcher = None
                print("✓  Done.\n")
            else:
                print("   Aborted.\n")
            continue

        if user_input.lower() == "/delete":
            n = pipeline.vector_store.count()
            print(f"\n⚠️  This will permanently delete ALL {n} indexed chunks")
            print("   and the BM25 index. This cannot be undone.")
            try:
                confirm = input("   Type 'yes' to confirm: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n   Aborted.\n")
                continue
            if confirm == "yes":
                print("🗑️  Wiping vector store and BM25 index …")
                pipeline.full_reset()
                active_filter = None
                print("✓  Done. Run  python main.py --ingest  to re-index papers.\n")
            else:
                print("   Aborted.\n")
            continue

        # ----------------------------------------------------------------
        # Normal query
        # ----------------------------------------------------------------
        try:
            answer, chunks = pipeline.query(user_input, expr=active_filter)
        except Exception as exc:
            print(f"⚠  Error: {exc}\n")
            continue

        print(f"\nAssistant:\n{answer}\n")

        if chunks:
            print("Sources:")
            for chunk in chunks:
                authors_str = f"  ({', '.join(chunk.authors)})" if chunk.authors else ""
                print(
                    f"  • [{chunk.paper_id}]{authors_str}  "
                    f"{chunk.section or '—'}  "
                    f"p.{chunk.metadata.page}  "
                    f"[{chunk.chunk_type}]"
                )
                # Debug: show chunk content preview (truncated to 300 chars)
                preview = chunk.content.replace("\n", " ")
                if len(preview) > 300:
                    preview = preview[:300] + "…"
                print(f"    ↳ {preview}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Study — chat with your academic papers",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest all PDFs from data/papers/ into the vector store.",
    )
    parser.add_argument(
        "--ingest-file",
        type=str,
        metavar="PDF",
        help="Ingest a single PDF file into the vector store.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start the interactive chat after ingestion (use with --ingest).",
    )
    parser.add_argument(
        "--eval-generate",
        action="store_true",
        help="Generate an evaluation Q&A dataset from the indexed chunks.",
    )
    parser.add_argument(
        "--manual-generator",
        action="store_true",
        help="Run clustering + seed selection, then print each chunk group\n"
             "with the system prompt for manual Q&A generation via an external model.",
    )
    parser.add_argument(
        "--eval-score",
        action="store_true",
        help="Score retrieval quality against data/eval_dataset.json (Recall@K, MRR)\n"
             "and generate LLM answers for comparison.",
    )
    parser.add_argument(
        "--eval-retrieve",
        action="store_true",
        help="Score retrieval quality (Recall@K, MRR) without LLM answer generation.",
    )
    args = parser.parse_args()

    pipeline = RAGPipeline()

    if args.eval_retrieve:
        run_eval_score(pipeline, retrieve_only=True)
    elif args.eval_score:
        run_eval_score(pipeline)
    elif args.manual_generator:
        run_manual_generator(pipeline)
    elif args.eval_generate:
        run_eval_generate(pipeline)
    elif args.ingest_file:
        from pathlib import Path
        pdf = Path(args.ingest_file)
        if not pdf.exists():
            print(f"⚠  File not found: {pdf}")
            return
        print(f"\n=== Ingesting {pdf.name} ===")
        pipeline.ingest_pdf(pdf)
        if args.chat:
            run_chat(pipeline)
    elif args.ingest:
        run_ingest(pipeline)
        if args.chat:
            run_chat(pipeline)
    else:
        # Default: chat mode
        run_chat(pipeline)


if __name__ == "__main__":
    main()
