"""
Retrieval evaluation scorer — Phase 4b.

Loads data/eval_dataset.json, runs the retrieval pipeline on each question,
and computes Recall@1/3/5 and MRR against the ground-truth relevant_chunk_ids.
Writes data/eval_results.json.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from tqdm import tqdm

import config

logger = logging.getLogger(__name__)

RECALL_KS = [1, 3, 5]


def _recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for cid in retrieved[:k] if cid in relevant)
    return hits / len(relevant)


def _reciprocal_rank(retrieved: List[str], relevant: set) -> float:
    for rank, cid in enumerate(retrieved, 1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


class EvalScorer:
    def __init__(self, pipeline) -> None:
        self._pipeline = pipeline

    def score(
        self,
        dataset_path: Path = config.EVAL_DATASET_PATH,
        output_path: Path = config.EVAL_RESULTS_PATH,
        top_k: int = max(RECALL_KS),
        retrieve_only: bool = False,
    ) -> dict:
        """
        Run retrieval for every question in the dataset, compute metrics, write JSON.

        Args:
            dataset_path:  Path to eval_dataset.json produced by Phase 4a.
            output_path:   Where to write eval_results.json.
            top_k:         How many chunks to retrieve per question (default: max Recall K = 5).
            retrieve_only: If True, skip LLM answer generation (faster, retrieval metrics only).

        Returns:
            Summary dict with aggregate metrics.
        """
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Eval dataset not found: {dataset_path}\n"
                "Run  python main.py --eval-generate  first."
            )

        with open(dataset_path, encoding="utf-8") as f:
            dataset = json.load(f)

        items = dataset.get("items", [])
        if not items:
            raise ValueError("Eval dataset is empty — no items to score.")

        mode = "retrieve-only" if retrieve_only else "retrieve+generate"
        print(f"Scoring {len(items)} questions  (top_k={top_k}, mode={mode}) ...")

        per_question = []
        for item in tqdm(items, unit="q"):
            question      = item["question"]
            relevant      = set(item["relevant_chunk_ids"])

            if retrieve_only:
                retrieved_chunks = self._pipeline.retrieve(question, top_k=top_k)
                generated_answer = ""
            else:
                # Each eval question is independent — clear conversation history so
                # answer N is not conditioned on the previous questions' turns.
                self._pipeline.chat_engine.reset()
                generated_answer, retrieved_chunks = self._pipeline.query(question, top_k=top_k)

            retrieved_ids = [c.chunk_id for c in retrieved_chunks]

            row = {
                "question_id":        item["question_id"],
                "question":           question,
                "difficulty":         item.get("difficulty", ""),
                "source_paper_ids":   item.get("source_paper_ids", []),
                "relevant_chunk_ids": list(relevant),
                "retrieved_chunk_ids": retrieved_ids,
                "expected_answer":    item.get("answer", ""),
                "generated_answer":   generated_answer,
                "reciprocal_rank":    _reciprocal_rank(retrieved_ids, relevant),
            }
            for k in RECALL_KS:
                row[f"recall@{k}"] = _recall_at_k(retrieved_ids, relevant, k)

            per_question.append(row)

        # Aggregate
        n = len(per_question)
        aggregate = {"mrr": sum(r["reciprocal_rank"] for r in per_question) / n}
        for k in RECALL_KS:
            aggregate[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in per_question) / n

        # Break down aggregate by difficulty
        by_difficulty: dict = {}
        for diff in ("easy", "medium", "hard"):
            subset = [r for r in per_question if r["difficulty"] == diff]
            if subset:
                nd = len(subset)
                by_difficulty[diff] = {
                    "n": nd,
                    "mrr": sum(r["reciprocal_rank"] for r in subset) / nd,
                    **{f"recall@{k}": sum(r[f"recall@{k}"] for r in subset) / nd for k in RECALL_KS},
                }

        results = {
            "evaluated_at":    datetime.now(timezone.utc).isoformat(),
            "dataset":         str(dataset_path),
            "total_questions": n,
            "top_k":           top_k,
            "aggregate":       aggregate,
            "by_difficulty":   by_difficulty,
            "per_question":    per_question,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        summary = {
            "total_questions": n,
            "output_path":     str(output_path),
            **aggregate,
        }

        self._print_summary(aggregate, by_difficulty, n)
        if not retrieve_only:
            self._print_qa_comparison(per_question)
        return summary

    @staticmethod
    def _print_qa_comparison(per_question: list) -> None:
        width = 80
        print(f"\n{'─'*width}")
        print("  Q&A Comparison (expected vs generated)")
        print(f"{'─'*width}")
        for i, row in enumerate(per_question, 1):
            print(f"\n[{i}/{len(per_question)}]  difficulty={row['difficulty']}  MRR={row['reciprocal_rank']:.3f}")
            print(f"Q: {row['question']}")
            print(f"\nExpected:\n{row['expected_answer']}")
            print(f"\nGenerated:\n{row['generated_answer']}")
            print(f"\n{'─'*width}")

    @staticmethod
    def _print_summary(aggregate: dict, by_difficulty: dict, n: int) -> None:
        print(f"\n{'─'*44}")
        print(f"  Retrieval Evaluation  ({n} questions)")
        print(f"{'─'*44}")
        print(f"  {'Metric':<14} {'All':>8}", end="")
        for diff in ("easy", "medium", "hard"):
            if diff in by_difficulty:
                nd = by_difficulty[diff]["n"]
                print(f"  {diff.capitalize():>8}(n={nd})", end="")
        print()
        print(f"{'─'*44}")
        for k in RECALL_KS:
            key = f"recall@{k}"
            print(f"  {key:<14} {aggregate[key]:>8.3f}", end="")
            for diff in ("easy", "medium", "hard"):
                if diff in by_difficulty:
                    print(f"  {by_difficulty[diff][key]:>14.3f}", end="")
            print()
        print(f"  {'MRR':<14} {aggregate['mrr']:>8.3f}", end="")
        for diff in ("easy", "medium", "hard"):
            if diff in by_difficulty:
                print(f"  {by_difficulty[diff]['mrr']:>14.3f}", end="")
        print()
        print(f"{'─'*44}\n")
