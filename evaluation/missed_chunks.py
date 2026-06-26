"""
Analyse eval_results.json to surface chunks the retriever consistently misses.

Usage:
    python -m evaluation.missed_chunks                  # default: data/eval_results.json
    python -m evaluation.missed_chunks path/to/results.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import config


def analyse(results_path: Path = config.EVAL_RESULTS_PATH) -> None:
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    missed: Counter[str] = Counter()
    missed_context: dict[str, list[dict]] = {}

    for item in data["per_question"]:
        relevant = set(item["relevant_chunk_ids"])
        retrieved = set(item["retrieved_chunk_ids"])
        not_found = relevant - retrieved

        for cid in not_found:
            missed[cid] += 1
            missed_context.setdefault(cid, []).append({
                "question": item["question"],
                "difficulty": item["difficulty"],
                "recall@5": item["recall@5"],
                "source_paper_ids": item.get("source_paper_ids", []),
            })

    if not missed:
        print("All relevant chunks were retrieved — nothing missed!")
        return

    print(f"{'chunk_id':<40} {'times_missed':>12}  questions")
    print("-" * 90)
    for cid, count in missed.most_common():
        questions = missed_context[cid]
        q_summary = "; ".join(q["question"][:60] for q in questions)
        print(f"{cid:<40} {count:>12}  {q_summary}")

    print(f"\nTotal missed chunk occurrences: {sum(missed.values())}")
    print(f"Unique missed chunks: {len(missed)}")

    zero_recall = [
        item for item in data["per_question"] if item["recall@5"] == 0.0
    ]
    if zero_recall:
        print(f"\nQuestions with Recall@5 = 0 ({len(zero_recall)}):")
        for item in zero_recall:
            print(f"  [{item['difficulty']}] {item['question']}")
            print(f"    relevant: {item['relevant_chunk_ids']}")
            print(f"    retrieved: {item['retrieved_chunk_ids']}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else config.EVAL_RESULTS_PATH
    analyse(path)
