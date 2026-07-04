"""
LLM-as-a-judge answer-quality evaluation — Phase 4c.

Loads a results file produced by  --eval-score  (which contains, per question,
the expected answer, the LLM-generated answer, and the retrieved chunk_ids) and
asks a *separate* judge model to score each generated answer on five metrics:

    correctness       — factual agreement with the expected/reference answer
    completeness      — coverage of the key facts in the expected answer
    answer_relevance  — does it actually address the question asked
    faithfulness      — is every claim grounded in the retrieved chunks (no hallucination)
    citation_quality  — are the [paper_id, Section, Page] citations present, well-formed,
                        and pointing to chunks that support the claim

Each metric is scored 1-5 with a short rationale. The judge model MUST differ
from the generator model (config.EVAL_JUDGE_MODEL vs config.OLLAMA_CHAT_MODEL)
to avoid self-preference bias.

Faithfulness and citation_quality need the actual retrieved passage text, so the
judge fetches chunk content from Milvus by id before scoring.

Writes data/eval_judge_results.json.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

import config

logger = logging.getLogger(__name__)

# Metric keys the judge must return, in display order.
METRICS = ["correctness", "completeness", "answer_relevance", "faithfulness", "citation_quality"]
SCORE_MIN, SCORE_MAX = 1, 5

_PROMPT_TEMPLATE = """\
You are a strict, impartial evaluator of a Retrieval-Augmented Generation (RAG) system.
A different model produced the GENERATED ANSWER below; your job is only to score it.

You will score five metrics, each an INTEGER from 1 (worst) to 5 (best):

- correctness: Does the generated answer factually agree with the reference answer?
  Reward a semantically correct answer even if it is longer or worded differently than
  the reference. Penalise contradictions or wrong facts. 5 = fully correct, 1 = wrong.
- completeness: Does it cover all the key facts present in the reference answer?
  5 = every key point covered, 1 = misses the main point.
- answer_relevance: Does it directly address the QUESTION (not evasive, off-topic, or padded)?
  5 = focused and on-topic, 1 = does not answer the question.
- faithfulness: Is every claim in the answer supported by the RETRIEVED CONTEXT below?
  Penalise any statement not grounded in the context (hallucination). 5 = fully grounded,
  1 = mostly unsupported. If there is no context, judge on whether claims look fabricated.
- citation_quality: The answer should cite sources as [paper_id, Section: ..., Page: ...].
  Are citations present, well-formed, and do they point to context passages that actually
  support the cited claim? 5 = accurate, well-formed citations; 1 = missing, malformed, or wrong.

QUESTION:
{question}

REFERENCE ANSWER (ground truth):
{expected_answer}

RETRIEVED CONTEXT (the passages the generator was shown):
{context}

GENERATED ANSWER (evaluate this):
{generated_answer}

Return a JSON object ONLY — no markdown, no extra text — with exactly this shape:
{{
  "correctness":      {{"score": <1-5>, "reasoning": "<one sentence>"}},
  "completeness":     {{"score": <1-5>, "reasoning": "<one sentence>"}},
  "answer_relevance": {{"score": <1-5>, "reasoning": "<one sentence>"}},
  "faithfulness":     {{"score": <1-5>, "reasoning": "<one sentence>"}},
  "citation_quality": {{"score": <1-5>, "reasoning": "<one sentence>"}}
}}"""


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by models like qwen3."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def _extract_json_object(text: str) -> Optional[dict]:
    """
    Best-effort parse of a single JSON object from an LLM response.

    Handles thinking tags, ```json fences, and leading/trailing prose by locating
    the outermost balanced { ... } span. Returns None if nothing parses.
    """
    cleaned = _strip_thinking(text).strip()
    # Fast path: whole response is JSON.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Strip markdown code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback: outermost brace span.
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _coerce_score(raw) -> Optional[int]:
    """Clamp a judge score to the [SCORE_MIN, SCORE_MAX] integer range, or None."""
    try:
        val = int(round(float(raw)))
    except (TypeError, ValueError):
        return None
    return max(SCORE_MIN, min(SCORE_MAX, val))


class EvalJudge:
    def __init__(self, vector_store, judge_client) -> None:
        """
        Args:
            vector_store: VectorStore — used to fetch retrieved chunk content by id.
            judge_client: OllamaClient — its .generate(model=..., think=...) is called
                          with config.EVAL_JUDGE_MODEL so the judge differs from the generator.
        """
        self._vector_store = vector_store
        self._judge = judge_client

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def judge(
        self,
        input_path: Path = config.EVAL_JUDGE_INPUT_PATH,
        output_path: Path = config.EVAL_JUDGE_RESULTS_PATH,
    ) -> dict:
        if config.EVAL_JUDGE_MODEL == config.OLLAMA_CHAT_MODEL:
            logger.warning(
                "Judge model (%s) is the same as the generator model — self-preference "
                "bias likely. Set a different EVAL_JUDGE_MODEL in config.py.",
                config.EVAL_JUDGE_MODEL,
            )

        if not input_path.exists():
            raise FileNotFoundError(
                f"Results file not found: {input_path}\n"
                "Run  python main.py --eval-score  first to generate answers."
            )

        with open(input_path, encoding="utf-8") as f:
            results = json.load(f)

        rows = results.get("per_question", [])
        # Only judge rows that actually have a generated answer (retrieve-only runs don't).
        judgeable = [r for r in rows if (r.get("generated_answer") or "").strip()]
        skipped = len(rows) - len(judgeable)
        if not judgeable:
            raise ValueError(
                "No generated answers found in the results file. "
                "Run  --eval-score  (not --eval-retrieve) so answers are produced."
            )
        if skipped:
            logger.warning("Skipping %d question(s) with no generated answer.", skipped)

        print(
            f"Judging {len(judgeable)} answers with '{config.EVAL_JUDGE_MODEL}' "
            f"(generator was '{config.OLLAMA_CHAT_MODEL}') ..."
        )

        per_question = []
        for row in tqdm(judgeable, unit="q"):
            per_question.append(self._judge_one(row))

        aggregate = self._aggregate(per_question)
        by_difficulty = {
            diff: self._aggregate([r for r in per_question if r["difficulty"] == diff])
            for diff in ("easy", "medium", "hard")
            if any(r["difficulty"] == diff for r in per_question)
        }

        out = {
            "judged_at":       datetime.now(timezone.utc).isoformat(),
            "judge_model":     config.EVAL_JUDGE_MODEL,
            "generator_model": config.OLLAMA_CHAT_MODEL,
            "source_results":  str(input_path),
            "total_judged":    len(per_question),
            "skipped":         skipped,
            "scale":           f"{SCORE_MIN}-{SCORE_MAX}",
            "aggregate":       aggregate,
            "by_difficulty":   by_difficulty,
            "per_question":    per_question,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        self._print_summary(aggregate, by_difficulty, len(per_question))
        print(f"  Output: {output_path}\n")

        return {
            "total_judged": len(per_question),
            "output_path":  str(output_path),
            **{m: aggregate[m]["mean"] for m in METRICS},
            "overall":      aggregate["overall"]["mean"],
        }

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _judge_one(self, row: dict) -> dict:
        context = self._build_context(row.get("retrieved_chunk_ids", []))
        prompt = _PROMPT_TEMPLATE.format(
            question=row.get("question", ""),
            expected_answer=row.get("expected_answer", "") or "(none provided)",
            context=context or "(no context available)",
            generated_answer=row.get("generated_answer", ""),
        )

        result = {
            "question_id": row.get("question_id", ""),
            "question":    row.get("question", ""),
            "difficulty":  row.get("difficulty", ""),
            "scores":      {},
            "reasoning":   {},
            "overall":     None,
            "judge_error": None,
        }

        try:
            raw = self._judge.generate(prompt, model=config.EVAL_JUDGE_MODEL, think=False)
        except Exception as exc:  # noqa: BLE001 — never let one bad call kill the run
            logger.warning("Judge call failed for %s: %s", result["question_id"], exc)
            result["judge_error"] = f"generation failed: {exc}"
            return result

        parsed = _extract_json_object(raw)
        if parsed is None:
            logger.warning("Could not parse judge JSON for %s", result["question_id"])
            result["judge_error"] = "unparseable judge response"
            return result

        for metric in METRICS:
            entry = parsed.get(metric)
            if isinstance(entry, dict):
                score = _coerce_score(entry.get("score"))
                reasoning = str(entry.get("reasoning", "")).strip()
            else:
                # Tolerate a bare number for the metric.
                score = _coerce_score(entry)
                reasoning = ""
            if score is not None:
                result["scores"][metric] = score
                result["reasoning"][metric] = reasoning

        got = list(result["scores"].values())
        if got:
            result["overall"] = round(sum(got) / len(got), 3)
        else:
            result["judge_error"] = "no valid metric scores returned"
        return result

    def _build_context(self, chunk_ids: List[str]) -> str:
        """Fetch retrieved chunks and render them with citation metadata for the judge."""
        if not chunk_ids:
            return ""
        try:
            chunks = self._vector_store.fetch_by_ids(chunk_ids)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch chunk content for judging: %s", exc)
            return ""
        # Preserve retrieval order (fetch_by_ids may reorder).
        by_id = {c.chunk_id: c for c in chunks}
        blocks = []
        for i, cid in enumerate(chunk_ids, 1):
            c = by_id.get(cid)
            if c is None:
                continue
            blocks.append(
                f"[{i}] paper_id: {c.paper_id}, Section: {c.section or '—'}, "
                f"Page: {c.metadata.page}\n{c.content}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _aggregate(rows: List[dict]) -> dict:
        """Mean of each metric (and overall) across rows that produced a score."""
        agg: dict = {"n": len(rows)}
        for metric in METRICS:
            vals = [r["scores"][metric] for r in rows if metric in r["scores"]]
            agg[metric] = {
                "mean": round(sum(vals) / len(vals), 3) if vals else None,
                "n":    len(vals),
            }
        overall_vals = [r["overall"] for r in rows if r["overall"] is not None]
        agg["overall"] = {
            "mean": round(sum(overall_vals) / len(overall_vals), 3) if overall_vals else None,
            "n":    len(overall_vals),
        }
        return agg

    @staticmethod
    def _print_summary(aggregate: dict, by_difficulty: dict, n: int) -> None:
        diffs = [d for d in ("easy", "medium", "hard") if d in by_difficulty]
        width = 30 + 16 * (1 + len(diffs))
        print(f"\n{'─' * width}")
        print(f"  LLM-as-a-Judge — Answer Quality  ({n} answers, scale {SCORE_MIN}-{SCORE_MAX})")
        print(f"{'─' * width}")
        header = f"  {'Metric':<20} {'All':>10}"
        for d in diffs:
            col = f"{d.capitalize()}(n={by_difficulty[d]['n']})"
            header += f"  {col:>13}"
        print(header)
        print(f"{'─' * width}")

        def fmt(v):
            return f"{v:.2f}" if isinstance(v, (int, float)) else "—"

        for metric in METRICS + ["overall"]:
            label = metric.replace("_", " ")
            line = f"  {label:<20} {fmt(aggregate[metric]['mean']):>10}"
            for d in diffs:
                line += f"  {fmt(by_difficulty[d][metric]['mean']):>13}"
            print(line)
        print(f"{'─' * width}")
