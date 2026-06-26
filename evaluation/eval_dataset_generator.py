"""
Evaluation dataset generator — Phase 4a.

Pipeline:
  1. Fetch all embeddings from Milvus (chunk_id, paper_id, embedding)
  2. K-Means clustering with silhouette-score sweep to pick K
  3. Stratified seed sampling proportional to each paper's chunk share
  4. Nearest-neighbour group assembly within each cluster (in-memory cosine)
  5. LLM Q&A generation: EVAL_QUESTIONS_PER_GROUP pairs per group (easy/medium/hard)
  6. Write data/eval_dataset.json
"""
from __future__ import annotations

import json
import logging
import math
import random
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import config
from llm.ollama_client import OllamaClient
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class EvalDatasetGenerator:
    def __init__(self, vector_store: VectorStore, ollama_client: OllamaClient) -> None:
        self._vs = vector_store
        self._llm = ollama_client

    def manual_generate(self) -> None:
        """Run clustering + seed selection, then print chunks for manual Q&A generation."""

        # Step 1 — fetch embeddings
        print("Fetching embeddings from Milvus...")
        records = self._vs.fetch_all_embeddings()
        N = len(records)
        min_required = config.EVAL_CLUSTER_K_MIN * config.EVAL_CHUNKS_PER_GROUP
        if N < min_required:
            raise ValueError(
                f"Only {N} chunks in the store; need at least {min_required}. "
                "Ingest more papers first."
            )

        chunk_ids = [r["chunk_id"] for r in records]
        paper_ids = [r["paper_id"] for r in records]
        embeddings = np.array([r["embedding"] for r in records], dtype=np.float32)

        # Step 2 — clustering
        cluster_k, cluster_labels = self._cluster(embeddings, N)

        # Step 3 — stratified seed sampling
        n_seeds_base = math.ceil(config.EVAL_NUM_QUESTIONS / config.EVAL_QUESTIONS_PER_GROUP)
        n_seeds = min(n_seeds_base * 2, N)
        seed_indices = self._stratified_sample(paper_ids, n_seeds)

        # Build all groups
        groups: List[List[str]] = []
        for seed_idx in seed_indices:
            group_ids = self._build_group(seed_idx, chunk_ids, cluster_labels, embeddings)
            groups.append(group_ids)

        # Print the system prompt template first
        n_q = config.EVAL_QUESTIONS_PER_GROUP
        if n_q >= 3:
            difficulty_instruction = (
                '- Vary difficulty: include "easy" (single-passage factual recall), '
                '"medium" (requires connecting two passages), and '
                '"hard" (requires inference, comparison, or critical thinking).'
            )
        else:
            difficulty_instruction = (
                '- Assign a difficulty label: "easy" (single-passage factual recall), '
                '"medium" (requires connecting two passages), or '
                '"hard" (requires inference, comparison, or critical thinking).'
            )

        system_prompt = (
            "You are an expert researcher creating evaluation questions for a RAG system.\n\n"
            "Below are {N} text passages from academic papers. "
            "Each passage is identified by its chunk_id.\n\n"
            "{PASSAGES}\n\n"
            f"Generate exactly {n_q} question-answer pair{'s' if n_q != 1 else ''} based ONLY on the information in these passages.\n"
            f"IMPORTANT: Return exactly {n_q} item{'s' if n_q != 1 else ''} in the JSON array — no more, no fewer.\n\n"
            "Requirements:\n"
            f"{difficulty_instruction}\n"
            "- Ground every answer strictly in the passages — do not hallucinate.\n"
            "- Do NOT reference passages by their bracket number (e.g. [1], [3]). "
            "Questions and answers must be self-contained and understandable without the passages.\n"
            "- For each Q&A pair, list the chunk_id(s) from the passages above that are needed to answer. "
            "Use the EXACT chunk_id UUIDs shown after 'chunk_id:' in each passage header — "
            "do NOT use the passage number in brackets.\n\n"
            "Return a JSON array only — no markdown fences, no extra text:\n"
            "[\n"
            "  {\n"
            '    "question": "...",\n'
            '    "answer": "...",\n'
            '    "difficulty": "easy|medium|hard",\n'
            '    "relevant_chunk_ids": ["copy-exact-uuid-here", ...]\n'
            "  }\n"
            "]"
        )

        print("\n" + "=" * 70)
        print("SYSTEM PROMPT (use this with each group below)")
        print("Replace {N} with the number of passages and {PASSAGES} with the chunk block.")
        print("=" * 70)
        print(system_prompt)
        print("=" * 70)

        # Print each group one at a time, pausing for Enter between groups
        print(f"\n{len(groups)} groups total, {config.EVAL_CHUNKS_PER_GROUP} chunks each.")
        print(f"Target: {config.EVAL_NUM_QUESTIONS} Q&A pairs "
              f"({config.EVAL_QUESTIONS_PER_GROUP} per group).")
        print("Press Enter to view each group. Type 'q' to stop.\n")

        for g_idx, group_ids in enumerate(groups, 1):
            if g_idx > 1:
                try:
                    resp = input(f"--- Press Enter for group {g_idx}/{len(groups)} (q to quit) ---")
                except (EOFError, KeyboardInterrupt):
                    print("\nStopped.")
                    return
                if resp.strip().lower() == "q":
                    print("Stopped.")
                    return

            chunks = self._vs.fetch_by_ids(group_ids)
            if not chunks:
                continue

            print(f"\n{'━' * 70}")
            print(f"GROUP {g_idx}/{len(groups)}")
            print(f"{'━' * 70}")

            passages = []
            for i, c in enumerate(chunks, 1):
                block = (
                    f"[{i}] chunk_id: {c.chunk_id}\n"
                    f"Paper: {c.title} ({c.year})\n"
                    f"Section: {c.section}\n"
                    f"---\n{c.content}"
                )
                passages.append(block)

            print("\n\n".join(passages))
            print(f"\n{'─' * 70}")

    def generate(self, output_path: Path = config.EVAL_DATASET_PATH) -> dict:
        """Run the full pipeline, write JSON, and return a summary dict."""

        # Step 1 — fetch embeddings
        print("Fetching embeddings from Milvus...")
        records = self._vs.fetch_all_embeddings()
        N = len(records)
        min_required = config.EVAL_CLUSTER_K_MIN * config.EVAL_CHUNKS_PER_GROUP
        if N < min_required:
            raise ValueError(
                f"Only {N} chunks in the store; need at least {min_required} "
                f"(EVAL_CLUSTER_K_MIN={config.EVAL_CLUSTER_K_MIN} × "
                f"EVAL_CHUNKS_PER_GROUP={config.EVAL_CHUNKS_PER_GROUP}). "
                "Ingest more papers first."
            )

        chunk_ids = [r["chunk_id"] for r in records]
        paper_ids = [r["paper_id"] for r in records]
        embeddings = np.array([r["embedding"] for r in records], dtype=np.float32)

        # Step 2 — clustering
        cluster_k, cluster_labels = self._cluster(embeddings, N)
        counts = np.bincount(cluster_labels)
        print(f"K={cluster_k}  cluster sizes: min={counts.min()}, max={counts.max()}")

        # Step 3 — stratified seed sampling
        # Oversample seeds (2×) to absorb items dropped by validation filters
        n_seeds_base = math.ceil(config.EVAL_NUM_QUESTIONS / config.EVAL_QUESTIONS_PER_GROUP)
        n_seeds = min(n_seeds_base * 2, N)
        seed_indices = self._stratified_sample(paper_ids, n_seeds)
        print(f"Sampled {len(seed_indices)} seed chunks (target: {config.EVAL_NUM_QUESTIONS} questions).")

        # Steps 4–6 — group assembly → content fetch → Q&A generation
        items: List[dict] = []
        for seed_idx in seed_indices:
            if len(items) >= config.EVAL_NUM_QUESTIONS:
                break

            group_ids = self._build_group(seed_idx, chunk_ids, cluster_labels, embeddings)
            chunks = self._vs.fetch_by_ids(group_ids)
            if not chunks:
                continue

            generated: List[dict] = []
            for _attempt in range(3):
                generated = self._generate_qa(chunks)
                if generated:
                    break
                logger.info("Retry %d/2 for seed %d", _attempt + 1, seed_idx)
            print(
                f"  Seed {seed_idx} (cluster {int(cluster_labels[seed_idx])}): "
                f"LLM returned {len(generated)} Q&A pair(s) — "
                f"total so far: {len(items) + len(generated)}/{config.EVAL_NUM_QUESTIONS}"
            )
            cluster_id = int(cluster_labels[seed_idx])
            source_paper_ids = list({c.paper_id for c in chunks})

            for qa in generated:
                if len(items) >= config.EVAL_NUM_QUESTIONS:
                    break
                items.append({
                    "question_id": str(uuid.uuid4()),
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "difficulty": qa["difficulty"],
                    "relevant_chunk_ids": qa["relevant_chunk_ids"],
                    "source_paper_ids": source_paper_ids,
                    "cluster_id": cluster_id,
                })

        dataset = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": config.EVAL_LLM_MODEL,
            "total_questions": len(items),
            "cluster_k": cluster_k,
            "items": items,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        summary = {
            "total_questions": len(items),
            "cluster_k": cluster_k,
            "output_path": str(output_path),
        }
        print(f"Done. {len(items)} Q&A pairs written to {output_path}")
        return summary

    # ------------------------------------------------------------------ #
    # Step 2 — K-Means clustering with silhouette sweep                   #
    # ------------------------------------------------------------------ #

    def _cluster(self, embeddings: np.ndarray, N: int) -> Tuple[int, np.ndarray]:
        k_min = config.EVAL_CLUSTER_K_MIN
        # Never ask for more clusters than we can fill with full groups
        k_max = min(config.EVAL_CLUSTER_K_MAX, N // config.EVAL_CHUNKS_PER_GROUP)
        k_max = max(k_min, k_max)

        # Subsample for the sweep when the corpus is large (silhouette is O(N²))
        if N > 5000:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(N, size=5000, replace=False)
            sample = embeddings[sample_idx]
        else:
            sample = embeddings

        print(f"Sweeping K={k_min}..{k_max} on {len(sample)} vectors...")
        best_k, best_score = k_min, -1.0
        for k in range(k_min, k_max + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(sample)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(
                sample, labels,
                sample_size=min(2000, len(sample)),
                random_state=42,
            )
            if score > best_score:
                best_score, best_k = score, k

        print(f"Best K={best_k} (silhouette={best_score:.4f}). Fitting on all {N} vectors...")
        final_km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        labels = final_km.fit_predict(embeddings)
        return best_k, labels

    # ------------------------------------------------------------------ #
    # Step 3 — stratified seed sampling                                   #
    # ------------------------------------------------------------------ #

    def _stratified_sample(self, paper_ids: List[str], n_seeds: int) -> List[int]:
        # Group indices by paper
        paper_indices: Dict[str, List[int]] = {}
        for idx, pid in enumerate(paper_ids):
            paper_indices.setdefault(pid, []).append(idx)

        papers = list(paper_indices.keys())
        counts = [len(paper_indices[p]) for p in papers]
        total = sum(counts)

        # Proportional slot allocation
        raw = [n_seeds * c / total for c in counts]
        slots = [max(1, round(r)) for r in raw]

        # Fix rounding so sum(slots) == n_seeds
        diff = n_seeds - sum(slots)
        fracs = sorted(
            range(len(raw)),
            key=lambda i: raw[i] - int(raw[i]),
            reverse=(diff > 0),
        )
        for i in fracs[:abs(diff)]:
            slots[i] += 1 if diff > 0 else -1
            slots[i] = max(1, slots[i])

        rng = random.Random(42)
        seed_indices: List[int] = []
        for paper, slot in zip(papers, slots):
            pool = paper_indices[paper]
            # Sample without replacement up to pool size, then fill with replacement
            chosen = rng.sample(pool, min(slot, len(pool)))
            if slot > len(pool):
                chosen += rng.choices(pool, k=slot - len(pool))
            seed_indices.extend(chosen)

        rng.shuffle(seed_indices)
        return seed_indices[:n_seeds]

    # ------------------------------------------------------------------ #
    # Step 4 — nearest-neighbour group assembly                           #
    # ------------------------------------------------------------------ #

    def _build_group(
        self,
        seed_idx: int,
        chunk_ids: List[str],
        cluster_labels: np.ndarray,
        embeddings: np.ndarray,
    ) -> List[str]:
        seed_cluster = cluster_labels[seed_idx]
        cluster_indices = np.where(cluster_labels == seed_cluster)[0]

        seed_vec = embeddings[seed_idx]
        # Cosine similarity: embeddings are already L2-normalised by the embedder
        sims = embeddings[cluster_indices] @ seed_vec

        # Sort descending, skip the seed itself
        order = np.argsort(sims)[::-1]
        n_neighbours = config.EVAL_CHUNKS_PER_GROUP - 1
        neighbours: List[int] = []
        for pos in order:
            global_idx = int(cluster_indices[pos])
            if global_idx == seed_idx:
                continue
            neighbours.append(global_idx)
            if len(neighbours) >= n_neighbours:
                break

        group_global = [seed_idx] + neighbours
        seen: set = set()
        result: List[str] = []
        for gi in group_global:
            cid = chunk_ids[gi]
            if cid not in seen:
                seen.add(cid)
                result.append(cid)
        return result

    # ------------------------------------------------------------------ #
    # Step 5 — LLM Q&A generation                                         #
    # ------------------------------------------------------------------ #

    def _generate_qa(self, chunks) -> List[dict]:  # noqa: C901
        passages = "\n\n".join(
            f"[{i}] chunk_id: {c.chunk_id}\n"
            f"Paper: {c.title} ({c.year})\n"
            f"Section: {c.section}\n"
            f"---\n{c.content}"
            for i, c in enumerate(chunks, 1)
        )
        n = config.EVAL_QUESTIONS_PER_GROUP
        if n >= 3:
            difficulty_instruction = (
                '- Vary difficulty: include "easy" (single-passage factual recall), '
                '"medium" (requires connecting two passages), and '
                '"hard" (requires inference, comparison, or critical thinking).\n'
            )
        else:
            difficulty_instruction = (
                '- Assign a difficulty label: "easy" (single-passage factual recall), '
                '"medium" (requires connecting two passages), or '
                '"hard" (requires inference, comparison, or critical thinking).\n'
            )
        prompt = (
            f"You are an expert researcher creating evaluation questions for a RAG system.\n\n"
            f"Below are {len(chunks)} text passages from academic papers. "
            f"Each passage is identified by its chunk_id.\n\n"
            f"{passages}\n\n"
            f"Generate exactly {n} question-answer pair{'s' if n != 1 else ''} based ONLY on the information in these passages.\n"
            f"IMPORTANT: Return exactly {n} item{'s' if n != 1 else ''} in the JSON array — no more, no fewer.\n\n"
            "Requirements:\n"
            f"{difficulty_instruction}"
            "- Ground every answer strictly in the passages — do not hallucinate.\n"
            "- Do NOT reference passages by their bracket number (e.g. [1], [3]). "
            "Questions and answers must be self-contained and understandable without the passages.\n"
            "- For each Q&A pair, list the chunk_id(s) from the passages above that are needed to answer. "
            "Use the EXACT chunk_id UUIDs shown after 'chunk_id:' in each passage header — "
            "do NOT use the passage number in brackets.\n\n"
            "Return a JSON array only — no markdown fences, no extra text:\n"
            "[\n"
            "  {\n"
            '    "question": "...",\n'
            '    "answer": "...",\n'
            '    "difficulty": "easy|medium|hard",\n'
            '    "relevant_chunk_ids": ["copy-exact-uuid-here", ...]\n'
            "  }\n"
            "]"
        )

        chunk_papers = [f"{c.paper_id}:{c.section}" for c in chunks]
        logger.info(
            "Sending prompt to LLM (%d chars, %d chunks from %s)",
            len(prompt), len(chunks), chunk_papers,
        )

        try:
            t0 = time.perf_counter()
            raw = self._llm.generate(prompt).strip()
            elapsed = time.perf_counter() - t0
            logger.info(
                "LLM responded in %.1fs (%d chars). Preview: %.200s",
                elapsed, len(raw), raw,
            )

            # Strip accidental markdown code fences
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.rsplit("```", 1)[0].strip()

            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError("LLM did not return a JSON array")
            logger.info("Parsed %d item(s) from LLM JSON response", len(parsed))

            known_ids = {c.chunk_id for c in chunks}
            valid = []
            dropped_keys = 0
            dropped_bracket = 0
            dropped_ids = 0
            required_keys = {"question", "answer", "difficulty", "relevant_chunk_ids"}
            for item in parsed:
                if not required_keys.issubset(item):
                    dropped_keys += 1
                    logger.warning("Dropping malformed Q&A item (missing keys): %s", item)
                    continue
                _BRACKET_REF = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
                if _BRACKET_REF.search(item["question"]) or _BRACKET_REF.search(item["answer"]):
                    dropped_bracket += 1
                    logger.warning(
                        "Dropping Q&A item — contains bracket passage references: %s",
                        item["question"][:80],
                    )
                    continue
                clean_ids = [cid for cid in item["relevant_chunk_ids"] if cid in known_ids]
                if not clean_ids:
                    dropped_ids += 1
                    logger.warning(
                        "Dropping Q&A item — no valid chunk_ids (got %s)",
                        item["relevant_chunk_ids"],
                    )
                    continue
                item["relevant_chunk_ids"] = clean_ids
                valid.append(item)

            if dropped_keys or dropped_bracket or dropped_ids:
                logger.info(
                    "Validation: %d valid, %d dropped (missing_keys=%d, bracket_refs=%d, bad_ids=%d)",
                    len(valid), dropped_keys + dropped_bracket + dropped_ids,
                    dropped_keys, dropped_bracket, dropped_ids,
                )
            return valid[:n]

        except json.JSONDecodeError as exc:
            logger.warning("JSON parse failed: %s — raw response: %.500s", exc, raw)
            return []
        except Exception as exc:
            logger.warning("Q&A generation failed: %s", exc)
            return []
