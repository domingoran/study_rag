"""
Chat engine: builds prompts from retrieved chunks and drives the LLM.

Design:
  • Short in-session history (last N turns) to give the model memory of
    the conversation without flooding the context with full retrieved docs.
  • Retrieved chunks are injected fresh on every turn (they vary by query).
  • Citations follow the format:  [paper_id, Section: …, Page: …]
"""
from __future__ import annotations

import logging
from typing import List, Tuple

from core.schemas import Chunk
from llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert research assistant helping users explore academic papers.

Rules:
1. Base your answer ONLY on the provided context excerpts. Do NOT use outside
   knowledge. If the context lacks enough information, say so clearly — do NOT
   invent facts.
2. Be complete. Break the question into its parts and answer each one. Include
   every relevant fact, mechanism, condition, and qualifier the context provides
   — not just the first or most obvious. If several excerpts bear on the
   question, draw on all of them rather than stopping at one.
3. Do not pad. Completeness means covering everything in the context that answers
   the question — never filler, repetition, or outside knowledge.
4. After every claim that comes from a specific source, add a citation in the
   form: [paper_id, Section: <section>, Page: <page>]
5. Structure the answer clearly — short paragraphs, or bullet points for
   multi-part answers.
"""

# Maximum number of user/assistant turns kept in history
_MAX_HISTORY_TURNS = 6


class ChatEngine:
    """
    Manages chat state (history) and formats prompts for the Ollama LLM.

    Usage::

        engine = ChatEngine(ollama_client)
        answer = engine.answer("What is multi-head attention?", chunks)
    """

    def __init__(self, ollama_client: OllamaClient) -> None:
        self._client = ollama_client
        self._history: List[dict] = []   # list of {"role": …, "content": …}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def answer(self, query: str, chunks: List[Chunk]) -> str:
        """
        Generate an answer to *query* grounded in *chunks*.

        Args:
            query:  User's natural-language question.
            chunks: Retrieved Chunk objects (already ranked).

        Returns:
            LLM-generated answer string with inline citations.
        """
        context = self._build_context(chunks)

        user_content = (
            f"Context excerpts from research papers:\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"Question: {query}"
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *self._history,
            {"role": "user", "content": user_content},
        ]

        logger.debug(
            "Calling LLM with %d context chunk(s) + %d history turn(s)",
            len(chunks),
            len(self._history) // 2,
        )

        response = self._client.chat(messages)

        # Store only the bare question (not the full context) in history
        # to avoid the history growing too large
        self._history.append({"role": "user",      "content": query})
        self._history.append({"role": "assistant",  "content": response})

        # Trim history to last N turns
        max_msgs = _MAX_HISTORY_TURNS * 2   # each turn = 2 messages
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

        return response

    def reset(self) -> None:
        """Clear conversation history."""
        self._history = []
        logger.info("Chat history cleared.")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_context(chunks: List[Chunk]) -> str:
        """
        Format retrieved chunks into a numbered context block.

        Each entry looks like:
            [1] [paper_id, Section: Introduction, Page: 3]
            <content text>
        """
        parts: List[str] = []
        for i, chunk in enumerate(chunks, start=1):
            citation = (
                f"[{chunk.paper_id}, "
                f"Section: {chunk.section or 'Unknown'}, "
                f"Page: {chunk.metadata.page}]"
            )
            parts.append(f"[{i}] {citation}\n{chunk.content}")
        return "\n\n".join(parts)
