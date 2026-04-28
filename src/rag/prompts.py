from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

NOT_FOUND_REPLY = (
    "I cannot find this information in the provided documents. "
    "Try rephrasing your question or uploading the relevant document."
)

# Separated into atomic, non-conflicting rules.  Previously Rules 2 and 4
# conflicted: "output exactly NOT_FOUND" vs "always add a Citations section".
_SYSTEM_PROMPT = """\
You are a precise document assistant. Answer questions using ONLY the information \
from the sources provided below.

Rules:
1. Base every claim on the provided sources. Never use outside knowledge.
2. If the sources do not contain enough information to answer, output exactly: NOT_FOUND
3. Otherwise, cite sources inline using [Source N] notation.
4. After your answer add a "Citations" section with 1-3 bullet points, each quoting \
   an exact short phrase from the sources and referencing the corresponding [Source N].
5. Be concise and direct. Do not pad your answer."""

# Used by pipeline.contextualize_query to rewrite follow-up questions into
# standalone questions before retrieval, so the vector search is not misled
# by pronouns ("it", "that", "them") that only make sense with history.
_CONTEXTUALIZE_PROMPT = """\
Given the conversation history and the user's latest question, rewrite the question \
as a fully self-contained question that can be understood without the conversation history.
- If the question is already standalone, return it unchanged.
- Do NOT answer the question — only rewrite it.
- Output only the rewritten question, nothing else."""


def format_context(
    docs_with_scores: list[tuple[Document, float]],
    total_budget: int = 6000,
) -> str:
    """Format retrieved chunks into a numbered source block for the prompt.

    *total_budget* is the maximum total character count for all sources combined.
    Characters are distributed evenly across chunks so no single chunk dominates
    the context window while short chunks are not padded.
    """
    if not docs_with_scores:
        return ""

    per_chunk = max(200, total_budget // len(docs_with_scores))
    parts: list[str] = []
    for i, (doc, _score) in enumerate(docs_with_scores, 1):
        source = (
            doc.metadata.get("source_name")
            or Path(doc.metadata.get("source", "unknown")).name
        )
        page = doc.metadata.get("page", "?")
        text = (doc.page_content or "").strip().replace("\x00", "")
        if len(text) > per_chunk:
            text = text[:per_chunk].rsplit(" ", 1)[0] + " …"
        parts.append(f"[Source {i}] {source} (page {page})\n\"\"\"{text}\"\"\"")
    return "\n\n".join(parts)


def build_rag_messages(context: str, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Sources:\n{context}\n\nQuestion: {question}",
        },
    ]


def build_history_messages(
    context: str,
    question: str,
    chat_history: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Build messages including prior conversation turns for multi-turn QA.

    History entries should be the full injected messages (including sources),
    not just the bare questions — otherwise the LLM references answers from
    prior turns without access to the context that grounded them.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # Inject history (cap at last 6 turns to limit token growth)
    for turn in chat_history[-6:]:
        messages.append(turn)

    messages.append(
        {
            "role": "user",
            "content": f"Sources:\n{context}\n\nQuestion: {question}",
        }
    )
    return messages


def build_contextualize_messages(
    question: str,
    chat_history: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Build messages for the query-contextualization LLM call."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _CONTEXTUALIZE_PROMPT}
    ]
    for turn in chat_history[-4:]:
        messages.append(turn)
    messages.append({"role": "user", "content": question})
    return messages
