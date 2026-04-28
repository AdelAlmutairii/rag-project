from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

NOT_FOUND_REPLY = (
    "I cannot find this information in the provided documents. "
    "Try rephrasing your question or uploading the relevant document."
)

_SYSTEM_PROMPT = """\
You are a precise document assistant. Answer questions using ONLY the information \
from the sources provided below.

Rules you must follow:
1. Cite sources inline using [Source N] notation.
2. If the answer is not present in the sources, respond with exactly: NOT_FOUND
3. Never use outside knowledge or fabricate information.
4. After your answer add a "Citations" section with 1-3 bullet points, each quoting \
   an exact phrase from the sources and referencing the corresponding [Source N].
5. Be concise and direct."""


def format_context(docs_with_scores: list[tuple[Document, float]], max_chars: int = 1500) -> str:
    parts: list[str] = []
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        source = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page", "?")
        text = (doc.page_content or "").strip().replace("\x00", "")
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + " …"
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
    """Build messages including prior conversation turns for multi-turn QA."""
    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # Inject history (cap at last 6 turns to stay within context)
    for turn in chat_history[-6:]:
        messages.append(turn)

    messages.append(
        {
            "role": "user",
            "content": f"Sources:\n{context}\n\nQuestion: {question}",
        }
    )
    return messages
