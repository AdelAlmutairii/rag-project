"""Tests for prompt builders and context formatting."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.prompts import (
    NOT_FOUND_REPLY,
    build_contextualize_messages,
    build_history_messages,
    build_rag_messages,
    format_context,
)


def _doc(text: str, source: str = "doc.pdf", page: int = 1) -> Document:
    return Document(page_content=text, metadata={"source": source, "page": page})


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------


class TestFormatContext:
    def test_empty_returns_empty_string(self):
        assert format_context([]) == ""

    def test_single_doc_contains_source_and_page(self):
        result = format_context([(_doc("Some content.", source="guide.pdf", page=3), 0.5)])
        assert "guide.pdf" in result
        assert "page 3" in result
        assert "Some content." in result

    def test_source_name_metadata_preferred_over_source_path(self):
        doc = Document(
            page_content="text",
            metadata={"source": "/long/path/to/file.pdf", "source_name": "file.pdf", "page": 1},
        )
        result = format_context([(doc, 0.4)])
        assert "file.pdf" in result
        assert "/long/path/to" not in result

    def test_long_text_is_truncated(self):
        long_text = "word " * 2000
        result = format_context([(_doc(long_text), 0.3)], total_budget=500)
        assert len(result) < len(long_text)
        assert "…" in result

    def test_multiple_docs_all_numbered(self):
        docs = [(_doc(f"Content {i}", page=i), 0.5) for i in range(1, 4)]
        result = format_context(docs)
        assert "[Source 1]" in result
        assert "[Source 2]" in result
        assert "[Source 3]" in result

    def test_budget_distributed_across_chunks(self):
        long_text = "x " * 5000
        docs = [(_doc(long_text, source=f"doc{i}.pdf"), 0.5) for i in range(4)]
        result = format_context(docs, total_budget=1000)
        # Each chunk should get roughly 250 chars — no single chunk should dominate
        parts = result.split("[Source ")
        content_lengths = [len(p) for p in parts[1:]]
        assert max(content_lengths) < 600  # no chunk consuming most of the budget

    def test_null_bytes_stripped(self):
        doc = _doc("clean\x00text\x00here")
        result = format_context([(doc, 0.3)])
        assert "\x00" not in result


# ---------------------------------------------------------------------------
# build_rag_messages
# ---------------------------------------------------------------------------


class TestBuildRagMessages:
    def test_returns_two_messages(self):
        msgs = build_rag_messages("ctx", "What is X?")
        assert len(msgs) == 2

    def test_first_message_is_system(self):
        msgs = build_rag_messages("ctx", "Q?")
        assert msgs[0]["role"] == "system"

    def test_user_message_contains_context_and_question(self):
        msgs = build_rag_messages("my context", "What is Y?")
        user_content = msgs[1]["content"]
        assert "my context" in user_content
        assert "What is Y?" in user_content

    def test_system_prompt_forbids_outside_knowledge(self):
        system_content = build_rag_messages("ctx", "Q?")[0]["content"]
        assert "ONLY" in system_content or "only" in system_content


# ---------------------------------------------------------------------------
# build_history_messages
# ---------------------------------------------------------------------------


class TestBuildHistoryMessages:
    def test_includes_system_message(self):
        msgs = build_history_messages("ctx", "Q?", [])
        assert msgs[0]["role"] == "system"

    def test_empty_history_produces_two_messages(self):
        msgs = build_history_messages("ctx", "Q?", [])
        assert len(msgs) == 2  # system + user

    def test_history_turns_appear_between_system_and_user(self):
        history = [
            {"role": "user", "content": "Prior question"},
            {"role": "assistant", "content": "Prior answer"},
        ]
        msgs = build_history_messages("ctx", "Follow-up?", history)
        roles = [m["role"] for m in msgs]
        assert roles[0] == "system"
        assert roles[-1] == "user"
        assert "user" in roles[1:-1]
        assert "assistant" in roles[1:-1]

    def test_caps_at_last_six_history_turns(self):
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"}
            for i in range(20)
        ]
        msgs = build_history_messages("ctx", "Q?", history)
        # system + 6 history turns + 1 user = 8
        assert len(msgs) == 8

    def test_latest_question_is_last(self):
        msgs = build_history_messages("ctx", "Final question?", [])
        assert "Final question?" in msgs[-1]["content"]


# ---------------------------------------------------------------------------
# build_contextualize_messages
# ---------------------------------------------------------------------------


class TestBuildContextualizeMessages:
    def test_system_message_instructs_rewriting(self):
        msgs = build_contextualize_messages("What about it?", [])
        assert msgs[0]["role"] == "system"
        assert "rewrite" in msgs[0]["content"].lower()

    def test_question_is_last_user_message(self):
        msgs = build_contextualize_messages("Follow-up?", [])
        assert msgs[-1]["role"] == "user"
        assert "Follow-up?" in msgs[-1]["content"]

    def test_history_injected(self):
        history = [
            {"role": "user", "content": "What is X?"},
            {"role": "assistant", "content": "X is something."},
        ]
        msgs = build_contextualize_messages("And Y?", history)
        roles = [m["role"] for m in msgs]
        assert "assistant" in roles


# ---------------------------------------------------------------------------
# NOT_FOUND_REPLY
# ---------------------------------------------------------------------------


def test_not_found_reply_is_nonempty_string():
    assert isinstance(NOT_FOUND_REPLY, str)
    assert len(NOT_FOUND_REPLY) > 0
