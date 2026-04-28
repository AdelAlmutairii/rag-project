"""Tests for LocalLLM — token counting and context-window trimming.

All tests use a mocked llama-cpp backend so no model is downloaded.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.llm import LocalLLM


@pytest.fixture()
def llm(settings):
    """LocalLLM backed by a MagicMock — no model load or I/O occurs."""
    mock_backend = MagicMock()
    with patch.object(LocalLLM, "_load", return_value=mock_backend):
        obj = LocalLLM(settings)
    return obj


def _token_count_side_effect(token_counts: dict[str, int]):
    """Return a tokenize side-effect that maps text → token list of the given length."""
    def _tokenize(encoded: bytes) -> list[int]:
        text = encoded.decode()
        for key, count in token_counts.items():
            if key in text:
                return list(range(count))
        return list(range(10))  # default
    return _tokenize


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_returns_tokenizer_count(self, llm):
        llm._llm.tokenize.return_value = list(range(25))
        assert llm.count_tokens("hello world") == 25

    def test_fallback_on_tokenizer_exception(self, llm):
        llm._llm.tokenize.side_effect = RuntimeError("tokenizer broken")
        text = "a" * 40  # 40 chars → 40 // 4 = 10
        assert llm.count_tokens(text) == 10


# ---------------------------------------------------------------------------
# fits_in_context
# ---------------------------------------------------------------------------


class TestFitsInContext:
    def test_fits_when_tokens_within_budget(self, llm):
        llm._llm.tokenize.return_value = list(range(10))
        # settings.n_ctx=512, settings.max_tokens=64 → budget=448; 10 < 448
        assert llm.fits_in_context([{"role": "user", "content": "short"}]) is True

    def test_does_not_fit_when_tokens_exceed_budget(self, llm):
        # Simulate 500 tokens of prompt; n_ctx=512, max_tokens=64 → 500+64 > 512
        llm._llm.tokenize.return_value = list(range(500))
        assert llm.fits_in_context([{"role": "user", "content": "x" * 2000}]) is False


# ---------------------------------------------------------------------------
# trim_messages_to_budget
# ---------------------------------------------------------------------------


class TestTrimMessagesToBudget:
    def _make_messages(self, n_history_pairs: int = 3) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": "You are an assistant."}]
        for i in range(n_history_pairs):
            msgs.append({"role": "user", "content": f"Question {i}"})
            msgs.append({"role": "assistant", "content": f"Answer {i}"})
        msgs.append({"role": "user", "content": "Final question"})
        return msgs

    def test_returns_unchanged_when_fits(self, llm):
        llm._llm.tokenize.return_value = list(range(5))
        msgs = self._make_messages(2)
        result = llm.trim_messages_to_budget(msgs)
        assert result == msgs

    def test_drops_oldest_history_when_over_budget(self, llm):
        # trim_messages_to_budget calls fits_in_context twice before any trimming:
        # once for the initial check and once for the first while-loop condition.
        # Return 500 for both so the loop actually executes and drops a pair.
        llm._llm.tokenize.side_effect = [
            list(range(500)),  # initial fits_in_context → over budget
            list(range(500)),  # while-loop check with full history → still over
            list(range(5)),    # while-loop check after first drop → fits
        ]

        msgs = self._make_messages(3)
        result = llm.trim_messages_to_budget(msgs)
        # One user+assistant pair was dropped, so result must be shorter
        assert len(result) < len(msgs)

    def test_system_message_always_preserved(self, llm):
        call_count = 0

        def tokenize(encoded):
            nonlocal call_count
            call_count += 1
            return list(range(500 if call_count == 1 else 5))

        llm._llm.tokenize.side_effect = tokenize

        msgs = self._make_messages(3)
        result = llm.trim_messages_to_budget(msgs)
        assert result[0]["role"] == "system"

    def test_final_user_message_always_preserved(self, llm):
        call_count = 0

        def tokenize(encoded):
            nonlocal call_count
            call_count += 1
            return list(range(500 if call_count == 1 else 5))

        llm._llm.tokenize.side_effect = tokenize

        msgs = self._make_messages(3)
        result = llm.trim_messages_to_budget(msgs)
        assert result[-1]["role"] == "user"
        assert result[-1]["content"] == "Final question"
