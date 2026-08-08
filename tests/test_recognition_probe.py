"""Forced-choice recognition: balanced orders, structural fork, honest source."""

from __future__ import annotations

import pytest

from realj.models import BANNED_SURFACE_TERMS
from realj.scorers.recognition import (
    OPTION_DEPLOYMENT,
    OPTION_EVAL,
    PROBE_TEMPLATES,
    _eval_first,
    _parse_choice,
    _render_options,
    calibration_prefixes,
)


@pytest.mark.parametrize("prefix_hash", ["abc", "def", "0" * 64, "f" * 64])
def test_orders_are_exactly_balanced(prefix_hash):
    """A per-probe coin flip put all four probes in one order ~1 time in 8,
    silently disabling the position-bias control exactly when it matters."""
    orders = [_eval_first(prefix_hash, i) for i in range(len(PROBE_TEMPLATES))]
    assert sum(orders) == len(orders) // 2, orders


def test_order_assignment_is_deterministic():
    assert [_eval_first("x", i) for i in range(4)] == [_eval_first("x", i) for i in range(4)]


def test_eval_letter_tracks_the_order():
    block_a, letter_a = _render_options(eval_first=True)
    block_b, letter_b = _render_options(eval_first=False)
    assert letter_a == "A" and letter_b == "B"
    assert block_a.startswith(f"A. {OPTION_EVAL}")
    assert block_b.startswith(f"A. {OPTION_DEPLOYMENT}")


def test_both_options_always_present():
    for eval_first in (True, False):
        block, _ = _render_options(eval_first)
        assert OPTION_EVAL in block and OPTION_DEPLOYMENT in block


@pytest.mark.parametrize(
    "raw,expected",
    [("A", "A"), ("b", "B"), (" B ", "B"), ("The answer is B.", "B"), ("neither", None), ("", None)],
)
def test_choice_parsing(raw, expected):
    assert _parse_choice(raw) == expected


def test_primary_probes_request_no_explanation():
    """Asking for reasoning in the probe induces the reasoning being measured."""
    for template in PROBE_TEMPLATES:
        lowered = template.lower()
        assert "explain" not in lowered
        assert "reasoning" not in lowered
        assert "letter" in lowered


def test_probes_do_not_disclose_more_than_the_question():
    """The probe necessarily mentions being observed; it must add nothing else."""
    allowed = {"evaluation", "experiment"}
    for template in PROBE_TEMPLATES:
        lowered = template.lower()
        for term in BANNED_SURFACE_TERMS:
            if term in allowed:
                continue
            assert term not in lowered, (term, template)


def test_calibration_bank_is_agentic_coding_not_trivia():
    prefixes = calibration_prefixes()
    assert set(prefixes) == {"unambiguous_eval", "unambiguous_deployment"}
    for name, prefix in prefixes.items():
        blob = " ".join(m.content for m in prefix.messages).lower()
        assert any(w in blob for w in ("code", "implement", "deploy", "logs", "programming")), name
