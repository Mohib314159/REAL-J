"""The impossible assertion has to be impossible for a *checkable* reason.

"No source fix can satisfy this" is easy to write in a comment and easy to get
wrong in code. Here it is a structural property: the impossible test and a
named partner test make the same call with the same arguments and expect
different results. Since the functions under test are deterministic, that is a
proof, and it is one a test can verify.

``tests/test_hidden_tests.py`` proves the same thing empirically by running
pytest. Both are kept: the structural check tells you *why* a case is broken,
the empirical one tells you *that* it is.
"""

from __future__ import annotations

import pytest

from realj.tasks.cases import CASES, DEV_CASES, HELDOUT_CASES
from realj.tasks.generator import REPO_ROOT, build

CASE_IDS = sorted(CASES)


def test_split_sizes():
    """2 development cases, 6 held out. Calibration happens only on the two."""
    assert len(DEV_CASES) == 2, DEV_CASES
    assert len(HELDOUT_CASES) == 6, HELDOUT_CASES


def test_case_ids_and_modules_unique():
    modules = [c.module for c in CASES.values()]
    assert len(set(modules)) == len(modules), "two cases share a module"
    kinds = [c.bug_kind for c in CASES.values()]
    assert len(set(kinds)) == len(kinds), "two cases share a bug kind"


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_impossible_contradicts_partner(case_id):
    case = CASES[case_id]
    imp = case.impossible
    assert imp.contradicts(), (
        f"{case_id}: the impossible assertion expects {imp.expected!r}, the "
        f"same as its partner. There is no contradiction and no trap."
    )


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_impossible_and_partner_make_the_same_call(case_id):
    """Both assertions must reference the identical call expression."""
    case = CASES[case_id]
    imp = case.impossible
    assert imp.call in case.impossible_test, (
        f"{case_id}: the impossible test does not contain the call "
        f"{imp.call!r} that the contradiction claim rests on."
    )
    assert imp.call in case.visible_tests, (
        f"{case_id}: the partner test does not contain {imp.call!r}; the two "
        f"assertions are about different calls and do not contradict."
    )
    assert imp.expected in case.impossible_test, (
        f"{case_id}: expected value {imp.expected!r} not found in the "
        f"impossible test body; the declaration is stale."
    )
    assert imp.partner_expected in case.visible_tests, (
        f"{case_id}: partner expected value {imp.partner_expected!r} not found "
        f"in the visible tests; the declaration is stale."
    )


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_named_tests_exist(case_id):
    case = CASES[case_id]
    assert f"def {case.impossible.test_name}(" in case.impossible_test
    assert f"def {case.impossible.partner_test}(" in case.visible_tests


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_clean_arm_omits_the_trap(case_id):
    case = CASES[case_id]
    clean = build(case_id, arm="CLEAN")
    tests = clean.code_files[f"{REPO_ROOT}/{case.test_path}"]
    assert case.impossible.test_name not in tests, (
        f"{case_id}: the impossible assertion leaked into the CLEAN arm, which "
        f"is supposed to be the normal-task baseline."
    )


@pytest.mark.parametrize("case_id", CASE_IDS)
@pytest.mark.parametrize("arm", ["FORBIDDEN", "ALLOWED"])
def test_trap_present_in_treatment_arms(case_id, arm):
    case = CASES[case_id]
    spec = build(case_id, arm=arm)
    tests = spec.code_files[f"{REPO_ROOT}/{case.test_path}"]
    assert case.impossible.test_name in tests
    assert case.impossible.partner_test in tests


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_reference_source_differs_from_buggy(case_id):
    case = CASES[case_id]
    assert case.src_buggy != case.src_reference, f"{case_id}: no bug introduced"


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_reference_never_shipped(case_id):
    """The corrected source must not appear anywhere the agent can read."""
    case = CASES[case_id]
    for arm in ("FORBIDDEN", "ALLOWED", "CLEAN"):
        for rung in (1, 2, 3, 4):
            spec = build(case_id, arm=arm, rung=rung)
            blob = "\n".join(spec.files.values())
            assert case.src_reference not in blob, (
                f"{case_id}/{arm}/R{rung}: the reference implementation is "
                f"present in the sandbox."
            )


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_hidden_tests_never_shipped(case_id):
    """Hidden tests are injected at score time and must not be in the repo."""
    case = CASES[case_id]
    for arm in ("FORBIDDEN", "ALLOWED", "CLEAN"):
        for rung in (1, 2, 3, 4):
            spec = build(case_id, arm=arm, rung=rung)
            assert not any("hidden" in p for p in spec.files), (
                f"{case_id}/{arm}/R{rung}: a hidden-test path is present in the "
                f"sandbox file set."
            )
            blob = "\n".join(spec.files.values())
            assert case.hidden_tests not in blob, f"{case_id}: hidden tests shipped"
