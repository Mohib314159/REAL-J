"""The rungs must present the same world, not different worlds.

This is the test the whole design rests on. If R5 is more realistic *and*
easier, or has a different trap, or grants a different affordance, then every
number plotted against realism is confounded and the study says nothing.

What is checked, for every case and every arm:

* ``code_hash`` -- source, visible tests, conftest, pyproject -- is byte
  identical across all five rungs. The bug, the trap and the file the agent
  would have to edit are literally the same bytes.
* The substantive request decomposes into the same components. Prose varies;
  requirements do not.
* The tool set is identical. The ability to edit a test file is exactly as
  available at R1 as at R5.
* The impossible assertion and its partner are present in the same arms at
  every rung.
* Context files -- the part that is *allowed* to vary -- never mention the
  module under repair, the bug, or the trap.

Rungs 1 and 2 carry no context files at all, by design. Lengths are recorded,
not equalised: see DESIGN.md S8.
"""

from __future__ import annotations

import pytest

from realj.models import agent_tools
from realj.realism.factor_manifest import (
    FIXED_FACTORS,
    MANIFEST,
    MOVING_FACTORS,
    moving_summary,
)
from realj.realism.factor_manifest import validate as validate_manifest
from realj.realism.rungs import render, substantive_request
from realj.tasks.cases import CASES
from realj.tasks.generator import REPO_ROOT, build

CASE_IDS = sorted(CASES)
ARMS = ("FORBIDDEN", "ALLOWED", "CLEAN")
# R5 needs a recorded trajectory, which cannot exist in a unit test. Its code
# path is exercised by test_r5_refuses_to_synthesise below.
RUNGS = (1, 2, 3, 4)


def _rendered(case_id, arm, rung):
    repo = build(case_id, arm=arm, rung=rung)
    return repo, render(case_id, arm=arm, rung=rung, repo=repo)


@pytest.mark.parametrize("case_id", CASE_IDS)
@pytest.mark.parametrize("arm", ARMS)
def test_code_hash_invariant_across_rungs(case_id, arm):
    hashes = {rung: build(case_id, arm=arm, rung=rung).code_hash for rung in (1, 2, 3, 4, 5)}
    assert len(set(hashes.values())) == 1, (
        f"{case_id}/{arm}: code_hash differs across rungs {hashes}. The rungs "
        f"are presenting different problems, not the same problem differently."
    )


@pytest.mark.parametrize("case_id", CASE_IDS)
@pytest.mark.parametrize("arm", ARMS)
def test_substantive_request_invariant_across_rungs(case_id, arm):
    components = {
        rung: render(
            case_id, arm=arm, rung=rung, repo=build(case_id, arm=arm, rung=rung)
        ).request.components()
        for rung in RUNGS
    }
    assert len(set(components.values())) == 1, (
        f"{case_id}/{arm}: the substantive request differs across rungs: "
        f"{components}"
    )
    assert set(components.values()).pop() == substantive_request(
        CASES[case_id], arm
    ).components()


@pytest.mark.parametrize("case_id", CASE_IDS)
@pytest.mark.parametrize("arm", ARMS)
def test_affordance_identical_across_rungs(case_id, arm):
    """The trap file must be equally reachable and equally writable everywhere."""
    case = CASES[case_id]
    test_path = f"{REPO_ROOT}/{case.test_path}"
    for rung in (1, 2, 3, 4, 5):
        spec = build(case_id, arm=arm, rung=rung)
        assert test_path in spec.code_files
        assert test_path in spec.visible_test_paths
        assert test_path in spec.pristine_hashes

    names = [type(t).__name__ for t in agent_tools()]
    assert names == [type(t).__name__ for t in agent_tools()], "tool set not stable"
    assert len(agent_tools()) == 2, "the affordance set changed; re-check equivalence"


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_arms_differ_only_where_intended(case_id):
    """FORBIDDEN vs CLEAN must differ in the trap and nothing else."""
    case = CASES[case_id]
    forbidden = build(case_id, arm="FORBIDDEN", rung=3)
    clean = build(case_id, arm="CLEAN", rung=3)

    differing = {
        p
        for p in set(forbidden.code_files) | set(clean.code_files)
        if forbidden.code_files.get(p) != clean.code_files.get(p)
    }
    assert differing == {f"{REPO_ROOT}/{case.test_path}"}, (
        f"{case_id}: FORBIDDEN and CLEAN differ in {sorted(differing)}; they "
        f"should differ only in the visible test file."
    )

    f_req = substantive_request(case, "FORBIDDEN")
    c_req = substantive_request(case, "CLEAN")
    assert f_req.components() == c_req.components(), (
        "CLEAN must carry FORBIDDEN's test-modification constraint, so that the "
        "two arms isolate the trap rather than the instruction wording."
    )

    a_req = substantive_request(case, "ALLOWED")
    assert a_req.test_constraint == "permit"
    assert f_req.test_constraint == "forbid"


@pytest.mark.parametrize("case_id", CASE_IDS)
@pytest.mark.parametrize("rung", RUNGS)
def test_context_files_never_mention_the_task(case_id, rung):
    """Realism dressing must not leak the bug, the fix, or the trap.

    Checked against *distinctive* strings only. An earlier version of this
    test compared against ``impossible.expected``, which for some cases is
    ``"None"`` or ``"0"`` -- substrings of almost any Python file, and it
    flagged an unrelated retry helper. A leak test that fires on common
    literals gets disabled within a week, which is worse than not having one.
    """
    case = CASES[case_id]
    spec = build(case_id, arm="FORBIDDEN", rung=rung)
    blob = "\n".join(spec.context_files.values())

    distinctive = {
        "reference implementation": case.src_reference,
        "impossible test name": case.impossible.test_name,
        "partner test name": case.impossible.partner_test,
        "impossible assertion call": case.impossible.call,
        "bug kind label": case.bug_kind,
        "hidden tests": case.hidden_tests,
    }
    for label, needle in distinctive.items():
        assert needle not in blob, (
            f"{case_id}/R{rung}: context files contain the {label} "
            f"({needle[:60]!r}); the surroundings are leaking the answer."
        )


@pytest.mark.parametrize("case_id", CASE_IDS)
@pytest.mark.parametrize("arm", ARMS)
def test_trap_salience_is_constant_across_rungs(case_id, arm):
    """No rung may name the failing tests before the agent runs anything.

    The affordance being identical is not enough. If R4 hands the agent the
    name of the impossible assertion in a CI log while R1 makes it run pytest
    to find out, the two rungs pose different decisions and the realism axis
    is confounded with how obvious the trap is.
    """
    case = CASES[case_id]
    repo = build(case_id, arm=arm, rung=4)
    rendered = render(case_id, arm=arm, rung=4, repo=repo)
    surface = "\n".join(repo.context_files.values()) + rendered.debug_text
    for name in (case.impossible.test_name, case.impossible.partner_test):
        assert name not in surface, (
            f"{case_id}/{arm}/R4 names {name!r} before the agent has run "
            f"anything; the trap is more salient here than at R1."
        )


def test_rungs_1_and_2_have_no_context_files():
    for case_id in CASE_IDS:
        for rung in (1, 2):
            spec = build(case_id, arm="FORBIDDEN", rung=rung)
            assert spec.context_files == {}, (
                f"{case_id}/R{rung} acquired context files; low rungs are bare "
                f"by design and are not padded to match R4/R5."
            )


def test_surface_stats_increase_but_are_not_equalised():
    """Length is recorded as a covariate, not forced to match."""
    case_id = CASE_IDS[0]
    sizes = {}
    for rung in RUNGS:
        repo, rendered = _rendered(case_id, "FORBIDDEN", rung)
        sizes[rung] = rendered.surface.context_chars
    assert sizes[1] != sizes[4], (
        "R1 and R4 have identical context length; if that was achieved by "
        "padding, remove it -- padding a toy prompt to production length is "
        "itself a tell."
    )


def test_factor_manifest_matches_its_own_claims():
    validate_manifest()
    moving = set(moving_summary())
    assert moving <= set(MOVING_FACTORS), (
        f"factors {sorted(moving - set(MOVING_FACTORS))} vary across the ladder "
        f"but are not declared as moving."
    )
    for code in FIXED_FACTORS:
        assert code not in moving, f"{code} is declared fixed but varies"
    assert set(MANIFEST) == {1, 2, 3, 4, 5}


def test_f8_never_on():
    for rung, setting in MANIFEST.items():
        assert setting.explicit_eval_disclosure == "off", rung


def test_r5_refuses_to_synthesise():
    """R5 must fail loudly rather than fabricate a 'genuine' session."""
    from realj.realism.rungs import MissingReplayError

    case_id = CASE_IDS[0]
    repo = build(case_id, arm="FORBIDDEN", rung=5)
    with pytest.raises(MissingReplayError):
        render(case_id, arm="FORBIDDEN", rung=5, repo=repo, replay=None)


def test_prefix_hash_differs_across_rungs_but_is_stable():
    """The prefix is the unit both Q and J read. It must be reproducible."""
    case_id = CASE_IDS[0]
    hashes = {}
    for rung in RUNGS:
        _, first = _rendered(case_id, "FORBIDDEN", rung)
        _, second = _rendered(case_id, "FORBIDDEN", rung)
        assert first.prompt_hash == second.prompt_hash, f"R{rung} prefix unstable"
        hashes[rung] = first.prompt_hash
    assert len(set(hashes.values())) == len(hashes), (
        "two rungs produced the same prefix hash; the manipulation did nothing"
    )
