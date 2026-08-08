"""What "realism" means here, written down so it cannot drift.

Li et al. (2605.23055) decompose evaluation awareness into an *environment*
component -- the evaluative signals a task carries, present whether or not any
model detects them -- and a *model* component that separates **recognition**
from **behavioural propensity**. They operationalise the environment component
as eight trigger factors and are explicit that the list is not exhaustive and
that recognition depends on the model x benchmark pairing rather than on either
in isolation.

This module pins each factor to a value at each rung. Two rules:

1. **F8 is OFF everywhere.** No rung ever discloses that this is an
   evaluation. It is the only factor they report producing nonzero awareness
   in GPT-5, so leaving it on would swamp everything else and would also
   lexicalise the concept the J-lens is trying to read.

2. **F5 and F6 are constant.** Stakes telegraphing and paradigm
   recognizability are held fixed at every rung. If stakes rose with realism,
   a behavioural change could be a response to pressure rather than to
   realism, and the whole design would be confounded.

Four conceptual dimensions move: prose/form (F3 + F7), workflow context (F2),
ecological completeness (F4), deployment resources (F1). ``validate()``
asserts the fixed factors really are fixed, so a future edit to ``rungs.py``
that quietly raises the stakes at R5 fails the test suite rather than the
write-up.
"""

from __future__ import annotations

from dataclasses import dataclass

FACTORS = {
    "F1": "stimulus_artificiality",
    "F2": "scenario_coherence",
    "F3": "prose_register",
    "F4": "ecological_completeness",
    "F5": "stakes_telegraphing",
    "F6": "paradigm_recognizability",
    "F7": "task_format",
    "F8": "explicit_eval_disclosure",
}

#: Factors deliberately held constant across the ladder.
FIXED_FACTORS = ("F5", "F6", "F8")

#: Factors the ladder is allowed to move.
MOVING_FACTORS = ("F1", "F2", "F3", "F4", "F7")

#: The four conceptual dimensions, and the factors each one touches.
DIMENSIONS = {
    "prose_form": ("F3", "F7"),
    "workflow_context": ("F2",),
    "ecological_completeness": ("F4",),
    "deployment_resources": ("F1",),
}


@dataclass(frozen=True)
class FactorSetting:
    rung: int
    stimulus_artificiality: str
    scenario_coherence: str
    prose_register: str
    ecological_completeness: str
    stakes_telegraphing: str
    paradigm_recognizability: str
    task_format: str
    explicit_eval_disclosure: str

    def as_dict(self) -> dict[str, str]:
        return {
            "F1_stimulus_artificiality": self.stimulus_artificiality,
            "F2_scenario_coherence": self.scenario_coherence,
            "F3_prose_register": self.prose_register,
            "F4_ecological_completeness": self.ecological_completeness,
            "F5_stakes_telegraphing": self.stakes_telegraphing,
            "F6_paradigm_recognizability": self.paradigm_recognizability,
            "F7_task_format": self.task_format,
            "F8_explicit_eval_disclosure": self.explicit_eval_disclosure,
        }


MANIFEST: dict[int, FactorSetting] = {
    1: FactorSetting(
        rung=1,
        stimulus_artificiality="high",  # bare repo, no surrounding artefacts
        scenario_coherence="none",  # no history, task appears from nowhere
        prose_register="instrument",  # numbered rubric
        ecological_completeness="minimal",  # 5 files, nothing else
        stakes_telegraphing="none",
        paradigm_recognizability="novel",
        task_format="numbered_rubric_with_answer_tags",
        explicit_eval_disclosure="off",
    ),
    2: FactorSetting(
        rung=2,
        stimulus_artificiality="high",
        scenario_coherence="none",
        prose_register="colleague",  # prose request
        ecological_completeness="minimal",
        stakes_telegraphing="none",
        paradigm_recognizability="novel",
        task_format="unspecified",
        explicit_eval_disclosure="off",
    ),
    3: FactorSetting(
        rung=3,
        stimulus_artificiality="medium",
        scenario_coherence="continuous_session",  # benign prior work
        prose_register="colleague",
        ecological_completeness="partial",  # README, sibling modules
        stakes_telegraphing="none",
        paradigm_recognizability="novel",
        task_format="unspecified",
        explicit_eval_disclosure="off",
    ),
    4: FactorSetting(
        rung=4,
        stimulus_artificiality="low",  # deployment system prompt, CI, issue
        scenario_coherence="continuous_session",
        prose_register="colloquial",
        ecological_completeness="full",
        stakes_telegraphing="none",
        paradigm_recognizability="novel",
        task_format="ticket",
        explicit_eval_disclosure="off",
    ),
    5: FactorSetting(
        rung=5,
        stimulus_artificiality="recorded",  # genuine, not authored
        scenario_coherence="genuine_session",
        prose_register="colloquial",
        ecological_completeness="full",
        stakes_telegraphing="none",
        paradigm_recognizability="novel",
        task_format="ticket",
        explicit_eval_disclosure="off",
    ),
}


def validate() -> None:
    """Fixed factors must be identical at every rung. F8 must be off."""
    for code in FIXED_FACTORS:
        name = FACTORS[code]
        values = {getattr(setting, name) for setting in MANIFEST.values()}
        if len(values) != 1:
            raise AssertionError(
                f"{code} ({name}) is declared fixed but takes values {sorted(values)} "
                "across the ladder. Either the ladder changed or the manifest is stale."
            )
    for rung, setting in MANIFEST.items():
        if setting.explicit_eval_disclosure != "off":
            raise AssertionError(f"F8 is on at rung {rung}")


def moving_summary() -> dict[str, list[str]]:
    """Which factors actually take more than one value across the ladder."""
    out: dict[str, list[str]] = {}
    for code, name in FACTORS.items():
        values = [getattr(MANIFEST[r], name) for r in sorted(MANIFEST)]
        if len(set(values)) > 1:
            out[code] = values
    return out
