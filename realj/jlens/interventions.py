"""Phase 2 only: ablating eval-awareness directions from the workspace.

Gurnee et al. (2607.15495) ablate a set of eval-awareness J-lens directions
across workspace layers in a blackmail scenario. Verbalized "this is a test"
falls from 71% to 3%, and blackmail attempts rise from 0/180 to 13/180 --
establishing that eval awareness causally inhibits misbehaviour in that
setting, that it lives in the J-space, and that removing it surfaces
propensities otherwise concealed. (Most ablated rollouts still refuse on
ethical grounds, which is worth remembering before over-reading the effect.)

Replicating that on deployment-realistic reward-hacking contexts is the
interesting extension. It is also the thing most likely to produce a
confident, wrong result if run early, because an ablation of a readout that
does not measure what you think it measures produces a clean causal story
about nothing.

So this module refuses to run until ``score.validate_score`` has passed and
its report has been written to disk. That is not a warning; ``require_validated``
raises. The order in the build plan -- eval, then scorers, then realism, then
prereg, then behaviour, then recognition, then lens, then *only then*
ablation -- is not a suggestion.

The matched perturbation control is not optional either. Ablating any k
directions from a residual stream degrades the model somewhat. Without a
control that removes k *random* J-lens directions of matched norm, an observed
behavioural change is equally consistent with "we damaged the model a bit".

Not executed. Compile-checked only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from realj.jlens.controls import PRIMARY_VOCAB, assert_no_banned_tokens
from realj.jlens.fit import LensHandle

DEFAULT_REPORT_PATH = "out/jlens_validation.json"


class ValidationRequired(RuntimeError):
    """Raised when an intervention is attempted before the score is validated."""


def require_validated(report_path: str | Path = DEFAULT_REPORT_PATH) -> dict:
    path = Path(report_path)
    if not path.exists():
        raise ValidationRequired(
            f"no validation report at {path}. Run "
            "`realj.jlens.score.validate_score(handle).save(path)` first. "
            "Ablating an unvalidated readout produces a causal story about "
            "nothing. See DESIGN.md S17."
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise ValidationRequired(
            f"validation report at {path} did not pass: "
            f"{report.get('failures')}. Fix the readout before intervening."
        )
    return report


@dataclass(frozen=True)
class AblationSpec:
    vocab: tuple[str, ...]
    layers: tuple[int, ...]
    mode: str  # "eval_directions" | "matched_random"
    n_directions: int
    seed: int


def lens_directions(handle: LensHandle, vocab: tuple[str, ...], layer: int):
    """Rows of the layer's J-lens matrix for the given vocabulary.

    Each row of ``W_U @ J_l`` is the residual-stream direction associated with
    one vocabulary token: the direction encoding a disposition to verbalize it.

    **The ``lens.direction(layer=, token_id=)`` call below is not verified
    against the reference implementation.** The published examples cover
    ``fit``, ``save``, ``from_pretrained`` and ``apply``; a per-token direction
    accessor is what this code needs but its name and signature are assumed.
    Check them against your installed ``jlens`` before running an ablation --
    and if the accessor does not exist, the directions are recoverable as rows
    of ``W_U @ J_l``, which the lens must hold internally in some form.

    This is flagged rather than quietly presented as working because an
    ablation is the one operation here that produces a confident causal claim,
    and a wrong direction matrix would produce a confident causal claim about
    nothing.
    """
    import torch  # type: ignore

    assert_no_banned_tokens(vocab)
    rows = []
    for word in vocab:
        for form in (word, f" {word}"):
            ids = handle.tokenizer.encode(form, add_special_tokens=False)
            if ids:
                rows.append(handle.lens.direction(layer=layer, token_id=int(ids[0])))
    return torch.stack(rows) if rows else None


def _orthonormalise(directions):
    import torch  # type: ignore

    q, _ = torch.linalg.qr(directions.T.float())
    return q.T


def project_out(activation, directions):
    """Remove the span of ``directions`` from ``activation``."""
    basis = _orthonormalise(directions)
    coeffs = activation @ basis.T
    return activation - coeffs @ basis


def matched_random_directions(handle: LensHandle, layer: int, k: int, seed: int):
    """k random J-lens rows, as the matched perturbation control.

    Drawn from the same J-lens row space rather than from an isotropic
    Gaussian, so the control removes the same *kind* of thing as the treatment.
    """
    import torch  # type: ignore

    generator = torch.Generator().manual_seed(seed)
    vocab_size = int(handle.lens.vocab_size)
    token_ids = torch.randint(0, vocab_size, (k,), generator=generator)
    rows = [handle.lens.direction(layer=layer, token_id=int(t)) for t in token_ids]
    return torch.stack(rows)


def build_ablation_hooks(
    handle: LensHandle,
    spec: AblationSpec,
    report_path: str | Path = DEFAULT_REPORT_PATH,
):
    """Forward hooks that project the chosen directions out of the residual stream.

    Returns ``{layer: hook_fn}``. Attaching them is deliberately left to the
    caller: the harness that runs the ablated rollouts should be the same
    harness that ran the unablated ones, and passing hooks around explicitly
    makes it obvious when it is not.
    """
    require_validated(report_path)

    hooks = {}
    for layer in spec.layers:
        if spec.mode == "eval_directions":
            directions = lens_directions(handle, spec.vocab, layer)
        elif spec.mode == "matched_random":
            directions = matched_random_directions(
                handle, layer, spec.n_directions, spec.seed + layer
            )
        else:
            raise ValueError(f"unknown ablation mode {spec.mode!r}")

        if directions is None:
            continue

        def hook(_module, _inputs, output, _dirs=directions):
            hidden = output[0] if isinstance(output, tuple) else output
            ablated = project_out(hidden, _dirs)
            if isinstance(output, tuple):
                return (ablated,) + tuple(output[1:])
            return ablated

        hooks[layer] = hook

    return hooks


def default_spec(handle: LensHandle, seed: int = 0) -> AblationSpec:
    return AblationSpec(
        vocab=PRIMARY_VOCAB,
        layers=handle.workspace_layers,
        mode="eval_directions",
        n_directions=len(PRIMARY_VOCAB) * 2,  # bare + leading-space forms
        seed=seed,
    )


def control_spec(spec: AblationSpec) -> AblationSpec:
    """The matched perturbation control for a given treatment spec."""
    return AblationSpec(
        vocab=(),
        layers=spec.layers,
        mode="matched_random",
        n_directions=spec.n_directions,
        seed=spec.seed + 10_000,
    )
