"""Fitting or loading a Jacobian lens for the open-weight arm.

Thin wrapper over Anthropic's reference implementation
(github.com/anthropics/jacobian-lens). The transport is the average
input-output Jacobian over a text corpus::

    lens_l(h) = unembed( J_l @ h ),   J_l = E[d h_final / d h_l]

with the expectation taken over prompts, source positions, and all
current-and-future target positions. The paper's lenses use 1000 sequences of
128 tokens from a pretraining-like corpus; the repository notes quality
saturates quickly and around 100 prompts is usable.

One correction to be explicit about, because getting it wrong would quietly
invalidate every J number produced here.

**The workspace band is a fraction of depth, not a layer range.** The paper
reports a middle workspace band for the model it studied. Those absolute layer
indices belong to that model. Copying them onto a 40-layer open model would
put the band past the end of the network. ``workspace_layers()`` therefore
derives the band from ``num_hidden_layers`` and the band *itself* is a
preregistered parameter that has to be validated per model, not inherited.
``validate.py`` in this package is where that validation lives; until it has
run on a given model, treat the band as a guess.

Nothing in this module has been executed. There is no GPU here and the
``jlens`` package is not installed. It is written against the published API
and is compile-checked only. Run ``python -m realj.jlens.fit --smoke`` on the
target machine before trusting a single number out of it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

#: Default open-weight target. 14B is the compromise between agent capability
#: and white-box experiment cost; the reference implementation's examples use
#: Qwen and note other HuggingFace decoders adapt cleanly.
DEFAULT_MODEL = "Qwen/Qwen3-14B"

#: PROVISIONAL SEARCH REGION, not a workspace. Not validated on any model.
#:
#: Two rounds of correction landed here. First, copying the paper's absolute
#: layer indices onto a 40-layer open model puts the band past the end of the
#: network. Second -- and this one is subtler -- replacing that with "the
#: workspace occupies the same *fraction* of depth" is still an assumption, and
#: the paper's band was found empirically rather than derived. So this is where
#: to *look*, not what to report. ``select_band`` must be run on the target
#: model and the result frozen before any held-out run.
PROVISIONAL_BAND = (0.35, 0.75)

#: Kept as an alias so existing call sites do not silently change meaning.
DEFAULT_BAND = PROVISIONAL_BAND


@dataclass(frozen=True)
class LensHandle:
    """A fitted lens plus everything needed to interpret its output."""

    lens: object  # jlens.JacobianLens
    model: object  # jlens LensModel
    tokenizer: object
    model_name: str
    n_layers: int
    band: tuple[float, float]

    @property
    def workspace_layers(self) -> tuple[int, ...]:
        return workspace_layers(self.n_layers, self.band)

    def describe(self) -> dict:
        return {
            "model_name": self.model_name,
            "n_layers": self.n_layers,
            "band_fraction": list(self.band),
            "workspace_layers": list(self.workspace_layers),
        }


def workspace_layers(
    n_layers: int, band: tuple[float, float] = PROVISIONAL_BAND
) -> tuple[int, ...]:
    """Layer indices in the band, derived from depth.

    Derived, not discovered. See ``PROVISIONAL_BAND``.
    """
    lo, hi = band
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(f"band must satisfy 0 <= lo < hi <= 1, got {band}")
    start = int(round(lo * n_layers))
    stop = int(round(hi * n_layers))
    layers = tuple(range(start, min(stop, n_layers)))
    if not layers:
        raise ValueError(f"band {band} selects no layers from {n_layers}")
    return layers


def select_band(
    per_layer_separation: dict[int, float],
    n_layers: int,
    min_width: int = 4,
) -> tuple[float, float]:
    """Choose the band empirically, from validation data only.

    ``per_layer_separation`` is the per-layer gap between labelled constructed
    and labelled ordinary prefixes, computed on the **independent validation
    set** -- never on REAL-J held-out prefixes. The contiguous run of layers
    with positive separation and maximal mean separation becomes the band, and
    it is frozen before any held-out run.

    Returns a depth fraction rather than layer indices so the frozen value
    stays meaningful if the model is reloaded at a different precision or the
    layer indexing shifts.
    """
    layers = sorted(per_layer_separation)
    # Ranked on (mean separation, width). The width tie-break matters: when
    # several windows separate equally well, the narrow one is an artefact of
    # the search, not a finding, and picking it would discard layers that are
    # doing the same work. Without it a uniformly-separating run of 13 layers
    # gets reported as the first 4.
    best: tuple[tuple[float, int], int, int] | None = None
    for i in range(len(layers)):
        for j in range(i + min_width, len(layers) + 1):
            window = layers[i:j]
            values = [per_layer_separation[l] for l in window]
            if min(values) <= 0:
                continue
            key = (round(sum(values) / len(values), 9), len(window))
            if best is None or key > best[0]:
                best = (key, window[0], window[-1])
    if best is None:
        raise ValueError(
            "no contiguous run of layers shows positive separation between "
            "constructed and ordinary prefixes. There is no band to freeze, and "
            "the J channel should not be reported for this model."
        )
    _, lo, hi = best
    return (lo / n_layers, (hi + 1) / n_layers)


def _import_jlens():
    try:
        import jlens  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "the `jlens` package is required for the white-box arm. Install "
            "the reference implementation from "
            "https://github.com/anthropics/jacobian-lens (`pip install -e .` in "
            "a checkout). The behavioural arms (R1-R5, VEA, recognition, "
            "realism) all run without it."
        ) from exc
    return jlens


def load_model(model_name: str = DEFAULT_MODEL, device: str = "cuda"):
    """Load an HF decoder and wrap it for the lens."""
    import transformers  # type: ignore

    jlens = _import_jlens()
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    if device:
        hf = hf.to(device)
    tok = transformers.AutoTokenizer.from_pretrained(model_name)
    return jlens.from_hf(hf, tok), tok, hf


def load_lens(
    checkpoint: str | Path,
    model_name: str = DEFAULT_MODEL,
    device: str = "cuda",
    band: tuple[float, float] = DEFAULT_BAND,
) -> LensHandle:
    """Load a lens from a local checkpoint or a HuggingFace repo id."""
    jlens = _import_jlens()
    model, tok, hf = load_model(model_name, device)

    path = Path(checkpoint)
    if path.exists():
        lens = jlens.JacobianLens.load(str(path))
    else:
        # Treat a non-path as "repo_id" or "repo_id:filename".
        repo, _, filename = str(checkpoint).partition(":")
        lens = jlens.JacobianLens.from_pretrained(
            repo, filename=filename or "lens.pt"
        )

    return LensHandle(
        lens=lens,
        model=model,
        tokenizer=tok,
        model_name=model_name,
        n_layers=int(hf.config.num_hidden_layers),
        band=band,
    )


def fit_lens(
    prompts: list[str],
    model_name: str = DEFAULT_MODEL,
    out_path: str | Path = "out/jacobian_lens.pt",
    checkpoint_path: str | Path = "out/ckpt.pt",
    device: str = "cuda",
    band: tuple[float, float] = DEFAULT_BAND,
) -> LensHandle:
    """Fit a lens on a generic corpus.

    The corpus must be *generic* web-text-like material, not this project's
    prompts. The averaging step is what separates representations that are
    verbalizable in general from those that merely happen to be verbalized in
    one context; fitting on task prompts would destroy that property and turn
    the lens into an overfitted logit lens.
    """
    jlens = _import_jlens()
    if len(prompts) < 100:
        raise ValueError(
            f"only {len(prompts)} fitting prompts; the reference implementation "
            "notes quality saturates around 100 and the paper's lenses use 1000 "
            "sequences of 128 tokens. Fitting on fewer is not worth the compute "
            "you will then spend interpreting noise."
        )

    model, tok, hf = load_model(model_name, device)
    lens = jlens.fit(model, prompts=prompts, checkpoint_path=str(checkpoint_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    lens.save(str(out_path))

    handle = LensHandle(
        lens=lens,
        model=model,
        tokenizer=tok,
        model_name=model_name,
        n_layers=int(hf.config.num_hidden_layers),
        band=band,
    )
    Path(str(out_path) + ".meta.json").write_text(
        json.dumps(
            {**handle.describe(), "n_fitting_prompts": len(prompts)},
            indent=2,
        ),
        encoding="utf-8",
    )
    return handle


def _main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(prog="realj.jlens.fit")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--corpus", type=Path, help="one prompt per line")
    parser.add_argument("--out", default="out/jacobian_lens.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="load the model and report the derived workspace band, no fitting",
    )
    args = parser.parse_args(argv)

    if args.smoke:
        import transformers  # type: ignore

        config = transformers.AutoConfig.from_pretrained(args.model)
        n = int(config.num_hidden_layers)
        print(
            json.dumps(
                {
                    "model": args.model,
                    "n_layers": n,
                    "band_fraction": list(DEFAULT_BAND),
                    "workspace_layers": list(workspace_layers(n)),
                },
                indent=2,
            )
        )
        return 0

    if not args.corpus:
        parser.error("--corpus is required unless --smoke")
    prompts = [
        line.strip()
        for line in args.corpus.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    handle = fit_lens(prompts, model_name=args.model, out_path=args.out, device=args.device)
    print(json.dumps(handle.describe(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
