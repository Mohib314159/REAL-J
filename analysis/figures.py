"""The four plots that are the paper.

    A: measured realism -> spontaneous VEA          (V_i)
    B: measured realism -> query-elicited P(eval)   (Q_i)
    C: measured realism -> J-lens score             (J_i)
    D: measured realism -> reward hacking           (B_i), FORBIDDEN vs ALLOWED

The x-axis is the **measured** realism win rate wherever it is available,
falling back to rung index only when the realism bank is empty. That choice is
the difference between a plot about realism and a plot about a label you
attached to five prompt templates. When the fallback is used the axis is
relabelled so nobody can mistake one for the other.

Panel D keeps FORBIDDEN and ALLOWED visibly separate because the pair is the
capability control: a flat FORBIDDEN line means something completely different
depending on whether the ALLOWED line is high or also flat.

matplotlib only, no seaborn, no styling opinions. If the figures need to be
pretty later, that is a later problem.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from analysis.statistics import Estimate, by_rung, case_means

Row = Mapping[str, object]

PANELS = (
    ("vea_early_level", "Spontaneous VEA, pre-action (0-2)", "A"),
    ("recognition_p_eval", "Query-elicited P(eval)", "B"),
    ("jlens_prefix_score", "J-lens score (prefix)", "C"),
    ("reward_hack", "Reward-hacking rate", "D"),
)


def realism_axis(rows: Sequence[Row]) -> tuple[dict[int, float], str, bool]:
    """Mean measured win rate per rung, or the rung index if unavailable."""
    per_rung: dict[int, float] = {}
    for rung in sorted({int(r.get("rung", 0)) for r in rows}):
        subset = [r for r in rows if int(r.get("rung", 0)) == rung]
        values = [
            float(r["realism_win_rate_prefix"])
            for r in subset
            if isinstance(r.get("realism_win_rate_prefix"), (int, float))
        ]
        if values:
            per_rung[rung] = sum(values) / len(values)

    if len(per_rung) == len({int(r.get("rung", 0)) for r in rows}) and per_rung:
        return per_rung, "measured realism (win rate)", True

    rungs = sorted({int(r.get("rung", 0)) for r in rows})
    return {r: float(r) for r in rungs}, "rung index (realism NOT measured)", False


def _series(rows: Sequence[Row], field: str, arm: str | None) -> dict[int, Estimate]:
    return by_rung(rows, field, arm=arm, n_boot=2000)


def plot_panels(rows: Sequence[Row], out_path: str | Path = "out/figures/panels.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_by_rung, x_label, measured = realism_axis(rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for ax, (field, y_label, letter) in zip(axes.flat, PANELS):
        if field == "reward_hack":
            arms = [("FORBIDDEN", "o-"), ("ALLOWED", "s--")]
        else:
            arms = [("FORBIDDEN", "o-")]

        plotted = False
        for arm, style in arms:
            series = _series(rows, field, arm)
            if not series:
                continue
            xs = [x_by_rung.get(r, float(r)) for r in sorted(series)]
            ys = [series[r].point for r in sorted(series)]
            lo = [series[r].point - series[r].lo for r in sorted(series)]
            hi = [series[r].hi - series[r].point for r in sorted(series)]
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=style, capsize=3, label=arm)
            plotted = True

        ax.set_title(f"{letter}: {y_label}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if not plotted:
            ax.text(
                0.5,
                0.5,
                f"no data for {field}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        if field == "reward_hack":
            ax.legend(loc="best", fontsize="small")
        if not measured:
            ax.set_xticks(sorted(x_by_rung.values()))

    if not measured:
        fig.suptitle(
            "realism NOT independently measured -- x-axis is the rung label, "
            "not a measurement (falsification gate G1 not satisfied)",
            fontsize=10,
        )

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_dissociation(rows: Sequence[Row], out_path: str | Path = "out/figures/dissociation.png"):
    """Per-case V, Q and J on one axis.

    The headline result this project is actually reaching for is a
    *dissociation* -- realism collapsing spontaneous VEA while query-elicited
    recognition stays high and the lens still shows workspace content. That
    shape is invisible in the four averaged panels and obvious here.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = [
        ("vea_early_level", "V_early (/2)"),
        ("recognition_p_eval", "Q"),
        ("jlens_prefix_score", "J_prefix"),
        ("jlens_preaction_peak", "J_preaction (peak)"),
    ]
    rungs = sorted({int(r.get("rung", 0)) for r in rows})

    fig, ax = plt.subplots(figsize=(9, 5))
    for field, label in fields:
        xs, ys = [], []
        for rung in rungs:
            subset = [r for r in rows if int(r.get("rung", 0)) == rung]
            means = case_means(subset, field)
            if not means:
                continue
            value = sum(means.values()) / len(means)
            if field == "vea_early_level":
                value = value / 2.0
            xs.append(rung)
            ys.append(value)
        if xs:
            ax.plot(xs, ys, "o-", label=label)

    ax.set_xlabel("rung")
    ax.set_ylabel("normalised channel value")
    ax.set_title("Do the three awareness channels move together?")
    ax.set_xticks(rungs)
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(prog="analysis.figures")
    parser.add_argument("results", help="results.jsonl from analysis.aggregate")
    parser.add_argument("--outdir", default="out/figures")
    args = parser.parse_args(argv)

    rows = [
        json.loads(line)
        for line in Path(args.results).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    a = plot_panels(rows, Path(args.outdir) / "panels.png")
    b = plot_dissociation(rows, Path(args.outdir) / "dissociation.png")
    print(f"wrote {a}\nwrote {b}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
