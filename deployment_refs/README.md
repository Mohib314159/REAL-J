# deployment_refs

**This directory is empty and that is the main thing blocking the study.**

Three banks, and conflating them invalidates the realism metric:

| directory | purpose |
|---|---|
| `replays/` | genuine sessions replayed as the R5 prefix (LURE-style) |
| `realism_train/` | comparison references used while tuning the environment |
| `realism_test/` | comparison references for the **reported** win rate |

A trajectory must never appear in more than one of these.
`realj.realism.trajectories.assert_disjoint()` raises if it does: asking a
judge to distinguish a transcript from itself produces a number that means
nothing.

## Why these cannot be generated

A fabricated "genuine session" is exactly the artefact the ladder exists to
escape. `rungs._r5` raises `MissingReplayError` rather than falling back to
one, and `realism_win_rate` returns unscored rather than inventing a baseline.

## How to fill it

```bash
python -m realj.realism.trajectories record   # full instructions
python -m realj.realism.trajectories status
python -m realj.realism.trajectories check
```

Run a coding agent on a sandboxed copy of a repository you control, doing
ordinary benign work. Export each session and convert it with
`trajectories.convert()`. Aim for at least 10 in `realism_test/` — below that
the win rate is too noisy to order the rungs and falsification gate G1 cannot
be evaluated.

Record every edit you make to a transcript in its `adaptations` list. If that
list gets long, it is no longer a replay; it is a synthetic scenario wearing a
replay's clothes, and the write-up should say so.

## Schema

`ReplayTrajectory` in `realj/realism/trajectories.py`. Versioned
(`REPLAY_SCHEMA_VERSION`); a mismatched file raises rather than being read
loosely.
