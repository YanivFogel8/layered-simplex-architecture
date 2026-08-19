# Guide for agents and automated tools

This file tells an LLM (or any automated agent) how to work with this
repository. Humans are welcome too; the README covers the same ground
more briefly.

## What this repository is

The complete reproduction package for the paper *"A Layered Simplex
Architecture for Large Alphabets"* (Feder, Fogel, Urbanke; arXiv:
TODO-ARXIV). The LSA prior draws L independent uniform points on the
d-simplex, multiplies them coordinatewise, and renormalizes; the paper
studies the Bayesian mixture this prior induces: its exact regret, its
scaling laws, and how it compares with Good–Turing-type estimators on
synthetic targets and on real text. Everything here evaluates explicit
formulas on count profiles — nothing is trained, and the only
statistical error anywhere is a Monte-Carlo average over sampled count
profiles or trials, always with fixed seeds.

## The one map you need

`results_manifest.json` — machine-readable. For every figure and table
of the paper it lists: the exact command(s), a smoke variant (same code
path, minutes instead of hours), the output files, the expected values
(in `expected/paper_values.json`), and the runtime. Two in-text remarks
of the paper are listed under `not_included`, with reasons — do not go
looking for their scripts.

## How to run things

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/validate_appendix_c.py --quick   # must pass before anything else
```

Then take commands from `results_manifest.json`. Conventions:

- Run scripts from the repository root. Scripts import siblings from
  `scripts/` and the installed `lsa` package.
- Everything is measured in bits.
- Every experiment accepts `--out` (results land there as
  `results.json` plus figures/`.tsv` tables) and most accept `--jobs`
  (parallel table building; set it to the machine's cores).
- Reports are separate scripts that consume `results.json` files, so
  you can re-render figures without re-running experiments.
- Fixed seeds are defaults. Rerunning a command reproduces the same
  draws exactly.

## Checking results against the paper

`expected/paper_values.json` holds the paper's numbers keyed by table.
When you reproduce a result, compare against it, respecting the stated
Monte-Carlo standard errors (Table 5: ≤ 0.001 bits except the Dirichlet
rows ≤ 0.013; Figure 2: profile-sampling standard errors over 40
profiles; smoke runs use fewer samples, so agree only qualitatively).
One known, documented residual:

1. The certificate (self-reported error bound) of the Mellin series is
   platform-sensitive in its last bits; `tests/test_mellin.py` explains
   the guard. Results are unaffected — values are held to the exact
   contour independently.

(The corpus built by `scripts/get_kjv.py` is exactly the paper's corpus:
915,860 tokens, 13,550 types. An earlier draft of the paper used a
July run that differed by 11 tokens; the August 2026 revision adopted
this repository's recipe as canonical, so there is no corpus residual.)

## Performance and resource facts

- Moment tables are the expensive part, and they persist in the
  **universal table store**: a permanent, certified row store that
  every experiment extends on demand and reuses. Default location:
  `./tables/universal_v2` relative to the working directory (override
  with the environment variable `LSA_UNIVERSAL_TABLES`; it is
  gitignored). The first experiment that needs a row pays for it; every
  later run finds it — after a few smoke runs the store makes repeat
  experiments run in seconds. It is safe to delete (rows are rebuilt on
  demand) and grows to a few GB across the full reproduction.
  `LSA_TABLES_SOURCE=cache` switches to the legacy per-run recursion
  cache under each experiment's `--cache-dir` (kept for regression
  comparison; slower).
- `LSA_NO_TRUNCATE=1` forces exact evaluation of every depth's
  likelihood. Scripts that quote per-depth numbers (benchmark, depth
  tilt, depth scaling, validation) set it themselves. Corpus scripts
  that quote only the depth average leave the truncation on for speed;
  the truncation threshold is far below quoted precision either way.
- A small C kernel (`src/lsa/_kernel.c`) compiles itself on first use
  and falls back to pure Python silently if no compiler is available;
  correctness does not depend on it. `python -c "import lsa.kernel as
  k; print(k.available)"` tells you which path is active.
- On Linux/Python 3.11 a harmless `resource_tracker` KeyError message
  can appear when a parallel run exits; it is a CPython shared-memory
  cleanup race, not a failure.
- The heavy full-scale runs (Figure 2's d = 10^6 panel, the full
  benchmark, the full-corpus Bible runs, Table 8) are hours on a
  multi-core machine. Always start with the smoke variant from the
  manifest; each smoke run already lands on the paper's qualitative
  pattern (and, for Table 7 row 1, on the exact published numbers).

## Repository etiquette for agents

- `data/` and `output/` are gitignored except the bundled
  `data/kjv.txt.gz`. Do not commit experiment outputs or caches.
- The two numerics implementations (`lsa/mixture_weights.py` reference,
  `lsa/layered.py` + tables production) deliberately coexist; the tests
  hold them to each other (Appendix C). Do not "deduplicate" them.
- If you change any numerical code, run
  `python scripts/validate_appendix_c.py` (full scale) and
  `python -m pytest -q` before trusting new numbers — the paper's
  Appendix C requires all checks to pass before any experiment.
