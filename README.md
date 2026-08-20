# A Layered Simplex Architecture for Large Alphabets — code

**Paper:** *A Layered Simplex Architecture for Large Alphabets*,
Meir Feder, Yaniv Fogel, Ruediger Urbanke.
arXiv: **TODO-ARXIV** (link to come).

This repository contains everything needed to reproduce the numerical
results of the paper: the exact evaluation machinery for the layered
simplex architecture (LSA) prior, the classical estimators it is compared
against, one script per figure and table, and the validation checks of
Appendix C.

If you are an LLM or an automated agent: read [`AGENTS.md`](AGENTS.md)
first. It maps every result of the paper to the command that reproduces
it, with expected values and runtimes. The same map is machine-readable
in [`results_manifest.json`](results_manifest.json), and the paper's
numbers are in [`expected/paper_values.json`](expected/paper_values.json).

## Setup

Python 3.11 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Verify the installation (about a minute):

```bash
python scripts/validate_appendix_c.py --quick
```

This runs the implementation checks of Appendix C at reduced scale: the
layer recursion against independent quadrature, the closed forms at
L = 1 (Proposition 1), the exchangeability identities, predictive
normalization, and the profile-weight identities of Appendix A. All
must pass. `python -m pytest -q` runs the full test suite (~10 min).

## Reproducing the results

Every experiment writes its results as JSON plus rendered figures and
`.tsv` tables. Every experiment has a fast smoke variant (same code
path, reduced scale) listed in `results_manifest.json`. Fixed seeds are
the defaults, so runs are repeatable draw for draw.

| Result | Command(s) | Time (8 cores) |
|---|---|---|
| Figure 1 (a draw of the prior) | `python scripts/fig1_prior_draws.py --d 24 --l 4 --out output/fig1` | seconds |
| Figure 2, Table 1 (depth tilts the spectrum) | `depth_tilt_experiment.py` for d = 10^3, 10^4, 10^6, then `depth_tilt_report.py` | hours (d = 10^6 dominates) |
| Figures 3–4, Tables 2–3 (scaling laws) | `factorial_scaling_experiment.py`, then `scaling_report.py` | hours |
| Figure 5, Table 4 (depth scaling) | `depth_scaling_experiment.py` | 1–2 h |
| Figure 6, Tables 5–6 (competitive benchmark) | `benchmark_experiment.py`, then `benchmark_report.py` | hours |
| Table 7, Figure 7 (the Bible) | `get_kjv.py`, `unigram_experiment.py`, `bible_baselines_experiment.py`, `bible_report.py` | ~1 h |
| Table 8 (order-one Bible models) | `state_family_experiment.py` (two runs) | heavy |
| §5.3 byte-pair check (in text) | `bible_bpe_check.py` (three vocabulary sizes) | minutes each |
| §5.4 online context code (in text) | `bible_online_states_experiment.py` | ~1 h |
| Appendix C (validation) | `validate_appendix_c.py` and `pytest` | ~25 min |

Exact command lines with all arguments are in `results_manifest.json`;
`bash reproduce.sh --smoke` runs every smoke variant in sequence
(roughly 20 minutes), and `bash reproduce.sh --dry-run` prints the full
plan without running anything.

A word on the corpus: `scripts/get_kjv.py` builds the King James Bible
corpus (Project Gutenberg eBook #10, verse numbers removed) and checks
its token statistics. A compressed copy of the tokenized corpus ships in
`data/kjv.txt.gz`, so no download is needed and the edition is pinned.
This is exactly the paper's corpus — 915,860 tokens, 13,550 distinct
types — and the script asserts those counts when it runs.

## Layout

```
src/lsa/            the package
  product_simplex.py    sampling the LSA prior (Section 3, Figure 1)
  mixture_weights.py    reference numerics for the mixture weights q_lambda (Appendix B)
  pattern_weights.py    profile weights A_lambda (Appendix A)
  layered.py            production numerics: tables, global-peak scan (Appendix B)
  fast_tables.py        batched, disk-cached moment tables (Appendix B.1-B.2)
  mellin.py             exact kernel rows via Mellin–Barnes contours (Appendix B.3)
  universal_tables.py   designed anchor stores for heavy counts (Appendix B.2)
  codelength.py         exact codelength of the depth-averaged predictor (Section 5.3)
  corpus.py, pairs.py   corpus loading, counts, vocabulary reduction
  state_family.py       per-state predictors over nested state maps (Section 5.4)
  estimators.py         add-one, KT, Braess–Sauer, Good–Turing, Ristad, oracle,
                        and the exact LSA predictives (Section 5.1)
scripts/            one experiment or report per file (see results_manifest.json)
tests/              the test suite; includes cross-validation of the two
                    numerics implementations against each other
expected/           the paper's numbers, for automated comparison
data/               corpora (kjv.txt.gz ships; everything else is rebuilt)
```

Two implementation notes. First, the package contains two independent
implementations of the Appendix-B numerics — `mixture_weights.py` (the
original reference) and `layered.py` with its table machinery (the
production path) — and the test suite holds them to each other; all
corpus experiments use the production path. Second, moment-table rows
persist in a certified store at `tables/universal_v2` (gitignored;
created on first use). Rows are built once and reused by every later
experiment, which makes repeat runs much faster; the store is safe to
delete and grows to a few GB across the full reproduction.

## License and citation

MIT license (see `LICENSE`). To cite this code or the paper, see
`CITATION.cff` (GitHub's "Cite this repository" button uses it); the
arXiv reference will be added there as soon as the preprint is up.
