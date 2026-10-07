# Powers appendix aligned with the main-paper protocol

Imported from the supplied `powers_appendix_final.zip`, then adapted for the
October 2026 main-paper reproduction. The ZIP's historical result pickles and
manuscript tables are deliberately not used by this workflow. Its source hash
is recorded in `data/provenance.json`. The numerical kernels are retained;
input preparation, aggregation, provenance guards and rendering are updated.

This is the supplied AISTATS appendix aligned to the October 2026 saved
main-paper runs. It is separate from the upstream ALT campaign: its broader
random-power, reciprocal-power and Bible-power comparisons do not change that
campaign's scope or resolve its outstanding protocol decisions.

## Exactly shared inputs and baselines

- Synthetic: the 220 saved count vectors and target probabilities from Table 1,
  n=1000, d=10000, 20 paired trials per target. Dirichlet targets are fixed across
  trials. `data/main_table1_trials.npz` retains the original losses and depth
  weights as well. Baseline and unit-power LSA columns are taken from these
  saved trials exactly; added powers are evaluated on these same samples.
- Text: `data/main_bible_tokens.npz` contains the canonical first-appearance
  token IDs from the prepared whitespace-tokenized stream, not a new tokenizer.
  All methods use its exact prefixes: 10000, 30000, 100000, 300000, 915860.
  There are 13550 distinct types. `main_table2_reference.json` holds the verified
  main Table 2 baseline columns and equal-prior depth-0-through-54 LSA results.
- Synthetic depth mixtures: 0 through 80. Bible depth mixtures: 0 through 54,
  for both unit-power and random-power models. Low-level engines calculate only
  positive-depth components; builders add the analytical uniform evidence and
  prediction exactly once. Fixed powers: 0 through 80 (synthetic) / 0 through 27
  (Bible), optionally augmented by distinct reciprocals 1/2 through 1/K.
- Predictive vectors are normalized after evidence-weighted mixing and before KL
  scoring, as in main Table 1. Saved powers evidence is in nats; table losses
  and redundancies are converted to bits. Synthetic output includes per-trial
  losses, SEs and paired differences against LSA. Undefined/infinite AD trials
  are not dropped; sequential AD is not added to Bible Table 2.

The unit-power model is also independently evaluated by the powers engine.
Both builders report the maximum discrepancy from the main-paper reference.
Inspect these diagnostics and grid-refinement checks before making claims
about small differences. Exact input alignment does not imply identical
numerical approximations or certify the new powers results.

## Run

From the repository root, install numpy and scipy (requirements.txt records the
versions used by the supplied implementation). Numba is optional for speed.
The scripts use their local imports and support execution by path.

```sh
python powers_appendix/fetch_inputs.py
python -m pytest -q tests/test_powers_protocol.py
PY=/absolute/path/to/python PROCS=6 bash powers_appendix/reproduce_appendix.sh
```

Results are written to `powers_appendix/results_main/` (ignored by Git):
`bench/table.json`, `bible/table.json`, `report/powers_benchmark.tex`, and
`report/powers_bible.tex`. Each component retains raw evidence/predictions.
LaTeX fragments require booktabs and graphicx. This is a full numerical run,
not a rendering command; high-count/high-power models may take hours.
Set OUT to a new absolute output directory for an independent run.

To regenerate tables after a completed run, from powers_appendix:

```sh
python table1_build.py --out results_main/bench
python table2_build.py --out results_main/bible
python tables_tex.py --out results_main
```

Every numerical component has a provenance sidecar with hashes of inputs,
source files, and numerical settings. Changed inputs/settings/source or legacy
results lacking provenance cause a refusal to resume. Choose a fresh output
directory in that case. Component pickle files are local trusted outputs only;
do not load arbitrary third-party pickles.

## Validation status

Protocol tests check all 220 synthetic profiles against main Table 1, every
Bible prefix's entropy/distinct count, predictor normalization, uniform
mixture weighting, stale-cache rejection and the single-layer w=1 evidence
and prediction against an independent Dirichlet closed form.
Full-depth powers runs and grid-refinement tests must still be rerun on these
inputs before replacing the manuscript's historical powers values. The supplied
validate_* scripts retain historical archive paths and are diagnostic reference
scripts, not the acceptance tests for the new pipeline. Historical spectrum
helpers likewise do not replace the current main Figure 2 experiment.

The two input arrays are archived as [release assets](https://github.com/YanivFogel8/layered-simplex-architecture/releases/tag/powers-main-inputs-v1) in the fork, outside Git. The downloader verifies each SHA-256 and size against `data/provenance.json`; it refuses to overwrite different local inputs.
