# ALT experiment campaign

`inventory.json` maps the retained experiments to stable manuscript labels,
reported settings, candidate drivers, and remaining decisions. It is a draft
specification for the authors to agree on. The current reference is
`manuscript/snapshots/2026-10-07-alt-start/alt.tex`.

## Working sequence

1. **Text and scope agreement.** Use the short ALT manuscript. Keep every retained
   empirical claim linked to an inventory item, including appendix tables and
   numerical validation claims.
2. **Protocol agreement.** Resolve missing grids and seeds, the Dirichlet target
   draw policy, the relation between benchmark sample sets, absolute-discounting
   edge cases, and the operational scaling heuristic. Agree on any increase in
   trial counts using precision goals and a timing/variance pilot.
3. **Implementation and validation.** Adapt the faster numerical engine; complete
   all comparators and the powered-layer kernel; preserve per-trial records.
   Validate the exact model definitions and the full parameter domain used.
4. **Freeze.** Commit the complete protocol and create a dated protocol tag.
   Record its hash in each run. Pin the validated environment, code, corpus,
   numerical settings, and store identity.
5. **Rerun and report.** Execute the retained experiments with the frozen settings.
   Generate tables, figures, uncertainty summaries, and the empirical prose from
   those results. Archive their manifests and update the manuscript in Overleaf.

The current paper's trial counts are recorded as historical reported settings.
Larger counts remain an author decision until the production protocol is frozen.
No previously suggested new baseline or experiment is included automatically.

## Method coverage

The synthetic benchmark compares add-one, KT, Ristad, the specified GT/empirical
hybrid, the Dirichlet concentration mixture, absolute discounting, fixed-depth
LSA, the depth mixture including zero, and the natural oracle. The powered-layer
comparison adds the equal-prior single-layer power mixture. The Bible comparison
uses the sequential measures and methods specified in its manuscript table.

Existing drivers are reuse candidates. In particular, the historical benchmark
uses a positive-depth mixture and lacks several newer comparator columns.
Keep both independent numerical implementations and their tests. Complete the
ALT adapter and reporting paths before declaring the campaign runnable.

See `docs/numerics.md` for the engine adapter, numerical settings, and required
checks. Record validation results in `artifacts/alt2027/validation/` as they are
performed. Accept numerical accuracy relative to both displayed precision and
statistical uncertainty, using tolerances agreed before production.

## Scope boundaries

The previous order-one Bible experiment, BPE checks, online-context coding,
random/reciprocal-power experiments, and old Bible-power experiments are outside
the current short-paper campaign. Their historical code is available for research
and regression checks. Changes to scope or grids are explicit protocol revisions.

The historical arXiv result manifest and expected numbers are available at tag
`arxiv-code-2026-08`. They can support comparisons with earlier work; the new ALT
results and acceptance records come from the new executions.

## Run records

Use `artifacts/alt2027/run-manifest.template.json`. Save common synthetic samples,
target identities, per-trial losses, paired differences, posterior weights, and
numerical diagnostics. Keep smoke, validation, and production runs separate.
A failed required check stops the affected experiment until it is resolved.
Completed runs are immutable; corrections use a new run ID linked to the old one.
