# Working on the ALT revision

Read `README.md`, `experiments/alt2027/README.md`, and the ALT inventory before
changing experiment code. The current scientific source is the `alt.tex`
snapshot referenced in `manuscript/README.md`; Overleaf is the authoring source.
Manuscript snapshots are copied manually at milestones.

## Scope and state

This repository is being prepared for a fresh rerun of the experiments retained
in the shorter ALT paper. Historical implementations are starting points.
`experiments/alt2027/inventory.json` distinguishes manuscript settings, candidate
implementations, missing work, and unresolved protocol choices. Preserve that
distinction when updating status or reporting progress.

The historical arXiv code and reproduction protocol are preserved at tag
`arxiv-code-2026-08`. Old paper numbers are historical comparisons. Production
acceptance comes from analytic identities, independent numerical checks, and the
agreed accuracy and sampling criteria.

## Scientific workflow

1. Agree on the manuscript scope and fill the outstanding protocol decisions.
2. Implement and independently validate every method in that protocol.
3. Freeze the protocol, code, environment, dataset, and numerical-store identity.
4. Use saved common samples for paired comparisons. Keep per-trial losses and
   relevant posterior weights, together with the seed scheme.
5. Generate figures and tables from those records; integrate them in Overleaf.
6. Archive source and artifact manifests at the agreed milestones.

New trials, sample-size changes, model grids, or preprocessing changes belong in
an explicit protocol revision. Keep smoke, validation, and production outputs in
separate run directories. A failed numerical check stops the affected campaign.
Do not overwrite completed run records.

## Numerical code

- Preserve the independent reference implementation (`mixture_weights.py`) and
  the table-based implementation (`layered.py`, `fast_tables.py`, `mellin.py`,
  `universal_tables.py`). Their coexistence supports cross-checks.
- Read `docs/numerics.md` before reusing the `product_model_with_memory` engine.
  Explicitly implement the paper's depth-zero atom and declared depth grids;
  check units, mixture weights, kernel family, and truncation settings.
- Keep existing APIs and regression tests working. Several scripts import
  helpers from other scripts; check references before moving or deleting them.
- After numerical changes, run the full `scripts/validate_appendix_c.py` and
  `python -m pytest -q`, plus the new checks for the affected ALT domain, before
  accepting experimental outputs. Documentation-only changes need structural
  checks rather than numerical campaigns.
- Record evaluation errors in their actual units. Separate deterministic
  numerical tolerances from sampling uncertainty.

## Records and storage

Follow `artifacts/alt2027/README.md`. Track small specifications, manifests,
validation summaries, generated tables, and final figure assets. Keep numerical
stores and bulk trial arrays outside Git, with durable archive locations and
checksums recorded before a paper result is finalized. Local paths alone are
working locations, not public reproducibility links.

The canonical corpus is `data/kjv.txt.gz`; its checksum and tokenization identity
are in `data/kjv.manifest.json`. Preserve the bundled bytes. Treat other historical
Bible files as distinct datasets until hashes and preprocessing are reconciled.

## Manuscript practice

Keep the scientific text concise and preserve the agreed scope. State numerical
comparisons symmetrically and distinguish cumulative regret from next-symbol
loss. The central contribution combines a simple architecture, analytic
expressions, and broad empirical competitiveness. Discuss unresolved provenance
or verification work in project records and author discussions. The AI Disclosure
accurately states that the authors reviewed the proofs by hand; it does not claim
manual code review.
