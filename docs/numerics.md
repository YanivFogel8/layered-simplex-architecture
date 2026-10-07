# Numerical engine integration for ALT

The next implementation phase will reuse the scientific evaluator in the sibling
`product_model_with_memory` project. The existing `src/lsa/` implementation stays
available for regression and independent checks. The preparation step records the
integration requirements; it does not replace either numerical implementation.

## Inspected source

On 7 October 2026 the sibling project was on `main` at
`240406d16be0e7c5dcd7d2ee0e14d5ee4f28c915`. Its relevant evaluator/store files
were clean relative to that commit, while other documentation and research files
had local changes. Capture hashes of the actual scientific files used during
integration, together with the code commit and working-tree state.

Useful entry points under that project's `src/product_model_with_memory/`:

| File/API | Role |
|---|---|
| `layered.py: log_q_lambda_scan` | Evidence of a count profile at one depth, with peak/convergence diagnostics |
| `layered.py: log_q_lambda_closed_l1` | Independent depth-one endpoint |
| `codelength.py: depth_averaged_codelength_profiles` | Batched profile evaluation; adapt its mixture convention |
| `codelength.py: depth_averaged_codelength_families` | Related base/augmented profiles for predictive evidence ratios |
| `universal_tables.py: UniversalTables` | Stored kernel rows, level tables, interpolation, and store checks |
| `mellin.py` | Contour/reference evaluation |

Confirm names against the pinned version when implementing. The scientific
count-profile layer serves the memoryless synthetic and word-token experiments
directly. The project's production BPE wrappers, memory models, and transformer
modules are outside this adapter.

## Match the paper's models explicitly

- The engine's current aggregate averages `L=1,...,Lmax`. The ALT mixture includes
  `L=0,...,Lmax`. Add the uniform atom with natural-log evidence `-N ln(d)`, use
  `Lmax+1` equal prior weights, and recompute mixture evidence and posterior
  probabilities. Pass each experiment's depth range explicitly.
- Prediction is an evidence ratio for base and augmented profiles. Validate its
  normalization and agreement with direct prediction before applying numerical
  normalization corrections.
- A single powered layer uses `Y=E^w`, not a product of `w` exponential layers.
  Implement a separately identified powered kernel, including analytic `w=0`
  and `w=1` endpoints. Reuse the outer evidence/profile machinery where valid.
- Record whether quantities are nats, bits, totals, or per-symbol averages.

## Store and numerical settings

The inspected engine's documented production path is `tables/anchors_prod`;
pass it explicitly. Its low-level default is an older store. The documented
settings are `PMM_PHI_LADDER_EVERY=1`, `PMM_PHI_LADDER_DEGREE=11`,
`PMM_PHI_SADDLE_MIN_L=54`, and `PMM_SCAN_LEGACY=0`. These are candidate settings
for validation, not an ALT accuracy certificate.

Disable heuristic depth-tail truncation during verification with
`PMM_NO_TRUNCATE=1`. Record all active `PMM_*` settings, scan mode, native/Python
path, quadrature rules, tolerances, and store hashes. Hash the store contents
through an archive or a per-file index as well as its metadata manifest. The production store has
`anchors.json`; the separate `tables/probe_exact` contour store is read-only.
At inspection, both store manifests had empty certification arrays. Preserve
existing stores and use an explicit independently checked reference path.

## Validation before production

Check analytic endpoints, evidence/prediction identities, full and sparse scans,
truncated and full depth sweeps, native and Python paths, and selected independent
contour evaluations. Retain narrow-peak and integration-boundary diagnostics;
the aggregate API currently drops messages that are needed for this assessment.

Cover the actual experiment domain, including zero and singleton counts, heavy
counts, interpolation boundaries, the fixed-power grid through 80, and the depth
spectrum through 138. Existing historical profile checks are useful starting
cases. Acceptance tolerances are to be agreed in the protocol in the units of
the reported results and relative to their statistical uncertainty.

Useful existing checks include the engine's `test_layered`, `test_mellin`,
`test_universal_integration`, `test_store_integrity`, `test_interp_kernel`, and
`scripts/compare_evaluators.py`. Pass store paths explicitly. Record the versions
and parameter domain actually tested.

## Environment

Neither inspected project currently has a locked numerical environment. Pin the
validated Python/library versions and record compiler, operating system, CPU,
worker counts, and nested-thread limits. Importing the engine can build/load its
optional C kernel, so capture the actual native-path availability. Preserve a
portable environment specification alongside the platform-specific run record.
