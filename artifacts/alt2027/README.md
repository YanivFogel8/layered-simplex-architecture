# ALT experiment artifacts

Each completed run has a unique identifier and a record linking its outputs to
its exact inputs. Start from `run-manifest.template.json`; fill measured values
when the run is executed. The template is a specification, not a completed run.

## Storage

- `output/alt2027/<run_id>/`: working outputs, saved sample profiles, per-trial
  losses, posterior weights, diagnostics, and logs. These are ignored by Git.
- `artifacts/alt2027/runs/<run_id>.json`: tracked run manifest after execution.
- `artifacts/alt2027/validation/`: compact validation reports tied to a code
  revision, numerical-store identity, and tested parameter domain.
- `artifacts/alt2027/paper/`: generated tables and final figure assets selected
  for the manuscript, with a manifest mapping them to their source runs and
  report-generation commands.

Create those subdirectories as records become available. Archive bulk records
outside Git and record their durable location and SHA-256 before finalizing the
paper. Every listed artifact includes its size and checksum; a local working path
can also be retained for convenience.

## Required run record

Record the protocol revision and hash, experiment ID, purpose (smoke, validation,
or production), timestamps, exact command, clean code commit, engine revision and
relevant source hashes, environment lock and installed package versions, machine
and resource settings, corpus/sample hashes, seed scheme, model grids, quadrature
and interpolation settings, store manifest hash and a content-hashed store
archive (or per-file hash index), truncation settings, logs, and
success/failure status.

Synthetic comparisons use common saved samples across methods. Preserve target
probabilities or enough information to reconstruct them, counts/profiles,
per-trial losses, paired differences, and relevant posterior weights. Distinguish
random target draws from sample draws. Store failed trials and numerical
warnings alongside successful ones; reruns receive a new run ID and link back.

Report-generation records identify their input runs, command, code revision,
aggregation and uncertainty calculation, and output checksums. Published figures
and tables are generated from those records rather than edited by hand.
