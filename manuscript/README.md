# Manuscript snapshots

Overleaf is the authoritative manuscript editor. This repository records
manual source snapshots at agreed milestones: the text freeze, the revision
after new experiments, and submission. There is no automatic Overleaf sync.

## ALT starting point: 7 October 2026

[`snapshots/2026-10-07-alt-start/`](snapshots/2026-10-07-alt-start/) contains
the revised shorter manuscript in the official anonymous ALT 2027 format.
The conversion preserves its scientific text and mathematics, bibliography,
and figure contents. It applies the ALT title and theorem conventions, renames
the AI statement to AI Disclosure, and removes the AISTATS checklist.

The main document is `main_shorter.tex`. The bibliography and figures have
separate names so the older arXiv source can remain in the same Overleaf project.
The official template's three class/support files are included. The build has
12 pages through the conclusion and 31 pages including disclosure, references,
and appendices. Existing experimental results are preserved in this starting
point; the planned reruns follow the authors' agreement on the text and protocol.

```sh
cd manuscript/snapshots/2026-10-07-alt-start
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main_shorter.tex
```

`snapshot.json` records the template source, source-file checksums, and build
checks. This is a starting-point snapshot, preceding the authors' text freeze.

## ArXiv baseline: 7 October 2026

[`snapshots/2026-10-07-arxiv/`](snapshots/2026-10-07-arxiv/) preserves the
downloaded `arxiv.tex`, `references.bib`, and all six referenced figures,
byte for byte. This is the historical arXiv manuscript, preceding the ALT
revision. `snapshot.json` records SHA-256 checksums for those source files.

The snapshot contains the manuscript's required files. Obsolete ICLR
templates, duplicate figures at the project root, old drafts, and experiment
archives are excluded from this manuscript snapshot.

To compile with a TeX Live installation containing the packages used by
the manuscript:

```sh
cd manuscript/snapshots/2026-10-07-arxiv
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build arxiv.tex
```

The main document is `arxiv.tex`. The compiled PDF and auxiliary files go
into the ignored `build/` directory. The PDF files in `figures/` are required
source assets for the manuscript.

## Later snapshots

After the authors agree on a milestone version in Overleaf, export its
source and preserve the required manuscript files in a new dated snapshot.
Record the experiment run identifiers and code revision used for new
results in that snapshot's manifest. Keep large numerical stores and raw
result archives outside the manuscript snapshot, referenced by manifests
and checksums.
