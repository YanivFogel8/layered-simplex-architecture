# Manuscript snapshots

Overleaf is the authoritative manuscript editor. This repository records manual
source snapshots at agreed milestones. There is no automatic Overleaf sync.

## ALT starting point: 7 October 2026

[`snapshots/2026-10-07-alt-start/`](snapshots/2026-10-07-alt-start/) contains the
revised shorter manuscript in the official anonymous ALT 2027 format.

- Main document: **`alt.tex`**.
- Bibliography: **`references_alt.bib`**.
- Graphics: `figures/alt/`.
- Required ALT template files: `alt2027.cls`, `jmlr.cls`, `jmlrutils.sty`.

The conversion preserves scientific text, mathematics, bibliography contents, and
figure contents. It applies ALT title/theorem conventions, renames the AI
statement to AI Disclosure, and removes the AISTATS checklist. A subsequent quick
layout pass improves float placement, lets the disclosure and references follow
the conclusion, and allows the validation table to break with repeated headers.

The conclusion is on page 12; disclosure/references begin on that page and end
on page 14. Appendices start on page 15. The complete PDF has 30 pages.
Existing results are preserved for the authors' text/protocol agreement and the
planned reruns.

```sh
cd manuscript/snapshots/2026-10-07-alt-start
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build alt.tex
```

`snapshot.json` records the template source, source-file checksums, and build
verification. Tag `alt-start-2026-10-07` preserves the first conversion before the
requested filenames and page-break improvements. Git history records those
same-day preparation updates; this starting point precedes the text freeze.

## ArXiv baseline: 7 October 2026

[`snapshots/2026-10-07-arxiv/`](snapshots/2026-10-07-arxiv/) preserves the
historical arXiv manuscript and its six referenced figures.

- Main document: **`arxiv.tex`**.
- Bibliography: **`references_arxiv.bib`**.
- Graphics: `figures/`.

The bibliography filename and its source command were renamed on request;
scientific content and figure bytes are unchanged. Tag `arxiv-source-2026-10-07`
preserves the original byte-for-byte export. The manifest records current hashes.

```sh
cd manuscript/snapshots/2026-10-07-arxiv
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build arxiv.tex
```

Compiled PDFs and auxiliary files go into the ignored `build/` directory. PDF
files under the figure folders are required source assets. Obsolete ICLR files,
unused drafts, and experiment archives are excluded from these snapshots.

## Later snapshots

After an agreed milestone in Overleaf, export its source and preserve the
required files in a new dated snapshot. Record the experiment run identifiers
and code revision used for revised results. Keep bulk numerical stores and raw
trial archives outside manuscript snapshots, referenced through manifests and
checksums in the artifact records.
