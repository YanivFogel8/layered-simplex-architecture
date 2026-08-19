#!/usr/bin/env python3
"""Provide the King James Bible corpus at data/kjv.txt (paper, Section 5.3).

The corpus is the entire King James Bible, Project Gutenberg eBook #10
("The King James Version of the Bible", plain text), with verse numbers
removed and the text split into word-and-punctuation tokens.  This script
downloads the source file (or uses a local copy), applies the exact
preprocessing below, writes the token stream to ``data/kjv.txt`` (one
space-separated token stream readable by ``lsa.corpus.load_tokens``), and
verifies the token statistics.

Preprocessing, in order:

1. Keep only the text between the ``*** START OF ...`` and
   ``*** END OF ...`` Project Gutenberg markers (the table of contents at
   the head of the body is part of the corpus; the license boilerplate is
   not).
2. Remove verse numbers: every ``chapter:verse`` digit pair (``1:1``,
   ``22:21``, ...).
3. Possessives: a curly apostrophe followed by ``s`` (``brother’s``) becomes
   the single token ``’s``; a trailing curly apostrophe (``sons’``) is
   dropped.
4. Drop the few stray ``*`` and ``-`` characters (a testament separator
   and 19 hyphenated names; the hyphen parts remain as words).
5. Tokenize: a token is a maximal run of letters ``[A-Za-z]+``, the
   possessive marker ``’s``, or a single punctuation character.  Case is
   preserved.

This yields 915,860 tokens over 13,550 distinct types — exactly the
corpus of the paper (its August 2026 revision adopted this recipe as
canonical).  The per-prefix distinct-type counts of Table 7 are
1,160 / 2,119 / 3,839 / 6,922, as printed.

Order of preference (mirrors scripts/get_text8.py of the source project):

  1. data/kjv.txt already present and verified;
  2. data/kjv.txt.gz bundled in the repository (works offline; this is a
     public-domain text, and the bundled copy pins the exact edition);
  3. --from-file, a local copy of the Gutenberg source file;
  4. download from gutenberg.org.

Usage:

    python scripts/get_kjv.py                       # bundled copy/download
    python scripts/get_kjv.py --from-file pg10.txt  # use a local source
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
import urllib.request
from pathlib import Path

URLS = [
    "https://www.gutenberg.org/cache/epub/10/pg10.txt",
    "https://www.gutenberg.org/files/10/10-0.txt",
]

EXPECTED_TOKENS = 915_860
EXPECTED_TYPES = 13_550
PAPER_TOKENS = 915_849
PREFIX_TYPES = {10_000: 1_160, 30_000: 2_119, 100_000: 3_839, 300_000: 6_922}


def fetch(url: str) -> str:
    print(f"downloading {url} ...")
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8")


def preprocess(raw: str) -> list[str]:
    start = raw.index("*** START OF THE PROJECT GUTENBERG EBOOK")
    start = raw.index("\n", start) + 1
    end = raw.index("*** END OF THE PROJECT GUTENBERG EBOOK")
    body = raw[start:end]
    body = re.sub(r"\d+:\d+", " ", body)      # verse numbers
    body = re.sub(r"’(?!s)", " ", body)  # trailing curly apostrophes
    body = body.replace("*", " ").replace("-", " ")
    return re.findall(r"[A-Za-z]+|’s|[^\sA-Za-z]", body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-file", default=None,
                        help="local pg10.txt instead of downloading")
    parser.add_argument("--out", default=None,
                        help="output path (default: data/kjv.txt)")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    out = Path(args.out) if args.out else repo / "data" / "kjv.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    bundled = repo / "data" / "kjv.txt.gz"

    if out.exists() and not args.from_file:
        tokens = out.read_text().split()
    elif bundled.exists() and not args.from_file:
        print(f"decompressing bundled {bundled} ...")
        out.write_bytes(gzip.decompress(bundled.read_bytes()))
        tokens = out.read_text().split()
    else:
        if args.from_file:
            raw = Path(args.from_file).read_text(encoding="utf-8")
        else:
            raw = None
            for url in URLS:
                try:
                    raw = fetch(url)
                    break
                except Exception as e:  # noqa: BLE001 - try the next mirror
                    print(f"  failed ({e})")
            if raw is None:
                print("could not download eBook #10; pass --from-file")
                return 1
        tokens = preprocess(raw)
        out.write_text(" ".join(tokens))

    n, types = len(tokens), len(set(tokens))
    print(f"tokens {n:,} (expected {EXPECTED_TOKENS:,}; "
          f"paper {PAPER_TOKENS:,}), types {types:,} "
          f"(expected {EXPECTED_TYPES:,})")
    ok = n == EXPECTED_TOKENS and types == EXPECTED_TYPES
    for k, expected in PREFIX_TYPES.items():
        got = len(set(tokens[:k]))
        print(f"  distinct types in first {k:>7,}: {got:,} "
              f"(expected {expected:,})")
        ok = ok and got == expected
    print(f"file: {out}")
    if not ok:
        print("WARNING: counts differ from the expected values; the "
              "Gutenberg source file may have changed since this script "
              "was written.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
