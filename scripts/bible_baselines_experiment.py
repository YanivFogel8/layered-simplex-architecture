#!/usr/bin/env python3
"""Classical baselines on the Bible corpus (Table 7 / Figure 7a).

Codes the token stream with the classical estimators of Section 5:
add-one, add-half (KT), Braess--Sauer, Ristad's natural law, and the
Good--Turing + empirical hybrid, over the fixed d-symbol alphabet.  Every
number is the codelength of an honest sequential code: predict the next
token, suffer the log-loss, update.  For the exchangeable add-constant
rules the accumulated log-loss equals the closed-form batch codelength
(chain rule; the identity is verified numerically in the test suite), so
those two columns can also be computed directly from the counts.

    python scripts/bible_baselines_experiment.py --corpus data/kjv.txt \
        --d 100000 --checkpoints 10000,30000,100000,300000,all \
        --out output/bible_baselines

The LSA columns of Table 7 come from scripts/unigram_experiment.py;
scripts/bible_report.py assembles the full table and Figure 7.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lsa.corpus import load_tokens
from lsa.estimators import sequential_codelength_bits

METHODS = ["add_one", "add_half", "braess_sauer", "ristad", "good_turing"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--d", type=int, required=True,
                        help="alphabet size (paper: 100000)")
    parser.add_argument("--checkpoints", required=True,
                        help="comma-separated prefix lengths ('all' allowed)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens = load_tokens(args.corpus)
    vocabulary: dict[str, int] = {}
    ids = np.fromiter(
        (vocabulary.setdefault(t, len(vocabulary)) for t in tokens),
        dtype=np.int64, count=len(tokens),
    )
    if len(vocabulary) > args.d:
        raise SystemExit(
            f"corpus has {len(vocabulary)} types > alphabet d={args.d}"
        )
    checkpoints = [
        len(ids) if c.strip() == "all" else int(c)
        for c in args.checkpoints.split(",")
    ]

    # empirical unigram entropy of each coded prefix
    entropies = {}
    for n in checkpoints:
        counts = np.bincount(ids[:n])
        counts = counts[counts > 0]
        entropies[n] = float(
            -(counts / n * np.log2(counts / n)).sum()
        )

    t0 = time.time()
    rows: dict[str, dict[str, float]] = {
        str(n): {"n": n, "empirical_entropy_bits": entropies[n]}
        for n in checkpoints
    }
    for method in METHODS:
        bits = sequential_codelength_bits(
            ids, args.d, method, checkpoints=checkpoints
        )
        for n in checkpoints:
            rows[str(n)][method + "_bits_per_token"] = bits[n] / n
            rows[str(n)][method + "_redundancy"] = (
                bits[n] / n - entropies[n]
            )
        print(f"{method:>14}: " + "  ".join(
            f"{bits[n]/n - entropies[n]:.3f}@{n}" for n in checkpoints
        ) + f"   ({time.time()-t0:.0f}s)", flush=True)

    payload = {
        "corpus": args.corpus,
        "d": args.d,
        "checkpoints": checkpoints,
        "methods": METHODS,
        "note": "redundancy = sequential codelength per token minus the "
                "empirical unigram entropy of the coded prefix, in bits",
        "rows": rows,
        "seconds": time.time() - t0,
    }
    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"written: {out_file}", flush=True)


if __name__ == "__main__":
    main()
