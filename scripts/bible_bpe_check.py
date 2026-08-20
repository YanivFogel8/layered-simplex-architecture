#!/usr/bin/env python3
"""The byte-pair check of Section 5.3 (in text, end of the section).

Train a byte-pair (BPE) vocabulary on the corpus itself, encode the corpus
with it, and run the depth-averaged predictor on the resulting stream.  A
BPE tokenizer is built to flatten the token distribution, so the posterior
over depths should collapse onto L = 1 and all reasonable methods should
essentially tie -- the opposite end of the complexity spectrum from the
word stream, detected automatically from the data.

The paper states this check qualitatively and does not pin the vocabulary
size; ``--vocab-size`` is therefore a free parameter here, and the claim is
expected to hold across sizes (the manifest runs 1024, 4096, and 16384).

Training is standard word-level BPE (merges within whitespace tokens, with
an end-of-word marker), deterministic: ties in pair counts break
lexicographically, so a rerun reproduces the same vocabulary exactly.

Example:

    python scripts/bible_bpe_check.py --corpus data/kjv.txt \
        --vocab-size 4096 --checkpoints 10000,100000,all \
        --jobs 2 --out output/bpe_check_v4096
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from lsa import default_l_max, load_tokens
from lsa.codelength import depth_averaged_codelength_profiles, profile_of
from lsa.estimators import sequential_codelength_bits

END = "▁"  # end-of-word marker appended to a word's last symbol


def train_bpe(words: Counter[str], vocab_size: int) -> list[tuple[str, str]]:
    """Learn BPE merges on a word-frequency table until the vocabulary
    (base symbols + merged symbols) reaches ``vocab_size``.  Returns the
    merge list in order."""
    seqs: list[list[str]] = []
    freqs: list[int] = []
    for w, c in sorted(words.items()):
        s = list(w)
        s[-1] = s[-1] + END
        seqs.append(s)
        freqs.append(c)
    base = {sym for s in seqs for sym in s}
    merges: list[tuple[str, str]] = []
    while len(base) + len(merges) < vocab_size:
        pairs: Counter[tuple[str, str]] = Counter()
        for s, c in zip(seqs, freqs):
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += c
        if not pairs:
            break
        best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))[0]
        merges.append(best)
        a, b = best
        ab = a + b
        for s in seqs:
            i = 0
            while i < len(s) - 1:
                if s[i] == a and s[i + 1] == b:
                    s[i : i + 2] = [ab]
                else:
                    i += 1
    return merges


def encode(words: list[str], merges: list[tuple[str, str]]) -> list[str]:
    """Encode a word stream with the learned merges (cached per word)."""
    rank = {m: i for i, m in enumerate(merges)}
    cache: dict[str, list[str]] = {}
    out: list[str] = []
    for w in words:
        enc = cache.get(w)
        if enc is None:
            s = list(w)
            s[-1] = s[-1] + END
            while len(s) > 1:
                ranked = [
                    (rank.get((a, b)), i)
                    for i, (a, b) in enumerate(zip(s, s[1:]))
                    if (a, b) in rank
                ]
                if not ranked:
                    break
                _, i = min(ranked)
                s[i : i + 2] = [s[i] + s[i + 1]]
            cache[w] = enc = s
        out.extend(enc)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument(
        "--checkpoints", default="10000,100000,all",
        help="prefix lengths of the SUBWORD stream ('all' allowed)",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--l-max", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"

    words = load_tokens(args.corpus)
    t0 = time.time()
    merges = train_bpe(Counter(words), args.vocab_size)
    stream = encode(words, merges)
    vocab = sorted(set(stream))
    ids = {s: i for i, s in enumerate(vocab)}
    d = len(vocab)
    id_stream = [ids[s] for s in stream]
    print(
        f"BPE: vocab {d} (target {args.vocab_size}), stream "
        f"{len(stream):,} subwords from {len(words):,} words "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    checkpoints = [
        len(stream) if c.strip() == "all" else int(c)
        for c in args.checkpoints.split(",")
    ]
    l_max = args.l_max or default_l_max(d)

    counts = {n: Counter(stream[:n]) for n in checkpoints}
    profiles = {n: profile_of(counts[n]) for n in checkpoints}
    entropies = {
        n: -sum(
            c / n * np.log2(c / n) for c in counts[n].values()
        )
        for n in checkpoints
    }

    results = depth_averaged_codelength_profiles(
        profiles, d=d, l_max=l_max, cache_dir=cache_dir, jobs=args.jobs,
    )

    classical = {}
    for method in ("add_one", "add_half", "good_turing", "ristad"):
        cumulative = sequential_codelength_bits(
            id_stream, d, method, checkpoints=checkpoints
        )
        classical[method] = {n: bits / n for n, bits in cumulative.items()}

    rows = []
    print(f"{'n':>9} {'H_n':>7} {'LSA':>7} {'L*':>4} {'w(L*)':>7} "
          f"{'add1':>7} {'KT':>7} {'GT':>7} {'Ristad':>7}", flush=True)
    for n in checkpoints:
        r = results[n]
        mode_weight = max(r.posterior)
        row = {
            "n": n,
            "d": d,
            "empirical_entropy_bits": entropies[n],
            "lsa_avg_bits_per_token": r.bits_per_token,
            "posterior_mode_depth": r.posterior_mode,
            "posterior_mode_weight": mode_weight,
            "classical_bits_per_token": {
                m: classical[m][n] for m in classical
            },
        }
        rows.append(row)
        print(
            f"{n:>9,} {entropies[n]:>7.3f} {r.bits_per_token:>7.3f} "
            f"{r.posterior_mode:>4} {mode_weight:>7.4f} "
            f"{classical['add_one'][n]:>7.3f} "
            f"{classical['add_half'][n]:>7.3f} "
            f"{classical['good_turing'][n]:>7.3f} "
            f"{classical['ristad'][n]:>7.3f}",
            flush=True,
        )

    payload = {
        "corpus": str(args.corpus),
        "vocab_size_target": args.vocab_size,
        "vocab_size_actual": d,
        "n_merges": len(merges),
        "stream_length": len(stream),
        "l_max": l_max,
        "rows": rows,
        "claim": (
            "posterior collapses onto L = 1 and all reasonable methods tie "
            "(paper, Section 5.3, closing check)"
        ),
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=1))
    print(f"wrote {out_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
