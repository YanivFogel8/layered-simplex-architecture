#!/usr/bin/env python3
"""Order-one models over a nested family of state maps (paper, Section 5.4).

States are the M most frequent context tokens plus a single backoff state
for all other histories.  Conditioned on the state, the successor
sub-sequence is exchangeable, so the mixture factorizes over states and the
total codelength is a sum of per-state codelengths, each computed exactly
from counts by the machinery of Section 3.1.  Every state carries its own
depth-averaged predictor.  The split count M is itself averaged over the
grid given by --m-grid (one more level of hierarchical averaging), and the
posterior over M is reported.

Reproduces Table 8 of the paper (per-state LSA rows) and the in-text
split-count numbers of Section 5.4.  Example (Table 8, M = 512 column):

    python scripts/state_family_experiment.py \
        --corpus data/kjv.txt --d 100000 \
        --m-grid 0,64,128,256,512 --l-max 60 \
        --jobs 8 --out output/table8_m512

The --d flag fixes the emission alphabet (the paper uses d = 100,000, the
same fixed vocabulary as Section 5.3); tokens beyond the corpus vocabulary
simply never occur.  Without --d, the observed vocabulary size is used.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import gammaln, logsumexp

from lsa.corpus import load_tokens
from lsa.pairs import empirical_entropies, reduce_vocabulary
from lsa.state_family import state_family_codelengths


def _enumerative_subset_bits(alphabet_size: int, subset_size: int) -> float:
    """Bits to name an unordered subset: log2 C(alphabet_size, subset_size).

    Section 5.4 ("Describing the context set") charges this two-part-code
    header for naming the M split contexts out of the d-token vocabulary.
    """
    if subset_size < 0 or subset_size > alphabet_size:
        raise ValueError("invalid subset size")
    return float(
        (
            gammaln(alphabet_size + 1)
            - gammaln(subset_size + 1)
            - gammaln(alphabet_size - subset_size + 1)
        ) / np.log(2.0)
    )


def _two_part_family_bits(
    member_data_bits: dict[int, float],
    member_description_bits: dict[int, float],
) -> float:
    """Uniform mixture over individually decodable two-part member codes."""

    grid = sorted(member_data_bits)
    if not grid or set(grid) != set(member_description_bits):
        raise ValueError("member and description grids must agree and be nonempty")
    total_bits = np.asarray([
        member_data_bits[m] + member_description_bits[m] for m in grid
    ], dtype=float)
    return float(
        -logsumexp(-total_bits * np.log(2.0)) / np.log(2.0)
        + np.log2(len(grid))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True,
                        help="whitespace-tokenized corpus file")
    parser.add_argument("--d", type=int, default=None,
                        help="fixed emission-alphabet size (paper: 100000); "
                             "default: the observed vocabulary size")
    parser.add_argument("--m-grid", required=True,
                        help="comma-separated top-M state counts, e.g. "
                             "0,64,128,256,512 (0 = memoryless)")
    parser.add_argument("--n", type=int, default=None,
                        help="use only the first n tokens")
    parser.add_argument("--l-max", type=int, default=None,
                        help="depth ceiling per state (Table 8 uses 40 for "
                             "the M<=256 grid and 60 for the M<=512 grid)")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    m_grid = sorted({int(x) for x in args.m_grid.split(",")})

    t0 = time.time()
    tokens = load_tokens(args.corpus)
    if args.n:
        tokens = tokens[: args.n]
    # Map tokens to integer ids in frequency order (id 0 = most frequent).
    # No vocabulary cap: the corpus vocabulary embeds into the fixed
    # d-symbol alphabet.
    reduced, vocab = reduce_vocabulary(tokens, len(set(tokens)))
    observed_v = len(vocab)
    V = args.d if args.d is not None else observed_v
    if V < observed_v:
        raise SystemExit(
            f"--d {V} is smaller than the observed vocabulary {observed_v}"
        )
    if not m_grid or any(m < 0 or m > V for m in m_grid):
        raise SystemExit(f"every M must satisfy 0 <= M <= V={V}")

    ent = empirical_entropies(reduced)
    print(
        f"n={len(reduced):,} observed types={observed_v:,} alphabet d={V:,} "
        f"family M in {m_grid}  targets: "
        f"H_unigram={ent['unigram_bits']:.3f} "
        f"H(next|prev)={ent['conditional_bits']:.3f}",
        flush=True,
    )

    def progress(event, _unused) -> None:
        kind, k, total = event
        if kind == "tables" and (k % 100 == 0 or k == total):
            print(f"  tables: {k}/{total} orders built ({time.time()-t0:.0f}s)",
                  flush=True)
        elif kind == "depth" and (k % 5 == 0 or k == total):
            print(f"  evaluation: depth {k}/{total} done ({time.time()-t0:.0f}s)",
                  flush=True)

    out = state_family_codelengths(
        reduced,
        vocabulary_size=V,
        m_grid=m_grid,
        l_max=args.l_max,
        cache_dir=cache_dir,
        jobs=args.jobs,
        progress=progress,
    )

    print(f"\n{'M':>6} {'states':>7} {'bits/token':>11} {'posterior':>10}",
          flush=True)
    for m in out["m_grid"]:
        print(
            f"{m:>6} {out['member_states_observed'][m]:>7} "
            f"{out['member_bits_per_token'][m]:>11.4f} "
            f"{out['posterior_over_m'][m]:>10.2e}",
            flush=True,
        )
    print(
        f"family mixture: {out['family_bits_per_token']:.4f} bits/token "
        f"(best member M={out['best_member']}: "
        f"{out['member_bits_per_token'][out['best_member']]:.4f}); "
        f"targets H_unigram={ent['unigram_bits']:.3f}, "
        f"H(next|prev)={ent['conditional_bits']:.3f}",
        flush=True,
    )

    # Section 5.4, "Describing the context set": the two-part header that
    # names the M split contexts out of the d-token vocabulary.
    n_coded = out["n_coded"]
    context_set_bits = {
        m: _enumerative_subset_bits(V, m) for m in out["m_grid"]
    }
    semi_adaptive = {
        str(m): out["member_bits_per_token"][m]
        + context_set_bits[m] / n_coded
        for m in out["m_grid"]
    }
    largest = max(out["m_grid"])
    print(
        f"context-set header at M={largest}: "
        f"{context_set_bits[largest]:,.0f} bits "
        f"({context_set_bits[largest] / n_coded:.4f} bits/token); "
        f"semi-adaptive total "
        f"{semi_adaptive[str(largest)]:.4f} bits/token",
        flush=True,
    )

    payload = {
        "corpus": args.corpus,
        "d": V,
        "observed_types": observed_v,
        "n_tokens": len(reduced),
        "empirical": ent,
        **{k: v for k, v in out.items()},
        "context_set_description_bits": {
            str(m): context_set_bits[m] for m in out["m_grid"]
        },
        "semi_adaptive_bits_per_token": semi_adaptive,
        "seconds": time.time() - t0,
    }
    # JSON keys must be strings
    for key in ("member_bits_per_token", "member_states_observed",
                "posterior_over_m"):
        payload[key] = {str(k): v for k, v in payload[key].items()}
    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"written: {out_file} ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
