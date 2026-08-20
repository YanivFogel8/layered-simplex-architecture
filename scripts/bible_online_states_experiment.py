#!/usr/bin/env python3
"""The online context-learning code of Section 5.4 (in text, last paragraph).

Fully sequential order-one code with NO context-set header: a context gets
its own state the moment its count so far crosses a threshold; until then
its successors are coded in the shared backoff state.  Which state a token
goes to depends only on the past, so the code is sequential and still
exactly computable: each state's emission subsequence is exchangeable, and
the total codelength is the sum of per-state batch codelengths (chain
rule), with every state carrying its own depth-averaged layered predictor.

The threshold itself is averaged over a small grid (uniform mixture over
sequential codes), as with the split-size grid of the semi-adaptive
scheme.  Because the mixture tracks its best member to within
log2(grid size)/n bits -- microscopically at corpus scale -- the reported
total is insensitive to the exact grid, provided the grid brackets the
best threshold.  Paper target: 7.206 bits per token, 0.11 above the
semi-adaptive 7.100.

Example (full corpus, paper setting):

    python scripts/bible_online_states_experiment.py --corpus data/kjv.txt \
        --d 100000 --l-max 60 --thresholds 16,32,64,128,256,512 \
        --jobs 2 --out output/online_states
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from lsa import load_tokens
from lsa.codelength import depth_averaged_codelength_profiles, profile_of


def online_state_profiles(tokens, theta):
    """One causal pass: per-state successor counters under threshold theta.

    Coding position t (t >= 2) has context c = tokens[t-2] (the previous
    token).  If c's occurrence count among tokens[0 .. t-2] has reached
    theta, c owns a state; otherwise the position codes in the backoff
    state.  Promotion is permanent because counts only grow.  Returns the
    per-state successor counters, the number of promoted states, and the
    promoted count at the halfway point (a diagnostic the paper quotes for
    the semi-adaptive top-512 list).
    """
    seen: Counter[str] = Counter()
    states: dict[str, Counter[str]] = defaultdict(Counter)
    promoted: set[str] = set()
    half = len(tokens) // 2
    promoted_at_half = 0
    seen[tokens[0]] += 1
    for t in range(1, len(tokens)):
        c = tokens[t - 1]
        # seen[] already includes tokens[0..t-1]; c's count there is causal
        if c in promoted or seen[c] >= theta:
            promoted.add(c)
            states[c][tokens[t]] += 1
        else:
            states["<backoff>"][tokens[t]] += 1
        seen[tokens[t]] += 1
        if t == half:
            promoted_at_half = len(promoted)
    return states, len(promoted), promoted_at_half


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--n", type=int, default=None,
                        help="use only the first n tokens (default: all)")
    parser.add_argument("--thresholds", default="16,32,64,128,256,512")
    parser.add_argument("--l-max", type=int, default=60)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"

    tokens = load_tokens(args.corpus)
    if args.n:
        tokens = tokens[: args.n]
    n_coded = len(tokens) - 1  # x_2 .. x_n, as in the semi-adaptive scheme
    grid = sorted(set(int(t) for t in args.thresholds.split(",")))

    t0 = time.time()
    per_theta_profiles: dict[int, list] = {}
    diagnostics: dict[int, dict] = {}
    for theta in grid:
        states, n_promoted, at_half = online_state_profiles(tokens, theta)
        per_theta_profiles[theta] = [
            profile_of(counter) for counter in states.values()
        ]
        diagnostics[theta] = {
            "promoted_states_final": n_promoted,
            "promoted_states_at_halfway": at_half,
        }
        print(
            f"  theta={theta}: {n_promoted} states promoted "
            f"({at_half} in place halfway) ({time.time() - t0:.0f}s)",
            flush=True,
        )

    distinct = {
        prof: prof
        for profiles in per_theta_profiles.values()
        for prof in profiles
    }
    print(
        f"evaluating {len(distinct)} distinct state profiles "
        f"(d={args.d}, L<={args.l_max})",
        flush=True,
    )

    def progress(event, _unused) -> None:
        kind, k, total = event
        if kind == "tables" and (k % 200 == 0 or k == total):
            print(f"  tables: {k}/{total} ({time.time() - t0:.0f}s)", flush=True)
        elif kind == "depth" and (k % 10 == 0 or k == total):
            print(f"  eval: depth {k}/{total} ({time.time() - t0:.0f}s)", flush=True)

    evaluated = depth_averaged_codelength_profiles(
        distinct, d=args.d, l_max=args.l_max,
        cache_dir=cache_dir, jobs=args.jobs, progress=progress,
    )

    member_log2_q = {
        theta: sum(evaluated[p].log2_q_avg for p in per_theta_profiles[theta])
        for theta in grid
    }
    logs = np.array([member_log2_q[t] for t in grid])
    family_log2_q = (
        -math.log2(len(grid))
        + math.log2(np.sum(np.exp((logs - logs.max()) * math.log(2.0))))
        + logs.max()
    )
    posterior = np.exp((logs - logs.max()) * math.log(2.0))
    posterior /= posterior.sum()

    rows = {
        theta: {
            "bits_per_token": -member_log2_q[theta] / n_coded,
            "posterior": float(w),
            **diagnostics[theta],
        }
        for theta, w in zip(grid, posterior)
    }
    total = -family_log2_q / n_coded
    print(f"{'theta':>7} {'bits/token':>11} {'posterior':>10} {'states':>7}")
    for theta in grid:
        r = rows[theta]
        print(
            f"{theta:>7} {r['bits_per_token']:>11.4f} "
            f"{r['posterior']:>10.4f} {r['promoted_states_final']:>7}",
            flush=True,
        )
    print(f"online threshold-averaged code: {total:.4f} bits/token "
          f"(paper: 7.206)", flush=True)

    payload = {
        "corpus": str(args.corpus),
        "n_tokens": len(tokens),
        "n_coded": n_coded,
        "d": args.d,
        "l_max": args.l_max,
        "threshold_grid": grid,
        "per_threshold": rows,
        "online_bits_per_token": total,
        "paper_value": 7.206,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=1))
    print(f"wrote {out_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
