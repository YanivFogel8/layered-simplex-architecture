#!/usr/bin/env python3
"""Depth tilts the complexity spectrum (paper, Figure 2 and Table 1).

For each alphabet size d, samples count profiles from Zipf targets over a
grid of exponents alpha and evaluates the exact online regret of every
single-depth mixture L = 1 .. L_max and of the uniform depth average, by
the procedure of Section 3.1: draw M ~ Multinomial(N, p) repeatedly with
fixed seeds, evaluate the exact formula on each draw, average.  Error bars
are the standard error of that profile average, the only statistical
uncertainty involved.

The paper's setting (three panels; the d = 10^6 panel dominates the cost):

    python scripts/depth_tilt_experiment.py --d 1000    --n 316  \
        --l-max 69  --out output/fig2_d1e3 --jobs 8
    python scripts/depth_tilt_experiment.py --d 10000   --n 1000 \
        --l-max 92  --out output/fig2_d1e4 --jobs 8
    python scripts/depth_tilt_experiment.py --d 1000000 --n 1000 \
        --l-max 138 --out output/fig2_d1e6 --jobs 8

Figure 2 and Table 1 are then rendered by scripts/depth_tilt_report.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Per-depth curves are quoted, so no depth's likelihood may be truncated
# as negligible.
os.environ.setdefault("LSA_NO_TRUNCATE", "1")

from lsa.codelength import depth_averaged_codelength_profiles  # noqa: E402
from lsa.metrics import entropy  # noqa: E402


def zipf(d: int, alpha: float) -> np.ndarray:
    w = np.arange(1, d + 1, dtype=float) ** (-alpha)
    return w / w.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--n", type=int, required=True,
                        help="sample size N (paper: 316 at d=10^3, else 1000)")
    parser.add_argument("--alphas", default="0,0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0")
    parser.add_argument("--l-max", type=int, required=True,
                        help="deepest depth (paper: c = L/ln d up to 10)")
    parser.add_argument("--profiles", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260401)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    alphas = [float(a) for a in args.alphas.split(",")]
    d, n, l_max = args.d, args.n, args.l_max

    t0 = time.time()
    per_alpha = []
    for a_index, alpha in enumerate(alphas):
        p = zipf(d, alpha)
        h = entropy(p)
        rng = np.random.default_rng([args.seed, d, a_index])
        profiles = {}
        for k in range(args.profiles):
            counts = rng.multinomial(n, p)
            profiles[k] = tuple(
                sorted((int(c) for c in counts[counts > 0]), reverse=True)
            )
        distinct = Counter(profiles.values())
        evaluated = depth_averaged_codelength_profiles(
            {prof: prof for prof in distinct},
            d=d, l_max=l_max, cache_dir=cache_dir, jobs=args.jobs,
        )
        # per-depth regret: R = -H(p) - (1/N) mean_profiles log2 q (eq. 4)
        samples_by_depth = np.array(
            [[evaluated[profiles[k]].log2_q_by_depth[L - 1]
              for L in range(1, l_max + 1)]
             for k in range(args.profiles)]
        )
        samples_avg = np.array(
            [evaluated[profiles[k]].log2_q_avg for k in range(args.profiles)]
        )
        regret_by_depth = -h - samples_by_depth / n
        regret_avg = -h - samples_avg / n
        result = {
            "alpha": alpha,
            "entropy_bits": h,
            "regret_by_depth_mean": [
                float(x) for x in regret_by_depth.mean(axis=0)
            ],
            "regret_by_depth_stderr": [
                float(x) for x in regret_by_depth.std(axis=0, ddof=1)
                / math.sqrt(args.profiles)
            ],
            "regret_avg_mean": float(regret_avg.mean()),
            "regret_avg_stderr": float(
                regret_avg.std(ddof=1) / math.sqrt(args.profiles)
            ),
            "distinct_profiles": len(distinct),
        }
        per_alpha.append(result)
        best = 1 + int(np.argmin(result["regret_by_depth_mean"]))
        print(
            f"[{time.time()-t0:7.0f}s] alpha={alpha:g}: best L={best} "
            f"(regret {min(result['regret_by_depth_mean']):.4g}), "
            f"avg {result['regret_avg_mean']:.4g}, L=1 "
            f"{result['regret_by_depth_mean'][0]:.4g}",
            flush=True,
        )

    payload = {
        "d": d, "n": n, "l_max": l_max, "profiles": args.profiles,
        "seed": args.seed, "alphas": alphas,
        "procedure": "Section 3.1: exact log2 q per sampled profile; "
                     "the profile average is the only statistical error",
        "seconds": time.time() - t0,
        "results": per_alpha,
    }
    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"written: {out_file} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
