#!/usr/bin/env python3
"""Depth scaling: regret per discovered symbol vs. depth (Figure 5, Table 4).

Fixes d and N, varies the depth coefficient c (L = round(c ln d)), and
measures the regret per discovered symbol P = N R_N / k_N, where k_N is
the exact expected number of discovered symbols (Equation 8).  R_N is
evaluated by the procedure of Section 3.1 (exact formula on sampled
profiles).  The prediction plotted alongside is Equation (14):

    P_pred(c) = rho(c) * x + B(alpha),      rho(c) = c I(1/c)  (c <= c*),
                                            rho(c) = 1         (c >= c*),

with I the Legendre transform of log Gamma(1 + s) (the large-deviation
cost of one layer's lift), x = E log2 C(d, K_N) / k_N the description
length per discovered symbol, and B(alpha) = alpha log2 e - 2 the
calibrated per-symbol offset used throughout Section 4.

The paper's setting:

    python scripts/depth_scaling_experiment.py --d 100000 --n 1000 \
        --alphas 2,3,4 --out output/fig5 --jobs 8

Writes results.json, fig5_depth_scaling.pdf and table4.tsv.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma, gammaln

os.environ.setdefault("LSA_NO_TRUNCATE", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lsa.codelength import depth_averaged_codelength_profiles  # noqa: E402
from lsa.metrics import entropy  # noqa: E402

EULER_GAMMA = float(np.euler_gamma)
C_STAR = 1.0 / (1.0 - EULER_GAMMA)


def zipf(d: int, alpha: float) -> np.ndarray:
    w = np.arange(1, d + 1, dtype=float) ** (-alpha)
    return w / w.sum()


def rho(c: float) -> float:
    """The price of one bit of description at depth coefficient c (eq. 14)."""

    if c >= C_STAR:
        return 1.0
    y = 1.0 / c
    # I(y) = sup_s [s y - log Gamma(1+s)]; the maximizer solves psi(1+s) = y
    upper = 10.0
    while digamma(1.0 + upper) < y:
        upper *= 2.0
    s = brentq(lambda t: digamma(1.0 + t) - y, 1e-9, upper, xtol=1e-12)
    return float(c * (s * y - gammaln(1.0 + s)))


def expected_discovered(p: np.ndarray, n: int) -> float:
    """k_N = sum_i [1 - (1 - p_i)^N], evaluated stably (eq. 8)."""

    return float(np.sum(-np.expm1(n * np.log1p(-p))))


def mean_log2_binom_d_kn(p: np.ndarray, n: int, draws: int,
                         rng: np.random.Generator) -> float:
    """E log2 C(d, K_N) over sampled profiles."""

    d = p.size
    total = 0.0
    remaining = draws
    while remaining:
        batch = min(200, remaining)
        counts = rng.multinomial(n, p, size=batch)
        k = (counts > 0).sum(axis=1)
        total += float(np.sum(
            (gammaln(d + 1) - gammaln(k + 1) - gammaln(d - k + 1))
            / math.log(2.0)
        ))
        remaining -= batch
    return total / draws


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=100_000)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--alphas", default="2,3,4")
    parser.add_argument("--c-values",
                        default="0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.37,2.75,"
                                "3,3.5,4,4.5,5,5.5,6")
    parser.add_argument("--profiles", type=int, default=40)
    parser.add_argument("--description-draws", type=int, default=1000,
                        help="profile draws for E log2 C(d, K_N)")
    parser.add_argument("--seed", type=int, default=20260402)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    d, n = args.d, args.n
    alphas = [float(a) for a in args.alphas.split(",")]
    c_values = sorted({float(c) for c in args.c_values.split(",")})
    ln_d = math.log(d)
    depth_of_c = {c: max(1, int(round(c * ln_d))) for c in c_values}
    l_max = max(depth_of_c.values())

    t0 = time.time()
    results = []
    for a_index, alpha in enumerate(alphas):
        p = zipf(d, alpha)
        h = entropy(p)
        rng = np.random.default_rng([args.seed, a_index])
        k_n = expected_discovered(p, n)
        x = mean_log2_binom_d_kn(p, n, args.description_draws, rng) / k_n
        b_offset = alpha * math.log2(math.e) - 2.0

        profiles = {}
        for k in range(args.profiles):
            counts = rng.multinomial(n, p)
            profiles[k] = tuple(
                sorted((int(c) for c in counts[counts > 0]), reverse=True)
            )
        evaluated = depth_averaged_codelength_profiles(
            {prof: prof for prof in set(profiles.values())},
            d=d, l_max=l_max, cache_dir=cache_dir, jobs=args.jobs,
        )
        rows = []
        for c in c_values:
            L = depth_of_c[c]
            samples = np.array([
                -h - evaluated[profiles[k]].log2_q_by_depth[L - 1] / n
                for k in range(args.profiles)
            ])
            regret = float(samples.mean())
            rows.append({
                "c": c,
                "L": L,
                "regret_bits": regret,
                "regret_stderr": float(
                    samples.std(ddof=1) / math.sqrt(args.profiles)
                ),
                "P_measured": n * regret / k_n,
                "P_predicted": rho(c) * x + b_offset,
            })
        best = min(rows, key=lambda r: r["P_measured"])
        at_cstar = min(rows, key=lambda r: abs(r["c"] - C_STAR))
        results.append({
            "alpha": alpha,
            "entropy_bits": h,
            "k_n": k_n,
            "description_per_symbol_bits": x,
            "b_offset_bits": b_offset,
            "rows": rows,
            "best_tested_c": best["c"],
            "P_cstar_over_P_min": at_cstar["P_measured"] / best["P_measured"],
        })
        print(
            f"[{time.time()-t0:6.0f}s] alpha={alpha:g}: k_N={k_n:.1f}, "
            f"x={x:.2f} bits, best c={best['c']:g} "
            f"(P={best['P_measured']:.2f}), "
            f"P(c*)/P_min={results[-1]['P_cstar_over_P_min']:.3f}",
            flush=True,
        )

    payload = {
        "d": d, "n": n, "c_star": C_STAR, "profiles": args.profiles,
        "seed": args.seed, "seconds": time.time() - t0, "results": results,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))

    # ----- Figure 5 -----
    fig, (ax_full, ax_log) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    colors = {2.0: "tab:blue", 3.0: "tab:orange", 4.0: "tab:green"}
    for res in results:
        alpha = res["alpha"]
        color = colors.get(alpha)
        cs = [r["c"] for r in res["rows"]]
        for ax, c_min in ((ax_full, 0.0), (ax_log, 0.75)):
            keep = [i for i, c in enumerate(cs) if c >= c_min]
            ax.plot([cs[i] for i in keep],
                    [res["rows"][i]["P_measured"] for i in keep],
                    "o", markersize=3.5, color=color,
                    label=f"measured, $\\alpha$ = {alpha:g}")
            ax.plot([cs[i] for i in keep],
                    [res["rows"][i]["P_predicted"] for i in keep],
                    "--", linewidth=1.1, color=color,
                    label=f"prediction, $\\alpha$ = {alpha:g}")
    for ax, title in ((ax_full, "Full depth range"),
                      (ax_log, "Logarithmic-depth range")):
        ax.axvline(C_STAR, color="0.4", linestyle=":", linewidth=1.0)
        ax.set_xlabel("depth coefficient c")
        ax.set_title(title, fontsize=9)
    ax_full.set_ylabel(r"cost per discovered symbol $N R_N / k_N$ [bits]")
    ax_log.legend(fontsize=6, frameon=False)
    fig.suptitle(f"Depth scaling at d = {d:g} and N = {n:g}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_depth_scaling.pdf", bbox_inches="tight")
    print(f"figure: {out_dir / 'fig5_depth_scaling.pdf'}")

    # ----- Table 4 -----
    lines = ["alpha\tbest tested c\tP(c*)/P_min"]
    print("\nTable 4:")
    print(f"{'alpha':>6} {'best c':>8} {'P(c*)/P_min':>12}")
    for res in results:
        lines.append(f"{res['alpha']:g}\t{res['best_tested_c']:g}\t"
                     f"{res['P_cstar_over_P_min']:.3f}")
        print(f"{res['alpha']:>6g} {res['best_tested_c']:>8g} "
              f"{res['P_cstar_over_P_min']:>12.3f}")
    (out_dir / "table4.tsv").write_text("\n".join(lines) + "\n")
    print(f"table: {out_dir / 'table4.tsv'}")


if __name__ == "__main__":
    main()
