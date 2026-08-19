#!/usr/bin/env python3
"""Symbol discovery and the scaling tests (Figures 3-4, Tables 2-3).

Consumes the regret grid produced by scripts/factorial_scaling_experiment.py
(the Section 4.1 factorial design: alphabet sizes x sample sizes at
c = c*) and produces:

- Figure 3 (fig3_discovery.pdf): exact expected discovery count
  (Equation 8), the large-sample estimate (Equation 10), and the mean and
  standard deviation over sampled profiles -- no regret data needed;
- Table 2 (table2.tsv): the alphabet-scaling test of Equation (12) -- for
  each alpha and N, regress the measured regret per discovered symbol on
  the description length per discovered symbol across alphabet sizes, and
  report slope and offset;
- Table 3 (table3.tsv) and Figure 4 (fig4_data_scaling.pdf): the
  data-scaling test -- measured exponents against the simple prediction
  1 - 1/alpha and the refined finite-size prediction of Equation (9) with
  the fixed offset B = alpha log2 e - 2.

    python scripts/scaling_report.py \
        --results output/factorial/factorial_results.csv --out output/scaling
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import gammaln, zeta

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data_scaling_experiment import read_results_csv  # noqa: E402
from depth_scaling_experiment import (  # noqa: E402
    expected_discovered,
    mean_log2_binom_d_kn,
    zipf,
)


def discovery_large_sample(n: float, alpha: float) -> float:
    """Equation (10): Gamma(1 - 1/alpha) (N / zeta(alpha))^(1/alpha) - 1/2."""

    return float(
        math.exp(gammaln(1.0 - 1.0 / alpha))
        * (n / zeta(alpha)) ** (1.0 / alpha) - 0.5
    )


def sampled_discovery(p, n, draws, rng):
    counts = []
    remaining = draws
    while remaining:
        batch = min(200, remaining)
        m = rng.multinomial(n, p, size=batch)
        counts.append((m > 0).sum(axis=1))
        remaining -= batch
    k = np.concatenate(counts)
    return float(k.mean()), float(k.std(ddof=1))


def fit_loglog_slope(x, y):
    slope, _ = np.polyfit(np.log(np.asarray(x, dtype=float)),
                          np.log(np.asarray(y, dtype=float)), 1)
    return float(slope)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True,
                        help="factorial_results.csv from "
                             "factorial_scaling_experiment.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--discovery-draws", type=int, default=5000,
                        help="profile draws for Figure 3's points")
    parser.add_argument("--description-draws", type=int, default=1000,
                        help="profile draws for E log2 C(d, K_N)")
    parser.add_argument("--seed", type=int, default=20260403)
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_results_csv(Path(args.results))
    alphas = sorted({r.alpha for r in rows})
    d_values = sorted({r.d for r in rows})
    n_values = sorted({r.N for r in rows})
    regret = {(r.alpha, r.d, r.N): r.regret_bits for r in rows}
    print(f"grid: alphas={alphas} d={d_values} N={n_values}")

    rng = np.random.default_rng(args.seed)

    # exact k_N and description length per cell
    k_exact: dict = {}
    x_desc: dict = {}
    for alpha in alphas:
        for d in d_values:
            p = zipf(d, alpha)
            for n in n_values:
                k_exact[(alpha, d, n)] = expected_discovered(p, n)
                x_desc[(alpha, d, n)] = (
                    mean_log2_binom_d_kn(p, n, args.description_draws, rng)
                    / k_exact[(alpha, d, n)]
                )

    # ----- Figure 3: symbol discovery -----
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.0))
    for ax, alpha in zip(axes.flat, alphas):
        for d in d_values:
            p = zipf(d, alpha)
            exact = [expected_discovered(p, n) for n in n_values]
            ax.plot(n_values, exact, "-", linewidth=1.0,
                    label=f"d = {d:,}")
        estimate = [discovery_large_sample(n, alpha) for n in n_values]
        ax.plot(n_values, estimate, "k--", linewidth=1.4,
                label="large-sample estimate")
        p = zipf(max(d_values), alpha)
        sampled = [sampled_discovery(p, n, args.discovery_draws, rng)
                   for n in n_values]
        ax.errorbar(n_values, [s[0] for s in sampled],
                    yerr=[s[1] for s in sampled], fmt="o", markersize=3,
                    color="tab:olive", label=f"sampled, d = {max(d_values):,}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"$\\alpha$ = {alpha:g}", fontsize=9)
        ax.tick_params(labelsize=7)
        if alpha == alphas[0]:
            ax.legend(fontsize=6, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("sample size N", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("number of discovered symbols", fontsize=8)
    fig.suptitle("Symbol discovery", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_discovery.pdf", bbox_inches="tight")
    print(f"figure: {out_dir / 'fig3_discovery.pdf'}")

    # ----- Table 2: alphabet scaling -----
    lines = ["alpha\tN\tslope\toffset_B_bits"]
    print("\nTable 2 (alphabet scaling; slope and offset of y = x + B):")
    print(f"{'alpha':>6} {'N':>7} {'slope':>7} {'offset B':>9}")
    slopes_by_alpha = defaultdict(dict)
    for alpha in alphas:
        for n in n_values:
            xs = [x_desc[(alpha, d, n)] for d in d_values]
            ys = [n * regret[(alpha, d, n)] / k_exact[(alpha, d, n)]
                  for d in d_values]
            slope, intercept = np.polyfit(xs, ys, 1)
            slopes_by_alpha[alpha][n] = (float(slope), float(intercept))
            lines.append(f"{alpha:g}\t{n}\t{slope:.4g}\t{intercept:.4g}")
            print(f"{alpha:>6g} {n:>7,} {slope:>7.2f} {intercept:>9.2f}")
    (out_dir / "table2.tsv").write_text("\n".join(lines) + "\n")
    print(f"table: {out_dir / 'table2.tsv'}")

    # ----- Table 3 and Figure 4: data scaling -----
    lines = ["alpha\td\tsimple\trefined\tmeasured"]
    print("\nTable 3 (data-scaling exponents):")
    print(f"{'alpha':>6} {'d':>8} {'simple':>7} {'refined':>8} {'measured':>9}")
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.0))
    for ax, alpha in zip(axes.flat, alphas):
        simple = 1.0 - 1.0 / alpha
        b_offset = alpha * math.log2(math.e) - 2.0
        measured, refined = [], []
        for d in d_values:
            measured.append(
                -fit_loglog_slope(n_values,
                                  [regret[(alpha, d, n)] for n in n_values])
            )
            pred = [
                (x_desc[(alpha, d, n)] + b_offset)
                * k_exact[(alpha, d, n)] / n
                for n in n_values
            ]
            refined.append(-fit_loglog_slope(n_values, pred))
        for d, r_meas, r_ref in zip(d_values, measured, refined):
            lines.append(f"{alpha:g}\t{d}\t{simple:.3f}\t{r_ref:.3f}\t"
                         f"{r_meas:.3f}")
            if d in (min(d_values), max(d_values)):
                print(f"{alpha:>6g} {d:>8,} {simple:>7.3f} {r_ref:>8.3f} "
                      f"{r_meas:>9.3f}")
        ax.axhline(simple, color="0.4", linestyle=":",
                   label="simple prediction")
        ax.plot(d_values, refined, "s--", color="tab:orange",
                markersize=3.5, label="refined prediction")
        ax.plot(d_values, measured, "o-", color="tab:blue",
                markersize=3.5, label="measured")
        ax.set_xscale("log")
        ax.set_title(f"$\\alpha$ = {alpha:g}", fontsize=9)
        ax.tick_params(labelsize=7)
        if alpha == alphas[0]:
            ax.legend(fontsize=6.5, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("alphabet size d", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("data exponent", fontsize=8)
    fig.suptitle("Data scaling: predicted and measured exponents",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_data_scaling.pdf", bbox_inches="tight")
    (out_dir / "table3.tsv").write_text("\n".join(lines) + "\n")
    print(f"figure: {out_dir / 'fig4_data_scaling.pdf'}")
    print(f"table: {out_dir / 'table3.tsv'}")


if __name__ == "__main__":
    main()
