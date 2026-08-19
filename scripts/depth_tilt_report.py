#!/usr/bin/env python3
"""Render Figure 2 and Table 1 from depth_tilt_experiment.py results.

    python scripts/depth_tilt_report.py \
        --results output/fig2_d1e3/results.json output/fig2_d1e4/results.json \
                  output/fig2_d1e6/results.json \
        --out output/fig2_report

Writes fig2_depth_tilt.pdf and table1.tsv.  Table 1 lists the best tested
depth per target for every panel; for the last (largest-d) panel it also
prints the envelope and L = 1 rows in bits/symbol, as in the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def representative_depths(l_max: int) -> list[int]:
    """A readable subset of depths, geometrically spaced up to l_max."""

    depths = {1}
    value = 1.0
    while value < l_max:
        value *= 1.75
        depths.add(min(l_max, int(round(value))))
    return sorted(depths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True,
                        help="one results.json per panel, in display order")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = [json.loads(Path(p).read_text()) for p in args.results]

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(3.6 * len(panels), 3.2))
    axes = np.atleast_1d(axes)
    for ax, panel in zip(axes, panels):
        alphas = [r["alpha"] for r in panel["results"]]
        l_max = panel["l_max"]
        shown = representative_depths(l_max)
        cmap = plt.get_cmap("viridis")
        for i, L in enumerate(shown):
            mean = [r["regret_by_depth_mean"][L - 1]
                    for r in panel["results"]]
            err = [r["regret_by_depth_stderr"][L - 1]
                   for r in panel["results"]]
            ax.errorbar(alphas, mean, yerr=err, label=f"L = {L}",
                        color=cmap(i / max(1, len(shown) - 1)),
                        marker="o", markersize=2.5, linewidth=1.0)
        avg = [r["regret_avg_mean"] for r in panel["results"]]
        avg_err = [r["regret_avg_stderr"] for r in panel["results"]]
        ax.errorbar(alphas, avg, yerr=avg_err, label="mixture, all L",
                    color="tab:red", linestyle="--", linewidth=1.6)
        ax.set_yscale("log")
        ax.set_title(f"$d = 10^{{{int(round(np.log10(panel['d'])))}}}$,  "
                     f"$N = {panel['n']}$", fontsize=9)
        ax.set_xlabel(r"Zipf exponent $\alpha$ of the target", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=5.5, ncol=2, frameon=False)
    axes[0].set_ylabel(r"online regret $R_N$ [bits/symbol]", fontsize=8)
    fig.tight_layout()
    fig_path = out_dir / "fig2_depth_tilt.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"figure: {fig_path}")

    # ----- Table 1 -----
    lines = []
    alphas = [r["alpha"] for r in panels[0]["results"]]
    header = ["alpha", *[f"{a:g}" for a in alphas]]
    lines.append("\t".join(header))
    print("\nTable 1:")
    print(" ".join(f"{h:>8}" for h in header))
    for panel in panels:
        best = [1 + int(np.argmin(r["regret_by_depth_mean"]))
                for r in panel["results"]]
        row = [f"best L (d = {panel['d']:g})", *[str(b) for b in best]]
        lines.append("\t".join(row))
        print(f"{row[0]:>18} " + " ".join(f"{b:>8}" for b in best))
    last = panels[-1]
    envelope = [min(r["regret_by_depth_mean"]) for r in last["results"]]
    l1 = [r["regret_by_depth_mean"][0] for r in last["results"]]
    for label, vals in [(f"envelope (d = {last['d']:g})", envelope),
                        (f"L=1 (d = {last['d']:g})", l1)]:
        lines.append("\t".join([label, *[f"{v:.6g}" for v in vals]]))
        print(f"{label:>18} " + " ".join(f"{v:>8.2f}" for v in vals))
    gaps = []
    for panel in panels:
        gap = max(r["regret_avg_mean"] - min(r["regret_by_depth_mean"])
                  for r in panel["results"])
        bound = np.log2(panel["l_max"]) / panel["n"]
        gaps.append((panel["d"], gap, bound))
    print("\nmeasured worst-case average-vs-envelope gaps "
          "(bound log2(L_max)/N):")
    for d, gap, bound in gaps:
        print(f"  d = {d:g}: {gap:.6f} (bound {bound:.6f})")
    lines.append("")
    lines.append("\t".join(["gap (avg - envelope)",
                            *[f"{g:.6f} <= {b:.6f}" for _, g, b in gaps]]))
    (out_dir / "table1.tsv").write_text("\n".join(lines) + "\n")
    print(f"table: {out_dir / 'table1.tsv'}")


if __name__ == "__main__":
    main()
