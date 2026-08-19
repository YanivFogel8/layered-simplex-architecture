#!/usr/bin/env python3
"""One draw of the layered simplex architecture (paper, Figure 1).

Draws L independent uniform simplex points at alphabet size d, multiplies
them coordinatewise, renormalizes, and plots the layers and the resulting
sparse, heavy-tailed draw on a common vertical scale.  The dotted line
traces the winning coordinate.

    python scripts/fig1_prior_draws.py --d 24 --l 4 --out output/fig1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=24)
    parser.add_argument("--l", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    layers = rng.dirichlet(np.ones(args.d), size=args.l)
    theta = layers.prod(axis=0)
    theta /= theta.sum()
    winner = int(np.argmax(theta))

    rows = args.l + 1
    fig, axes = plt.subplots(rows, 1, figsize=(6.4, 1.05 * rows),
                             sharex=True, sharey=True)
    x = np.arange(args.d)
    for ell in range(args.l):
        axes[ell].bar(x, layers[ell], color="0.55", width=0.7)
        axes[ell].set_ylabel(f"$U^{{({ell + 1})}}$", rotation=0,
                             labelpad=18, fontsize=9, va="center")
    axes[-1].bar(x, theta, color="black", width=0.7)
    axes[-1].set_ylabel(r"$\theta$", rotation=0, labelpad=18,
                        fontsize=9, va="center")
    for ax in axes:
        ax.axvline(winner, color="0.3", linestyle=":", linewidth=0.9)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[-1].set_xlabel(f"d = {args.d} coordinates, one per symbol; "
                        f"L = {args.l} layers multiplied and renormalized",
                        fontsize=8)
    fig.tight_layout()
    path = out_dir / "fig1_prior_draw.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"figure: {path}")
    print(f"winning coordinate: {winner} "
          f"(mass {theta[winner]:.3f}; uniform would be {1/args.d:.3f})")


if __name__ == "__main__":
    main()
