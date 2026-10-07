"""Validation of the fixed-power tables (power_table.py, mb_table.py) used for the appendix.

    python validate_power.py [--procs 20]

Collects the checks that were run while building Tables 3 and 4 of the appendix:
  P1  power_table._phi (the stable form) agrees with the original sinh/cosh form wherever that is finite;
  P2  single-layer tables at w = 1 agree with the closed form ht_r(u) = r u + log Gamma(r+1) - (r+1) log(1+e^u),
      on the benchmark rows (results14/trials.pkl) and on the Bible rows (up to r = 70,579);
  P3  step halving (h = 0.1 vs 0.05) of the single-layer quadrature at w = 1/80 ... 80;
  P4  Mellin--Barnes tables (L layers sharing one power) agree with the lattice recursion of modelb_lib at
      L = 2 (w = 1/2) and L = 3 (w = 1/3), and with power_table at L = 1;
  P5  end to end: the stored single-layer w = 1 results equal the depth-1 results of the original model
      (benchmark: results14; Bible: results12 vs results4, results15 at n = 1,000).
Each check prints its worst discrepancy; tolerances are those quoted in the appendix (~1e-10 for tables).
"""
import argparse, pickle, warnings
import numpy as np
from multiprocessing import Pool
import power_table as PT
import mb_table as MB
import run_power as RP
import run_bible as RBi
from modelb_lib import ht_base, advance
from run_chain import setup
from weight_priors import Delta


def _ht(args):
    rows, u, w, h = args
    return PT.ht_rows(rows, u, w, h)


def _rows_bench():
    T = pickle.load(open("results14/trials.pkl", "rb"))
    rows = {0, 1}
    for (_, r, _, _, _) in T.values():
        rows |= set(r.tolist()) | set((r + 1).tolist())
    return np.array(sorted(rows))


def _rows_bible():
    toks = np.load("data/bible_tokens.npy")
    rows = {0, 1}
    for ck in RBi.checkpoints(toks, [1000, 10_000, 30_000, 100_000, 300_000, 0]):
        rows |= set(ck["r"].tolist())
    return np.array(sorted(rows))


def _par(pool, rows, u, w, h, nproc):
    out = np.empty((len(rows), len(u)))
    for i, part in enumerate(pool.map(_ht, [(rows[i::nproc], u, w, h) for i in range(nproc)])):
        out[i::nproc] = part
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=20)
    a = ap.parse_args()

    # P1
    t = np.linspace(-60, 60, 240001)
    with np.errstate(over="ignore", invalid="ignore"):
        th, sh = np.tanh(t), np.sinh(t)
        old = (t + (sh - t) * (1 - th) / 2, 1 + (np.cosh(t) - 1) * (1 - th) / 2 - (sh - t) * (1 - th * th) / 2)
    new = PT._phi(t)
    print(f"P1 stable phi vs original form on [-60, 60]: max rel. diff phi {np.max(np.abs(new[0] - old[0]) / np.maximum(1, np.abs(old[0]))):.1e}, "
          f"phi' {np.max(np.abs(new[1] - old[1]) / np.abs(old[1])):.1e}; finite at t = 1e4: {np.all(np.isfinite(PT._phi(np.array([1e4]))))}")

    rb, rB = _rows_bench(), _rows_bible()
    with Pool(a.procs) as pool:
        # P2
        for name, rows in (("benchmark", rb), ("Bible", rB)):
            g = RP.grid_for(1.0, 0.025, 1, -40.0)
            u = g.u[::4]
            A = _par(pool, rows, u, 1.0, 0.1, a.procs)
            print(f"P2 w = 1 vs closed form, {name} rows ({len(rows)}, max r {rows.max()}): "
                  f"max |diff| {np.abs(A - PT.closed_form_w1(rows, u)).max():.1e}")
        # P3
        test_b = np.unique(np.r_[rb[:5], rb[::40], rb[-8:]])
        test_B = np.unique(np.r_[rB[:5], rB[::60], rB[-25:]])
        for w in (1 / 80, 1 / 22, 0.5, 2.0, 5.0, 22.0, 27.0, 50.0, 80.0):
            g = RP.grid_for(w, 0.025, 1, -40.0)
            u = g.u[::16]
            worst, bad = 0.0, 0
            for rows in (test_b, test_B):
                A = _par(pool, rows, u, w, 0.1, a.procs)
                B = _par(pool, rows, u, w, 0.05, a.procs)
                ok = np.isfinite(A) & np.isfinite(B)
                bad += int((~ok).sum())
                worst = max(worst, float(np.abs(A - B)[ok].max()))
            print(f"P3 step halving w = {w:.4f} ({g}): max |h - h/2| {worst:.1e}, non-finite {bad}")

    # P4
    rows = np.array([0, 1, 2, 3, 5, 10, 30, 100])
    for L, w in ((2, 0.5), (3, 1.0 / 3.0)):
        grid, k, lw = setup(Delta(w), 0.025, Lmax=L)
        ht = ht_base(rows, grid)
        for _ in range(L):
            ht = advance(ht, k, lw, grid)
        sel = (grid.u >= -20) & (grid.u <= 40)
        mb = MB.ht_rows(rows, grid.u[sel], L, w)
        lat = ht[:, sel]
        keep = lat > lat.max(axis=1, keepdims=True) - 40
        print(f"P4 Mellin-Barnes vs lattice, L = {L}, w = {w:.4f}: max |diff| {np.abs(mb - lat)[keep].max():.1e}")
    g = RP.grid_for(2.0, 0.025, 1, -40.0)
    u = g.u[::40]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = np.abs(MB.ht_rows(rows, u, 1, 2.0) - PT.ht_rows(rows, u, 2.0))
    print(f"P4 Mellin-Barnes vs power_table, L = 1, w = 2: max |diff| {np.nanmax(d):.1e}")

    # P5
    R, P = pickle.load(open("results14/delta1.pkl", "rb")), pickle.load(open("results14/pow_w1.pkl", "rb"))
    d = max(abs(P["logq"][kk] - R["logq"][kk]) for kk in P["logq"])
    print(f"P5 benchmark: single layer w = 1 vs depth 1, all trials: max |log Q diff| {d:.1e} nats")
    for dep, pw in (("results4/delta1.pkl", "results12/bible_pow_w1.pkl"), ("results15/delta1.pkl", "results15/bible_pow_w1.pkl")):
        R, P = pickle.load(open(dep, "rb")), pickle.load(open(pw, "rb"))
        print(f"P5 Bible ({pw.split('/')[0]}): single layer w = 1 vs depth 1: max |log Q diff| "
              f"{np.abs(np.asarray(P['logq']) - R['logq'][:, 0]).max():.1e} nats")


if __name__ == "__main__":
    main()
