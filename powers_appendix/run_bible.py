"""Canonical Table 2 positive-depth evidences; aggregation uses equal-prior L=0..54."""
import argparse, os, pickle, time
import numpy as np
from protocol import guard,load_tokens,TOKENS
from multiprocessing import Pool
from weight_priors import make_prior
from modelb_lib import (ht_base, advance, advance_fine, fine_kernel, substeps, untilt,
                        mixture_predictive, LOG2)
from run_chain import setup
from profiles import profile_of

D = 100_000
LADDER = list(range(1, 55))


def checkpoints(toks, ns=None):
    """Prefix statistics at the paper's checkpoints (or at the prefix lengths ns; 0 = the whole text)."""
    out = []
    for n in ([10_000, 30_000, 100_000, 300_000, len(toks)] if ns is None else [m or len(toks) for m in ns]):
        if n <= 0 or n > len(toks): raise ValueError("Prefix outside token stream")
        c = np.bincount(toks[:n]); c = c[c > 0]
        p = c / n
        r, cc = profile_of(c)
        out.append(dict(n=n, r=r, c=cc, H=float(-(p * np.log2(p)).sum()), distinct=len(c)))
    return out


def _adv(args):
    ht, k, lw, grid = args
    return advance(ht, k, lw, grid)


def _adv_fine(args):
    ht, rows, kern, grid, first = args
    return advance_fine(ht, rows, kern, grid, first)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prior")
    ap.add_argument("--du", type=float, default=0.025)
    ap.add_argument("--procs", type=int, default=20)
    ap.add_argument("--out", default="results4")
    ap.add_argument("--tag", default="")
    ap.add_argument("--Lmax", type=int, default=max(LADDER), help="deepest positive component (main protocol: 54)")
    ap.add_argument("--ckpts", help="comma-separated prefix lengths (default: the paper's 1e4, 3e4, 1e5, 3e5, N)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prior = make_prior(a.prior)
    tag = prior.name + (f"_{a.tag}" if a.tag else "")
    fres, fstate = f"{a.out}/{tag}.pkl", f"{a.out}/{tag}_state.npz"
    ladder = list(range(1, a.Lmax + 1))

    t0 = time.time()
    toks = load_tokens()
    cks = checkpoints(toks, [int(x) for x in a.ckpts.split(",")] if a.ckpts else None)
    guard(a.out,tag,{"tokens":TOKENS},{"prior":a.prior,"du":a.du,"Lmax":a.Lmax,"ckpts":[ck["n"] for ck in cks]})
    rows = {0, 1}
    for ck in cks:
        rows |= set(ck["r"].tolist())
    rows = np.array(sorted(rows))
    grid, k, lw = setup(prior, a.du, Lmax=max(ladder))
    m = substeps(rows, grid.du)
    coarse, fine = np.where(m == 1)[0], np.where(m > 1)[0]
    zlo, zhi = k[0] * grid.du, k[-1] * grid.du
    kern = {mm: fine_kernel(prior, grid, zlo, zhi, int(mm)) for mm in sorted(set(m[fine].tolist()))}
    ridx = {int(r): i for i, r in enumerate(rows)}
    print(f"{tag}: {grid}, {len(rows)} rows (max {rows.max()}), {len(fine)} on sub-lattices "
          f"m={sorted(kern)}, checkpoints {[ck['n'] for ck in cks]}, setup {time.time()-t0:.0f}s", flush=True)

    res = {"prior": tag, "grid": (grid.u_min, grid.u_max, grid.du), "ladder": ladder,
           "ckpts": [ck["n"] for ck in cks], "H": [ck["H"] for ck in cks],
           "distinct": [ck["distinct"] for ck in cks],
           "logq": np.full((len(cks), len(ladder)), np.nan), "saddle": np.full((len(cks), len(ladder)), np.nan)}
    if os.path.exists(fstate):
        st = np.load(fstate)
        if not (np.array_equal(st["rows"], rows) and st["ht"].shape[1] == grid.G):
            raise SystemExit("checkpoint rows/grid differ; refusing to resume")
        ht, ell = st["ht"], int(st["ell"])
        with open(fres, "rb") as f:
            res = pickle.load(f)
        print(f"resumed at L={ell}", flush=True)
    else:
        ht, ell = ht_base(rows, grid), 0

    edge = 4.0 / grid.du + 1
    with Pool(a.procs) as pool:
        while ell < max(ladder):
            t1 = time.time()
            cur = coarse if ell == 0 else np.arange(len(rows))
            jobs_c = [(ht[s], k, lw, grid) for s in np.array_split(cur, a.procs * 2)]
            rc = pool.map_async(_adv, jobs_c)
            if ell == 0:
                fs = fine[np.argsort(-m[fine])]
                jobs_f = [(ht[fs[i::a.procs]], rows[fs[i::a.procs]], kern, grid, True) for i in range(a.procs)]
                rf = pool.map_async(_adv_fine, jobs_f)
            new = np.empty_like(ht)
            for s, part in zip(np.array_split(cur, a.procs * 2), rc.get()):
                new[s] = part
            if ell == 0:
                for i, part in enumerate(rf.get()):
                    new[fs[i::a.procs]] = part
            ht = new; ell += 1
            t_adv = time.time() - t1

            H = untilt(ht, rows, grid)
            line = []
            for ci, ck in enumerate(cks):
                lq, _, (i0, us) = mixture_predictive(ck["n"], D, ck["r"], ck["c"], H, ridx, grid,
                                                     want_ratios=False, spline_pad=60)
                if i0 < edge or i0 > grid.G - 1 - edge:
                    raise SystemExit(f"saddle at grid edge (u={us:.1f}) for n={ck['n']}, L={ell}")
                res["logq"][ci, ell - 1] = lq
                res["saddle"][ci, ell - 1] = us
                line.append(-lq / LOG2 / ck["n"] - ck["H"])
            with open(fres + ".tmp", "wb") as f:
                pickle.dump(res, f)
            os.replace(fres + ".tmp", fres)
            np.savez(fstate, ht=ht, ell=ell, rows=rows)
            print(f"L={ell:2d} adv {t_adv:6.1f}s | redundancy " + " ".join(f"{v:.4f}" for v in line)
                  + f" [{time.time()-t0:.0f}s]", flush=True)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
