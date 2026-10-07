"""Evaluate positive-depth components on prepared main-paper trials; aggregation adds L=0."""
import argparse, os, pickle, time
import numpy as np
from protocol import guard
from multiprocessing import Pool
from weight_priors import make_prior
from modelb_lib import (ht_base, advance, advance_fine, fine_kernel, substeps, untilt,
                        mixture_predictive)
from run_chain import setup
from profiles import zipf, profile_of

K = 10_000
NS = [1000, 2000, 3000, 5000, 7000, 10000, 14000, 20000]
T = 20
LADDER = list(range(1, 81))
TARGETS = [(0.5, 8), (1.0, 2), (1.5, 3), (2.0, 6), (4.0, 7), (6.0, 9)]   # (alpha, paper seed index)
OTHER = [("uniform", 0), ("step", 1), ("dir1", 4), ("dir05", 5), ("geom", 10)]   # (name, paper seed index)
SETS = {"zipf": (TARGETS, "results3"), "other": (OTHER, "results6")}


def target_p(spec, D, t):
    """Target distribution of the paper's run_mixture2.py / run_more.py (trial t)."""
    if not isinstance(spec, str):
        return zipf(K, spec)
    if spec == "uniform":
        return np.full(K, 1.0 / K)
    if spec == "step":
        p = np.empty(K); p[: K // 2] = 0.5 / K; p[K // 2:] = 1.5 / K
        return p
    if spec == "geom":
        w = 0.998 ** np.arange(1, K + 1)
        return w / w.sum()
    rng = np.random.default_rng(10_000 + 97 * D + t)
    return rng.dirichlet((1.0 if spec == "dir1" else 0.5) * np.ones(K))


def build_trials(targets=TARGETS):
    """dict (ti, ni, t) -> (n, prof_r, prof_c, S_r, sum p log p); S_r = true mass of symbols
    with count r (index 0 of S is the unseen symbols, then prof_r order)."""
    out = {}
    for ti, (a, D) in enumerate(targets):
        for t in range(T):
            p = target_p(a, D, t)
            plp = float(np.sum(p[p > 0] * np.log(p[p > 0])))
            for ni, n in enumerate(NS):
                c = np.random.default_rng(500_000 + 9973 * D + 131 * t + ni).multinomial(n, p)
                r, cc = profile_of(c)
                S = np.array([p[c == 0].sum()] + [p[c == v].sum() for v in r])
                out[(ti, ni, t)] = (n, r, cc, S, plp)
    return out


_W = {}


def _init(f):
    with open(f, "rb") as fh:
        _W["trials"] = pickle.load(fh)


def _adv(args):
    ht, k, lw, grid = args
    return advance(ht, k, lw, grid)


def _adv_fine(args):
    ht, rows, kern, grid, first = args
    return advance_fine(ht, rows, kern, grid, first)


def _eval(args):
    keys, hfile, rows, grid = args
    if _W.get("hfile") != hfile:
        _W["H"], _W["hfile"] = np.load(hfile), hfile
        _W["ridx"] = {int(r): i for i, r in enumerate(rows)}
    out = []
    edge = 4.0 / grid.du + 1
    for key in keys:
        n, r, c, S, _ = _W["trials"][key]
        lq, pr, (i0, _) = mixture_predictive(n, K, r, c, _W["H"], _W["ridx"], grid, want_ratios=True,
                                             spline_pad=60)
        q = np.array([pr.get(0, 0.0)] + [pr[int(v)] for v in r])
        norm = (K - int(c.sum())) * q[0] + float(np.dot(c, q[1:])) - 1.0
        out.append((key, lq, q, norm, i0 < edge or i0 > grid.G - 1 - edge))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prior")
    ap.add_argument("--set", default="zipf", choices=sorted(SETS))
    ap.add_argument("--procs", type=int, default=20)
    ap.add_argument("--out", help="default: results3 (zipf) / results6 (other)")
    ap.add_argument("--du", type=float, default=0.025)
    ap.add_argument("--Lstop", type=int)
    ap.add_argument("--Lmax", type=int, default=max(LADDER), help="deepest positive component (main protocol: 80)")
    ap.add_argument("--targets", help="comma-separated target indices (0..5), default all")
    ap.add_argument("--ntrials", type=int, default=T, help="use trials t < ntrials (tests)")
    ap.add_argument("--ns", help="comma-separated sample sizes to keep (default: all of NS)")
    ap.add_argument("--shift", choices=["lam", "mean"], default="lam",
                    help="constant removed per layer: Lam(1) (E[Y] = 1, default) or E[Z] (typical log Y = 0; "
                    "theta and all predictions are unchanged, but the u-grid no longer drifts with depth)")
    ap.add_argument("--umin", type=float); ap.add_argument("--umax", type=float)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    targets, default_out = SETS[a.set]
    a.out = a.out or default_out
    os.makedirs(a.out, exist_ok=True)
    prior = make_prior(a.prior)
    if a.shift == "mean":
        prior.shift_override = prior.moments()["EZ"]
    tag = prior.name + (f"_{a.tag}" if a.tag else "")
    fres, fstate, ftr = f"{a.out}/{tag}.pkl", f"{a.out}/{tag}_state.npz", f"{a.out}/trials.pkl"
    hbase = f"{a.out}/_H_{tag}"

    t0 = time.time()
    if os.path.exists(ftr):
        with open(ftr, "rb") as f:
            trials = pickle.load(f)
    else:
        raise SystemExit("Prepare main-paper trials first with table1_trials.py --out OUT/trials.pkl")
    if a.targets:
        keep = {int(x) for x in a.targets.split(",")}
        trials = {kk: v for kk, v in trials.items() if kk[0] in keep}
    trials = {kk: v for kk, v in trials.items() if kk[2] < a.ntrials}
    if a.ns:
        keep_n = {NS.index(int(x)) for x in a.ns.split(",")}
        trials = {kk: v for kk, v in trials.items() if kk[1] in keep_n}
    guard(a.out, tag, {"trials": ftr}, {"prior":a.prior,"du":a.du,"Lmax":a.Lmax,"targets":a.targets,"ntrials":a.ntrials,"ns":a.ns,"shift":a.shift,"umin":a.umin,"umax":a.umax})
    rows = {0, 1}
    for (_, r, _, _, _) in trials.values():
        rows |= set(r.tolist()) | set((r + 1).tolist())
    rows = np.array(sorted(rows))
    ladder = list(range(1, a.Lmax + 1))
    LM = max(ladder) if a.Lstop is None else a.Lstop
    grid, k, lw = setup(prior, a.du, u_min=a.umin, u_max=a.umax, Lmax=max(ladder))
    m = substeps(rows, grid.du)
    coarse, fine = np.where(m == 1)[0], np.where(m > 1)[0]
    zlo, zhi = k[0] * grid.du, k[-1] * grid.du
    kern = {mm: fine_kernel(prior, grid, zlo, zhi, int(mm)) for mm in sorted(set(m[fine].tolist()))}
    print(f"{tag}: {grid}, {len(rows)} rows (max {rows.max()}), {len(fine)} on sub-lattices "
          f"m={sorted(kern)}, {len(trials)} trials, L=1..{LM}, setup {time.time()-t0:.0f}s", flush=True)

    res = {"logq": {}, "q": {}, "norm": {}, "targets": "main Table 1 row order (see table1_trials.py)", "ns": NS, "T": T, "ladder": ladder,
           "prior": tag, "grid": (grid.u_min, grid.u_max, grid.du)}
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

    keys = sorted(trials)
    ftr_run = f"{a.out}/trials_{tag}.pkl"
    with open(ftr_run, "wb") as f:
        pickle.dump(trials, f)
    with Pool(a.procs, initializer=_init, initargs=(ftr_run,)) as pool:
        while ell < LM:
            t1 = time.time()
            # Heavy rows need the sub-lattice only in the FIRST layer (closed form of the sharp
            # layer-0 bump); one convolution smooths the table to O(1) width, and the ordinary
            # lattice is then exact (checked vs direct quadrature to 6e-11 at L=3, r=19697).
            # Interpolating on the sub-lattice at later layers was both slower and less accurate.
            cur = coarse if ell == 0 else np.arange(len(rows))
            jobs_c = [(ht[s], k, lw, grid) for s in np.array_split(cur, a.procs * 2)]
            rc = pool.map_async(_adv, jobs_c)
            if ell == 0:
                fs = fine[np.argsort(-m[fine])]                    # heaviest first, dealt round-robin
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
            hfile = f"{hbase}_L{ell}.npy"
            np.save(hfile, H)
            if os.path.exists(f"{hbase}_L{ell - 1}.npy"):
                try:
                    os.remove(f"{hbase}_L{ell - 1}.npy")
                except OSError:
                    pass
            nmax, bad = 0.0, 0
            for out in pool.imap_unordered(_eval, [(keys[i::a.procs * 3], hfile, rows, grid)
                                                   for i in range(a.procs * 3)]):
                for key, lq, q, nrm, at_edge in out:
                    res["logq"][key + (ell,)] = lq
                    res["q"][key + (ell,)] = q
                    res["norm"][key + (ell,)] = nrm
                    nmax = max(nmax, abs(nrm))
                    bad += at_edge
            if bad:
                raise SystemExit(f"{bad} trials have their saddle at the grid edge at L={ell}")
            with open(fres + ".tmp", "wb") as f:
                pickle.dump(res, f)
            os.replace(fres + ".tmp", fres)
            np.savez(fstate, ht=ht, ell=ell, rows=rows)
            print(f"L={ell:2d} adv {t_adv:6.1f}s eval {time.time()-t1-t_adv:6.1f}s | "
                  f"max |sum of predictive - 1| {nmax:.1e} [{time.time()-t0:.0f}s]", flush=True)
    for f in os.listdir(a.out):
        if f.startswith(os.path.basename(hbase)):
            try:
                os.remove(os.path.join(a.out, f))
            except OSError:
                pass
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
