"""Build one prior's layer chain and evaluate the Zipf-spectrum experiment at every depth.

    python run_chain.py PRIOR [--du 0.025] [--procs 20] [--out results] [--Lmax0 N --Lmax1 N]

PRIOR: delta | logu:<b> | exp1 | unif:<a>.  At each depth L it records
  logq[pi, ai, L-1, t]  for all 880 profiles (paper seeds),
  identity errors       q(1)=1/d and d q(2) + d(d-1) q(1,1) = 1     (V4),
  predictive mass       on four measured profiles                    (V5),
and checkpoints the table so the run resumes / extends to larger L.
"""
import argparse, os, time
import numpy as np
from multiprocessing import Pool
from weight_priors import make_prior
from profiles import spectrum_profiles, rows_needed, ALPHAS, T, PANELS
from modelb_lib import Grid, layer_kernel, ht_base, advance, untilt, mixture_predictive

NORM_PROFILES = [(1, 5, 0), (1, 10, 0), (0, 0, 0), (1, 2, 0)]   # (pi, ai, t)


def default_lmax(prior, d):
    if prior.name in ("delta1", "exp1") or prior.name.startswith("unif"):
        return {1_000: 33, 10_000: 44}[d]                       # the paper's ladders
    return int(round(2 * prior.cstar() * np.log(d)))


def setup(prior, du, u_min=None, u_max=None, Lmax=44):
    m = prior.moments()
    if prior.bounded:
        mu = Lmax * (-m["EZ"] + prior.z_shift)                  # where tY ~ 1 at depth Lmax
        sd = np.sqrt(Lmax * m["VarZ"])
        u_min = -40.0 if u_min is None else u_min    # -20 (paper) biases deep-L identities; see V8
        u_max = max(65.0, mu + 6 * sd) if u_max is None else u_max
        grid = Grid(u_min, u_max, du)
        wm = prior.w_max
        z_lo = -(grid.u_max - grid.u_min) - 30 * wm
        z_hi = wm * np.log(1 + 900 * wm) + 6 * wm + 3
    else:
        grid = Grid(-120.0 if u_min is None else u_min, 80.0 if u_max is None else u_max, du)
        z_lo, z_hi = -250.0, 200.0
    k, lw = layer_kernel(prior, grid, z_lo, z_hi)
    return grid, k, lw


def _adv(args):
    ht, k, lw, grid = args
    return advance(ht, k, lw, grid)


def _eval(args):
    """logq for a batch of profiles at one depth."""
    items, H, rowidx, grid = args
    out = []
    for key, (d, n, r, c) in items:
        lq, _, (_, us) = mixture_predictive(n, d, r, c, H, rowidx, grid, want_ratios=False)
        out.append((key, lq, us))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prior")
    ap.add_argument("--du", type=float, default=0.025)
    ap.add_argument("--procs", type=int, default=20)
    ap.add_argument("--out", default="results")
    ap.add_argument("--Lmax0", type=int); ap.add_argument("--Lmax1", type=int)
    ap.add_argument("--umin", type=float); ap.add_argument("--umax", type=float)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ntrials", type=int, default=T, help="use trials t < ntrials only (V8 runs)")
    a = ap.parse_args()

    prior = make_prior(a.prior)
    Lmax = {0: a.Lmax0 or default_lmax(prior, 1_000), 1: a.Lmax1 or default_lmax(prior, 10_000)}
    LM = max(Lmax.values())
    tag = prior.name + (f"_{a.tag}" if a.tag else "")
    os.makedirs(a.out, exist_ok=True)
    fres, fstate = f"{a.out}/{tag}.npz", f"{a.out}/{tag}_state.npz"

    t0 = time.time()
    grid, k, lw = setup(prior, a.du, a.umin, a.umax, LM)
    profs, H2 = spectrum_profiles()
    profs = {key: v for key, v in profs.items() if key[2] < a.ntrials}
    extra = {0, 1, 2, 3}
    for key in NORM_PROFILES:
        extra |= set((profs[key][2] + 1).tolist())
    rows = rows_needed(profs, extra)
    rowidx = {int(r): i for i, r in enumerate(rows)}
    print(f"{tag}: {grid}, kernel offsets {len(k)} [{k[0]*grid.du:.1f},{k[-1]*grid.du:.1f}], "
          f"{len(rows)} rows, Lmax={Lmax}, c*={prior.cstar():.3f}, setup {time.time()-t0:.0f}s", flush=True)

    logq = np.full((2, len(ALPHAS), LM, T), np.nan)
    ident = np.full((LM, 2, 2), np.nan)      # [L, panel, (log-err q1, err 2-sample)]
    norm = np.full((LM, len(NORM_PROFILES)), np.nan)
    saddle = np.full((2, len(ALPHAS), LM, T), np.nan)
    if os.path.exists(fstate):
        st = np.load(fstate)
        ht, ell = st["ht"], int(st["ell"])
        old = np.load(fres)
        same = (np.array_equal(st["rows"], rows) and ht.shape[1] == grid.G
                and np.isclose(float(old["u_min"]), grid.u_min) and np.isclose(float(old["u_max"]), grid.u_max)
                and np.isclose(float(old["du"]), grid.du))
        if not same:
            raise SystemExit(f"checkpoint grid/rows differ from this run ({grid}); refusing to resume. "
                             f"Pass --umin {float(old['u_min'])} --umax {float(old['u_max'])} --du {float(old['du'])}")
        n0 = min(old["logq"].shape[2], LM)
        logq[:, :, :n0], ident[:n0], norm[:n0] = old["logq"][:, :, :n0], old["ident"][:n0], old["norm"][:n0]
        saddle[:, :, :n0] = old["saddle"][:, :, :n0]
        print(f"resumed at L={ell}", flush=True)
    else:
        ht, ell = ht_base(rows, grid), 0

    nch = min(a.procs, len(rows))
    splits = np.array_split(np.arange(len(rows)), nch)
    with Pool(a.procs) as pool:
        while ell < LM:
            t1 = time.time()
            parts = pool.map(_adv, [(ht[s], k, lw, grid) for s in splits])
            ht = np.concatenate(parts, axis=0); ell += 1
            t_adv = time.time() - t1
            H = untilt(ht, rows, grid)

            items = [(key, v) for key, v in profs.items() if ell <= Lmax[key[0]]]
            batches = [items[i::a.procs] for i in range(a.procs)]
            for res in pool.map(_eval, [(b, H, rowidx, grid) for b in batches]):
                for (pi, ai, t), lq, us in res:
                    logq[pi, ai, ell - 1, t] = lq
                    saddle[pi, ai, ell - 1, t] = us
            edge = np.nanmin(np.minimum(saddle[:, :, ell - 1] - grid.u_min, grid.u_max - saddle[:, :, ell - 1]))

            for pi, (d, _) in PANELS.items():
                l1 = mixture_predictive(1, d, [1], [1], H, rowidx, grid, want_ratios=False)[0]
                l2 = mixture_predictive(2, d, [2], [1], H, rowidx, grid, want_ratios=False)[0]
                l11 = mixture_predictive(2, d, [1], [2], H, rowidx, grid, want_ratios=False)[0]
                ident[ell - 1, pi] = (l1 + np.log(d), d * np.exp(l2) + d * (d - 1) * np.exp(l11) - 1)
            for j, key in enumerate(NORM_PROFILES):
                d, n, r, c = profs[key]
                _, pr, _ = mixture_predictive(n, d, r, c, H, rowidx, grid, want_ratios=True)
                s = int(c.sum())
                norm[ell - 1, j] = (d - s) * pr.get(0, 0.0) + sum(int(cc) * pr[int(rr)] for rr, cc in zip(r, c)) - 1

            np.savez(fres, logq=logq, ident=ident, norm=norm, saddle=saddle, alphas=ALPHAS,
                     panels=np.array([PANELS[0], PANELS[1]]), Lmax=np.array([Lmax[0], Lmax[1]]),
                     H2=np.array([[H2[(pi, ai)] for ai in range(len(ALPHAS))] for pi in (0, 1)]),
                     du=grid.du, u_min=grid.u_min, u_max=grid.u_max, cstar=prior.cstar())
            np.savez(fstate, ht=ht, ell=ell, rows=rows)
            print(f"L={ell:2d} adv {t_adv:5.1f}s eval {time.time()-t1-t_adv:5.1f}s | "
                  f"q1 logerr {ident[ell-1,0,0]:+.1e}/{ident[ell-1,1,0]:+.1e} "
                  f"2-sample {ident[ell-1,0,1]:+.1e}/{ident[ell-1,1,1]:+.1e} | "
                  f"norm max {np.nanmax(np.abs(norm[ell-1])):.1e} | saddle-edge {edge:.1f} [{time.time()-t0:.0f}s]"
                  + ("  WARNING: saddle near grid edge" if edge < 5 else ""), flush=True)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
