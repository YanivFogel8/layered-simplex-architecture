"""Validation suite for Model B (plan checks V1-V8).  Usage: python validate_b.py V2 [V6 ...] [only=unif2,...] [Ls=1,2,3]"""
import sys, time
import numpy as np
from weight_priors import Delta, LogUniform, Exp1, Uniform, EULER

B_CLND = np.log(1e4) / (1.0 - np.euler_gamma)          # b = c* ln d at d = 1e4 (Experiment 6)
PRIORS = [Delta(1.0)] + [LogUniform(b) for b in (1.25, 1.5, 2, 3, 4)] + [Exp1(), Uniform(2.0), Uniform(1.0),
                                                                     LogUniform(B_CLND)]
V6_PRIORS = [Delta(1.0), LogUniform(2), LogUniform(4), Exp1(), Uniform(2.0), Uniform(1.0), LogUniform(B_CLND)]


def _zgrid(prior):
    m = prior.moments()
    if not prior.bounded:
        return np.arange(-3000.0, 300.0, 0.05)
    lo = -40.0 * max(prior.w_max, 1.0) - 60.0          # left tail ~ exp(z / w_max)
    return np.arange(lo, 40.0 + 10 * prior.w_max, 0.01)


def check_V2():
    """g table on its own: normalization, mean, variance, node doubling."""
    ok = True
    for pr in PRIORS:
        z = _zgrid(pr)
        dz = z[1] - z[0]
        lg = pr.log_g(z)
        g = np.exp(lg)
        tot = g.sum() * dz
        mean = (z * g).sum() * dz / tot
        var = ((z - mean) ** 2 * g).sum() * dz / tot
        m = pr.moments()
        want_mean = m["EZ"] - pr.z_shift
        # node doubling
        if isinstance(pr, LogUniform):
            pr2 = LogUniform(pr.b, n=2 * pr.n - 1)
        elif isinstance(pr, Exp1):
            pr2 = Exp1(panel=0.05)          # moments of the singular g: see V2k
        elif isinstance(pr, Uniform):
            pr2 = Uniform(pr.a, panel=0.05, n_ts=801)
        else:
            pr2 = pr
        zs = np.linspace(z[0] * 0.5, z[-1], 4001)
        fin = np.isfinite(pr.log_g(zs)) & (pr.log_g(zs) > -5e4)
        dd = np.max(np.abs(pr.log_g(zs[fin]) - pr2.log_g(zs[fin])))
        e = (abs(tot - 1), abs(mean - want_mean), abs(var - m["VarZ"]) / m["VarZ"])
        good = e[0] < 1e-7 and e[1] < 1e-6 and e[2] < 1e-6 and dd < 1e-9
        ok &= good
        print(f"  {pr.name:<10} int g-1={e[0]:.1e}  mean err={e[1]:.1e}  rel var err={e[2]:.1e}  "
              f"node-doubling max|dlog g|={dd:.1e}  shift={pr.z_shift:+.4f}  c*={pr.cstar():.3f}"
              f"  {'OK' if good else 'FAIL'}", flush=True)
    return ok


def check_V2k(du=0.025):
    """Lattice kernel weights (what the recursion actually uses) reproduce the
    moments of Z' = log V - z_shift: mass 1, mean, variance.  For Exp1 this is
    the real test of the singular product-integration weights."""
    from modelb_lib import Grid, layer_kernel
    ok = True
    for pr in PRIORS:
        g = Grid(-20, 65, du)
        wm = pr.w_max if pr.bounded else 1.0
        zlo, zhi = (-3000.0, 300.0) if not pr.bounded else (-60.0 * wm - 60, 40 + 10 * wm)
        k, lw = layer_kernel(pr, g, zlo, zhi)
        z = k * du
        p = np.exp(lw)
        m = pr.moments()
        mass = p.sum()
        mean = (p * z).sum() / mass
        var = (p * (z - mean) ** 2).sum() / mass
        want_mean = m["EZ"] - pr.z_shift
        e = (abs(mass - 1), abs(mean - want_mean), abs(var - m["VarZ"]) / m["VarZ"])
        good = e[0] < 1e-8 and e[1] < 1e-7 and e[2] < 1e-7
        ok &= good
        print(f"  {pr.name:<10} du={du}  mass-1={e[0]:.1e}  mean err={e[1]:.1e}  rel var err={e[2]:.1e}"
              f"  (#offsets={len(k)})  {'OK' if good else 'FAIL'}", flush=True)
    return ok


def check_V1a():
    """Delta(1): one lattice layer from ht^(0) = r u - e^u reproduces the analytic
    L = 1 row  h_r = lgamma(r+1) - (r+1) log(1+e^u)  on the whole grid."""
    from scipy.special import gammaln
    from run_chain import setup
    from modelb_lib import ht_base, advance, untilt
    pr = Delta(1.0)
    grid, k, lw = setup(pr, 0.025, Lmax=44)
    rows = np.array([0, 1, 2, 5, 10, 40, 100, 300, 600, 859])
    t0 = time.time()
    H = untilt(advance(ht_base(rows, grid), k, lw, grid), rows, grid)
    dt = time.time() - t0
    want = gammaln(rows[:, None] + 1.0) - (rows[:, None] + 1.0) * np.logaddexp(0.0, grid.u)[None, :]
    err = np.abs(H - want).max(axis=1)
    for r, e in zip(rows, err):
        print(f"  r={r:<4d} max|h - exact| = {e:.1e}")
    print(f"  one layer, {len(rows)} rows, {grid}: {dt:.1f}s ({dt/len(rows):.2f}s per row)")
    return bool(np.all(err[rows <= 300] < 1e-8) and np.all(err < 1e-6))


def _mc_logq(prior, d, prof, L, n_draw=3_000_000, chunk=250_000, seed=0):
    """Direct Monte Carlo of q = E[prod_i theta_i^{m_i}], theta = Y/sum Y, Y_i = prod_l E^w.
    Returns (log q, relative standard error)."""
    rng = np.random.default_rng(seed)
    m = np.zeros(d); m[:len(prof)] = prof
    s1 = s2 = 0.0
    for _ in range(n_draw // chunk):
        Z = np.zeros((chunk, d))
        for _ in range(L):
            x = np.log(rng.exponential(size=(chunk, d)))
            if isinstance(prior, Delta):
                w = prior.w0
            elif isinstance(prior, LogUniform):
                w = prior.s * np.exp(rng.uniform(-np.log(prior.b), np.log(prior.b), size=(chunk, d)))
            elif isinstance(prior, Uniform):
                w = rng.uniform(0.0, prior.a, size=(chunk, d))
            else:
                w = rng.exponential(size=(chunk, d))
            Z += w * x
        Z -= Z.max(axis=1, keepdims=True)
        lth = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
        v = np.exp(lth @ m)
        s1 += v.sum(); s2 += (v * v).sum()
    n = (n_draw // chunk) * chunk
    mean = s1 / n
    se = np.sqrt(max(s2 / n - mean ** 2, 0.0) / n)
    return np.log(mean), se / mean


def check_V6(d=8, Ls=(1, 2, 5)):
    """Model-free check: lattice pipeline vs direct Monte Carlo over the prior (d = 8)."""
    from run_chain import setup
    from modelb_lib import ht_base, advance, untilt, mixture_predictive
    profs = [(5,), (3, 1, 1), (2, 2), (1, 1, 1, 1), (4, 2)]
    rows = np.arange(0, 7)
    ok = True
    for pr in V6_PRIORS:
        grid, k, lw = setup(pr, 0.025, Lmax=max(Ls))
        ht = ht_base(rows, grid)
        for L in range(1, max(Ls) + 1):
            ht = advance(ht, k, lw, grid)
            if L not in Ls:
                continue
            H = untilt(ht, rows, grid)
            ridx = {int(r): i for i, r in enumerate(rows)}
            for prof in profs:
                r, c = np.unique(prof, return_counts=True)
                lq = mixture_predictive(int(sum(prof)), d, r, c, H, ridx, grid, want_ratios=False)[0]
                lmc, rse = _mc_logq(pr, d, prof, L, seed=L * 100 + len(prof))
                z = (lq - lmc) / rse
                good = abs(z) < 4
                ok &= good
                print(f"  {pr.name:<9} L={L} prof={str(prof):<13} lattice {lq:+.6f}  MC {lmc:+.6f} "
                      f"(rel se {rse:.1e})  z={z:+.1f}  {'OK' if good else 'FAIL'}", flush=True)
    return ok


if __name__ == "__main__":
    only = [x[5:].split(",") for x in sys.argv[1:] if x.startswith("only=")]
    LS = next((tuple(int(v) for v in x[3:].split(",")) for x in sys.argv[1:] if x.startswith("Ls=")), None)
    if only:
        PRIORS[:] = [p for p in PRIORS if p.name in only[0]]
        V6_PRIORS[:] = [p for p in V6_PRIORS if p.name in only[0]]
    for name in [x for x in sys.argv[1:] if not x.startswith(("only=", "Ls="))]:
        t0 = time.time()
        print(f"== {name}", flush=True)
        res = globals()[f"check_{name}"](**({"Ls": LS} if name == "V6" and LS else {}))
        print(f"== {name}: {'PASS' if res else 'FAIL'} [{time.time() - t0:.0f}s]", flush=True)
