"""Heavy-count check for the Bible run: first layer of the tilted table for r up to 70,579.

    python validate_heavy.py PRIOR [--du 0.025]

The production first layer (advance_fine, sub-lattice from the closed form of ht^(0)) is
compared with an independent reference that integrates over the power w with the prior's own
tau-rule and over x = log E by a dense trapezoid refined around the sharp peak:

    ht_r^(1)(u) = log sum_w nu_w int g_G(x) exp( r v - e^v ) dx,   v = u + w x - z_shift.

The error is reported in nats over the u-range where the Bible's profile integral lives.
"""
import argparse
import numpy as np
from scipy.special import logsumexp
from weight_priors import make_prior, _log_gG
from modelb_lib import fine_kernel, advance_fine, substeps, ht_base
from run_chain import setup


def reference(prior, r, us):
    c = prior.z_shift
    if prior.singular:
        # independent, coarser rule (1-unit Gauss-Legendre panels, w > e^-30; the dropped
        # mass is ~1e-13) so that the reference stays affordable
        from weight_priors import Exp1, Uniform
        coarse = dict(panel=1.0, n=16, tau_lo=-30.0)
        tau, lw = (Exp1(**coarse) if prior.name == "exp1" else Uniform(prior.a, **coarse)).tau_rule()
    else:
        tau, lw = prior.tau_rule()
    lw = lw - logsumexp(lw)
    xb = np.arange(-45.0, 30.0, 0.002)

    def logf(x, u, w):
        v = u + w * x - c
        return _log_gG(x) + r * v - np.exp(np.minimum(v, 700.0))

    out = np.empty(len(us))
    for i, u in enumerate(us):
        terms = np.empty(len(tau))
        for j, (t, l) in enumerate(zip(tau, lw)):
            w = np.exp(t)
            fb = logf(xb, u, w)
            xs = xb[np.argmax(fb)]                               # peak located numerically
            x = np.union1d(xb, xs + np.linspace(-0.05, 0.05, 20001))
            f = logf(x, u, w)
            dx = np.diff(x)
            m = f.max()
            s = np.exp(f - m)
            terms[j] = l + m + np.log(0.5 * np.sum((s[1:] + s[:-1]) * dx))
        out[i] = logsumexp(terms)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prior")
    ap.add_argument("--du", type=float, default=0.025)
    a = ap.parse_args()
    prior = make_prior(a.prior)
    grid, k, lw = setup(prior, a.du, Lmax=40)
    zlo, zhi = k[0] * grid.du, k[-1] * grid.du
    for r in (19697, 38848, 70579):
        m = int(substeps(r, grid.du))
        kern = {m: fine_kernel(prior, grid, zlo, zhi, m)}
        ht1 = advance_fine(ht_base([r], grid), np.array([r]), kern, grid, True)[0]
        step = 120 if prior.singular else 40
        idx = np.arange(np.searchsorted(grid.u, -15.0), np.searchsorted(grid.u, 15.0), step)
        ref = reference(prior, r, grid.u[idx])
        err = np.abs(ht1[idx] - ref)
        print(f"{prior.name} du={grid.du} r={r:6d} m={m}: max |lattice - reference| = {err.max():.1e} nats "
              f"over u in [-15, 15] ({len(idx)} points)", flush=True)


if __name__ == "__main__":
    main()
