"""Exact table for L layers that all share one fixed power w (nu = delta_w at every layer).

Then Y = (E_1 ... E_L)^w e^{-s}, s = L log Gamma(1 + w) (so E[Y] = 1), with closed-form moments
E[Y^q] = e^{-s q} Gamma(1 + w q)^L, and the tilted table is a single Mellin-Barnes integral
(the paper's device for its exact rows, paper/code/mellin.py):

    exp ht_r(u) = E[(tY)^r e^{-tY}] = (1/2 pi i) int_{c - i inf}^{c + i inf} exp Phi(sig) d sig,
    Phi(sig) = log Gamma(sig) + (r - sig)(u - s) + L log Gamma(1 + w (r - sig)),   0 < c < r + 1/w.

We take c at the minimum of Phi on the real segment (Phi is convex there), i.e. the saddle, so the
vertical line is the steepest-descent direction at c: the phase is stationary at y = 0 and
|exp Phi(c + iy)| decreases monotonically in |y| (|Gamma(x + iy)| <= Gamma(x)).  By symmetry
exp ht = (1/pi) int_0^inf Re exp Phi(c + iy) dy, integrated with the trapezoid rule; the step
resolves the saddle width, the distance to the nearest pole (at 0 and r + 1/w) and the phase rate
Re Phi'(c + iy) -> (1 - L w) log y - L w log w - (u - s).
"""
import numpy as np
from scipy.special import gammaln, loggamma, digamma, polygamma

DROP = 50.0          # stop once |integrand| < e^-DROP relative to the saddle


def _dphi(c, r, u, L, w, s):
    return digamma(c) - (u - s) - L * w * digamma(1.0 + w * (r - c))


def _d2phi(c, r, L, w):
    return polygamma(1, c) + L * w * w * polygamma(1, 1.0 + w * (r - c))


def _phi(sig, r, u, L, w, s):
    return loggamma(sig) + (r - sig) * (u - s) + L * loggamma(1.0 + w * (r - sig))


def saddle(r, u, L, w, s):
    """Minimum of Phi on (0, r + 1/w) for each u (1-D array)."""
    top = r + 1.0 / w
    lo = np.full_like(u, 0.0); hi = np.full_like(u, top)
    for _ in range(200):                       # phi' increases from -inf to +inf on (0, top)
        mid = 0.5 * (lo + hi)
        neg = _dphi(mid, r, u, L, w, s) < 0
        lo = np.where(neg, mid, lo); hi = np.where(neg, hi, mid)
        if np.all(hi - lo <= 1e-15 * np.maximum(1.0, hi)):
            break
    c = 0.5 * (lo + hi)
    for _ in range(3):
        cn = c - _dphi(c, r, u, L, w, s) / _d2phi(c, r, L, w)
        c = np.where((cn > lo) & (cn < hi), cn, c)
    return c


def _ymap(tau, a, H):
    """y(tau) with y' = a H cosh(tau) / (a cosh(tau) + H): spacing a dtau at the saddle (resolves
    its width and the nearby pole), growing to H dtau far out (resolves the phase rate).  a < H."""
    q = np.sqrt((H - a) / (H + a))
    x = q * np.tanh(tau / 2.0)
    one_m_q = (2.0 * a / (H + a)) / (1.0 + q)
    em = np.exp(-tau)
    one_m_x = one_m_q + q * 2.0 * em / (1.0 + em)                     # 1 - x, without cancellation
    at = 0.5 * (np.log1p(x) - np.log(one_m_x))
    y = H * tau - 2.0 * H * H / np.sqrt((H - a) * (H + a)) * at
    dy = a * H / (a + H / np.cosh(np.minimum(tau, 700.0)))
    return y, dy


def ht_row(r, u, L, w, safety=1.0):
    """Tilted table ht_r(u) for one row r, grid points u (1-D)."""
    r = float(r)
    u = np.asarray(u, dtype=np.float64)
    s = L * float(gammaln(1.0 + w))
    c = saddle(r, u, L, w, s)
    p0 = _phi(c + 0j, r, u, L, w, s).real
    sd = 1.0 / np.sqrt(_d2phi(c, r, L, w))
    dist = np.minimum(c, r + 1.0 / w - c)                 # strip half-width of the analytic integrand
    ymax = 10.0 * sd + 2.0 * DROP / np.pi + 5.0           # |Gamma(c + iy)| alone decays like e^{-pi y/2}
    rate = lambda y: np.abs((1.0 - L * w) * np.log(y) - L * w * np.log(w) - (u - s))
    dt = 0.125 * safety
    hn = np.minimum(sd / 5.0, dist / 8.0)                 # node spacing at the saddle
    out = np.full_like(u, np.nan)
    todo = np.arange(len(u))
    for _ in range(8):
        hf = 1.0 / (np.maximum(rate(np.maximum(ymax, 1.0)), rate(1.0)) + 1.0)   # spacing far out
        H = hf / dt
        a = np.minimum(hn / dt, 0.5 * H)
        # tau at which y reaches ymax (y ~ H tau - const for large tau), plus a margin
        q = np.sqrt((H[todo] - a[todo]) / (H[todo] + a[todo]))
        const = 2.0 * H[todo] / np.sqrt((H[todo] - a[todo]) * (H[todo] + a[todo])) \
            * 0.5 * np.log((1.0 + q) / ((2.0 * a[todo] / (H[todo] + a[todo])) / (1.0 + q)))
        nn = np.ceil((ymax[todo] / H[todo] + const + 1.0) / dt).astype(np.int64)
        order = np.argsort(nn)
        redo = []
        for part in np.array_split(order, max(1, len(todo) // 64)):
            grp = todo[part]
            J = int(nn[part].max())
            tau = np.arange(J + 1) * dt
            y, dy = _ymap(tau[None, :], a[grp, None], H[grp, None])
            ph = _phi(c[grp, None] + 1j * y, r, u[grp, None], L, w, s)
            e = np.exp(ph - p0[grp, None]) * dy                     # |exp(Phi - Phi(c))| <= 1
            wt = np.where(np.arange(J + 1)[None, :] <= nn[part, None], 1.0, 0.0); wt[:, 0] = 0.5
            tot = np.where(wt > 0, e.real * wt, 0.0).sum(1)         # padded nodes may be inf/nan
            tail = np.abs(np.exp(ph - p0[grp, None]))[np.arange(len(grp)), nn[part]]
            bad = tail > np.exp(-DROP)
            redo.extend(grp[bad].tolist())
            ok = ~bad
            if np.any(tot[ok] <= 0):
                raise ValueError(f"non-positive contour integral (r={r}, L={L}, w={w})")
            out[grp[ok]] = p0[grp[ok]] + np.log(tot[ok] * dt / np.pi)
        if not redo:
            return out
        todo = np.array(redo)
        ymax[todo] *= 2.0
    raise ValueError(f"contour not decayed after 8 doublings (r={r}, L={L}, w={w})")


def ht_rows(rows, u, L, w, safety=1.0):
    return np.stack([ht_row(float(r), u, L, w, safety) for r in rows])
