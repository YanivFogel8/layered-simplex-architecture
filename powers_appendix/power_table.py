"""Single-layer table for a fixed power w (nu = delta_w, L = 1), by direct quadrature.

    ht_r(u) = log int exp f(y) dy,   f(y) = r v - e^v + y - e^y,   v = u + w y - s,

with y = log E, E ~ Exp(1), and s = log Gamma(1 + w) = Lam(1), so that E[Y] = 1 as for
the lattice tables.  f is strictly concave (f'' = -w^2 e^v - e^y), and |f''| increases to
the right of the mode and decreases to its left.  For each (r, u) we:
  - find the mode y* (bisection on the monotone f' inside exact brackets, then Newton);
  - find y_R > y* with f(y_R) = f(y*) - DROP; the smallest local scale on [y*, y_R] is
    ell = 1/sqrt|f''(y_R)| (a broad Gumbel mode can sit next to the sharp e^v wall);
  - integrate with the trapezoid rule in t, y = y* + ell phi(t), where
    phi(t) = t + (sinh t - t)(1 - tanh t)/2 is analytic, ~ t on the right (uniform nodes up
    to y_R) and ~ sinh t on the left, where the local scale grows at least exponentially and
    the tail decays only like e^{(r w + 1) y}.
No lattice, so the width of the layer bump (~ 1/(w sqrt r)) needs no sub-lattice for any w.
"""
import numpy as np
from scipy.special import gammaln, logsumexp

DROP = 46.0       # log-integrand drop at the right end (e^-46 ~ 1e-20 relative)
REACH_L = 400.0   # left reach in y


def _f(y, r, u, w, s):
    v = u + w * y - s
    return r * v - np.exp(np.minimum(v, 700.0)) + y - np.exp(np.minimum(y, 700.0))


def _fp(y, r, u, w, s):
    v = u + w * y - s
    ev, ey = np.exp(np.minimum(v, 700.0)), np.exp(np.minimum(y, 700.0))
    return w * (r - ev) + 1.0 - ey, -(w * w * ev + ey)


def mode(r, u, w, s):
    """argmax_y f; r, u broadcastable arrays."""
    # at the mode w e^v + e^y = w r + 1 >= 1, so e^y >= 1/2 or w e^v >= 1/2 (lower bound), and
    # e^y <= w r + 1, w e^v <= w r + 1 (upper bound); v = u + w y - s is solved for y
    lo = np.minimum(-np.log(2.0), (-np.log(2.0 * w) - u + s) / w) - 1.0 + 0.0 * r
    hi = np.minimum(np.log(w * r + 1.0), (np.log(r + 1.0 / w) - u + s) / w) + 1.0
    for _ in range(80):                       # f' is decreasing: bisection, then Newton
        mid = 0.5 * (lo + hi)
        g, _ = _fp(mid, r, u, w, s)
        pos = g > 0
        lo = np.where(pos, mid, lo); hi = np.where(pos, hi, mid)
    y = 0.5 * (lo + hi)
    for _ in range(4):
        g, h = _fp(y, r, u, w, s)
        y = np.clip(y - g / h, lo, hi)
    return y


def _phi(t):
    """phi(t) = t + (sinh t - t)(1 - tanh t)/2 and phi'(t).  For t >= 0 the same expressions are written with
    e^-t only (sinh, cosh overflow past t ~ 710, which large powers w reach); identical to rounding."""
    t = np.asarray(t, dtype=np.float64)
    tn = np.minimum(t, 0.0)                                   # t < 0: the direct form (|t| stays moderate)
    th, sh = np.tanh(tn), np.sinh(tn)
    ph_n = tn + (sh - tn) * (1.0 - th) / 2.0
    dph_n = 1.0 + (np.cosh(tn) - 1.0) * (1.0 - th) / 2.0 - (sh - tn) * (1.0 - th * th) / 2.0
    tp = np.maximum(t, 0.0)                                   # t >= 0: (1 - tanh t)/2 = e^-2t / (1 + e^-2t)
    e1, e2 = np.exp(-tp), np.exp(-2.0 * tp)
    q = 1.0 + e2
    ph_p = tp + (0.5 * e1 - 0.5 * e1 * e2 - tp * e2) / q
    dph_p = 1.0 + (0.5 * e1 + 0.5 * e1 * e2 - e2) / q - (e1 - e1 * e2 - 2.0 * tp * e2) / (q * q)
    neg = t < 0.0
    return np.where(neg, ph_n, ph_p), np.where(neg, dph_n, dph_p)


def ht_row(r, u, w, h=0.1):
    """Tilted single-layer values ht_r(u) for one row r and grid points u (1-D)."""
    s = float(gammaln(1.0 + w))
    u = np.asarray(u, dtype=np.float64)
    y0 = mode(r, u, w, s)
    f0 = _f(y0, r, u, w, s)
    _, fpp = _fp(y0, r, u, w, s)
    sig = 1.0 / np.sqrt(-fpp)
    # right end: f(y*) - f(y) >= (y - y*)^2 / (2 sig^2), so y_R <= y* + sqrt(2 DROP) sig
    lo, hi = y0.copy(), y0 + np.sqrt(2.0 * DROP) * sig * 1.0001
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        above = _f(mid, r, u, w, s) > f0 - DROP
        lo = np.where(above, mid, lo); hi = np.where(above, hi, mid)
    yR = hi
    _, fppR = _fp(yR, r, u, w, s)
    ell = np.minimum(sig, 1.0 / np.sqrt(-fppR))
    tR = (yR - y0) / ell + 2.0
    tL = np.arcsinh(REACH_L / ell)
    out = np.empty_like(u)
    # group points with similar right reach so that no point pays for another's node count
    order = np.argsort(tR)
    for grp in np.array_split(order, max(1, len(u) // 256)):
        t = np.arange(-int(np.ceil(tL[grp].max() / h)), int(np.ceil(tR[grp].max() / h)) + 1) * h
        ph, dph = _phi(t)
        y = y0[grp, None] + ell[grp, None] * ph[None, :]
        f = _f(y, r, u[grp, None], w, s) + np.log(dph)[None, :]
        f = np.where((t[None, :] <= tR[grp, None]) & (t[None, :] >= -tL[grp, None]), f, -np.inf)
        out[grp] = logsumexp(f, axis=1) + np.log(h * ell[grp])
    return out


def ht_rows(rows, u, w, h=0.1):
    return np.stack([ht_row(float(r), u, w, h) for r in rows])


def closed_form_w1(rows, u):
    """w = 1: ht_r(u) = r u + log Gamma(r+1) - (r+1) log(1 + e^u)."""
    r = np.asarray(rows, dtype=np.float64)[:, None]
    return r * u[None, :] + gammaln(r + 1) - (r + 1) * np.logaddexp(0.0, u[None, :])
