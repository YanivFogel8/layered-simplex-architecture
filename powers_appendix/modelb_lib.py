"""Numerics for Model B: product of L i.i.d. layer factors V = E^w, w ~ nu.

Tables store the tilted kernels  ht_r(u) = h_r(u) + r u = log E[(tY)^r e^{-tY}],
t = e^u.  In this form one layer is a pure convolution with the layer density g
of Z' = log V - z_shift, with the same kernel for every row r:

    ht_r^{(l)}(u) = log int g(z) exp( ht_r^{(l-1)}(u + z) ) dz ,
    ht_r^{(0)}(u) = r u - e^u .

The z-integral is a lattice sum on offsets z_k = k du (so u + z_k is a grid
point; no interpolation), with log-weights lw_k.  For priors whose g is smooth
lw_k = log g(z_k) + log du (trapezoid, spectrally accurate).  For priors with
nu(0) > 0 (Exp(1), Uniform[0, a]), whose g has an integrable log singularity from
w -> 0, the part w < w_c instead gets product-integration weights exact for cubic
interpolants of exp(ht).

The outer profile integral is the paper's `mixture_predictive`, with the grid
passed in instead of module globals.
"""
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.interpolate import CubicSpline
from weight_priors import _log_gG

LOG2 = np.log(2.0)


class Grid:
    def __init__(self, u_min, u_max, du):
        self.du = float(du)
        self.i_min = int(round(u_min / du))
        self.u = np.arange(self.i_min, int(round(u_max / du)) + 1) * self.du
        self.u_min, self.u_max, self.G = self.u[0], self.u[-1], len(self.u)

    def __repr__(self):
        return f"Grid[{self.u_min:g},{self.u_max:g}]/{self.du:g} (G={self.G})"


# ---------------------------------------------------------------- layer kernel
def _cubic_basis(f):
    """4-point Lagrange weights for nodes -1,0,1,2 at fractional position f in [0,1)."""
    return np.stack([-f * (f - 1) * (f - 2) / 6, (f + 1) * (f - 1) * (f - 2) / 2,
                     -(f + 1) * f * (f - 2) / 2, (f + 1) * f * (f - 1) / 6])


def _singular_weights(prior, du, k_lo, k_hi, dx=0.002, tau_point=-30.0):
    """Product-integration weights for the w < w_split part of a prior with nu(0) > 0
    (Exp1, Uniform):
    sum_k W_k F(z_k) = int_{w<w_split} nu(dw) int g_G(x) Fc(w x - c) dx, where Fc
    is the piecewise-cubic interpolant of F on the lattice (exact for cubic F).
    tau-nodes below tau_point (|log V| < 1e-11) are merged into a point mass."""
    (tau_s, lw_s), _ = prior.split()
    c = prior.z_shift
    nW = k_hi - k_lo + 1
    W = np.zeros(nW)
    x = np.arange(-45.0, 4.0, dx)
    wx = np.exp(_log_gG(x)) * dx
    mass = np.exp(lw_s)

    def deposit(z, wt):
        pos = z / du - k_lo
        i0 = np.floor(pos).astype(np.int64)
        basis = _cubic_basis(pos - i0)
        for j, off in enumerate((-1, 0, 1, 2)):
            W[:] += np.bincount(i0 + off, weights=wt * basis[j], minlength=nW)

    far = tau_s >= tau_point
    for t, m in zip(tau_s[far], mass[far]):
        deposit(np.exp(t) * x - c, m * wx)
    deposit(np.array([-c]), np.array([mass[~far].sum()]))
    return W


def layer_kernel(prior, grid, z_lo, z_hi):
    """Integer offsets k and log-weights lw_k for one layer step."""
    k = np.arange(int(np.floor(z_lo / grid.du)), int(np.ceil(z_hi / grid.du)) + 1)
    z = k * grid.du
    if not prior.singular:
        return k, prior.log_g(z) + np.log(grid.du)
    _, smooth_rule = prior.split()
    lsm = prior.log_g_raw(z + prior.z_shift, rule=smooth_rule) + np.log(grid.du)
    W = _singular_weights(prior, grid.du, k[0], k[-1])
    # cubic product weights are slightly negative in places (Lagrange basis);
    # keep the signed sum: total weight = smooth + singular, per offset
    lw = lsm.copy()
    m = W != 0
    tot = np.exp(lsm[m]) + W[m]
    if np.any(tot <= 0):
        raise ValueError(f"non-positive combined kernel weight {tot.min():.2e}")
    lw[m] = np.log(tot)
    return k, lw


def ht_base(rows, grid):
    r = np.asarray(rows, dtype=np.float64)[:, None]
    return r * grid.u[None, :] - np.exp(grid.u)[None, :]


try:
    from numba import njit

    @njit(cache=True, fastmath=False)
    def _lattice_lse(ext, lw, base, G, cut):
        """out[i] = logsumexp_k (ext[base + i + k] + lw[k]); terms more than `cut`
        below the running max are skipped (e^-cut relative, ~1e-22 at cut=50)."""
        K = lw.shape[0]
        out = np.empty(G)
        for i in range(G):
            j0 = base + i
            mx = -np.inf
            for kk in range(K):
                v = ext[j0 + kk] + lw[kk]
                if v > mx:
                    mx = v
            s = 0.0
            thr = mx - cut
            for kk in range(K):
                v = ext[j0 + kk] + lw[kk]
                if v > thr:
                    s += np.exp(v - mx)
            out[i] = mx + np.log(s)
        return out
    @njit(cache=True)
    def _lattice_lse_fine(ext, lw, q, B, base, G, cut):
        """As _lattice_lse, but on sub-lattice offsets: term j reads ext at the
        fractional index base + i + q[j] + f_j through 4-point cubic weights B[j]
        (stencil q[j]-1 .. q[j]+2), i.e. ht is interpolated in the log domain."""
        K = lw.shape[0]
        out = np.empty(G)
        tmp = np.empty(K)
        for i in range(G):
            mx = -np.inf
            for j in range(K):
                p = base + i + q[j]
                v = (B[j, 0] * ext[p - 1] + B[j, 1] * ext[p] + B[j, 2] * ext[p + 1]
                     + B[j, 3] * ext[p + 2]) + lw[j]
                tmp[j] = v
                if v > mx:
                    mx = v
            s = 0.0
            thr = mx - cut
            for j in range(K):
                if tmp[j] > thr:
                    s += np.exp(tmp[j] - mx)
            out[i] = mx + np.log(s)
        return out

    @njit(cache=True)
    def _lattice_lse_base(u, lw, z, r, cut):
        """First layer from the closed form ht^(0)(v) = r v - e^v (no interpolation)."""
        G, K = u.shape[0], lw.shape[0]
        out = np.empty(G)
        tmp = np.empty(K)
        for i in range(G):
            mx = -np.inf
            for j in range(K):
                v = u[i] + z[j]
                t = lw[j] + r * v - np.exp(min(v, 700.0))
                tmp[j] = t
                if t > mx:
                    mx = t
            s = 0.0
            for j in range(K):
                if tmp[j] > mx - cut:
                    s += np.exp(tmp[j] - mx)
            out[i] = mx + np.log(s)
        return out
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


def substeps(r, du, factor=1.2):
    """Sub-lattice refinement m for row r: the layer-0 bump of ht_r has width
    ~1/sqrt(r+1); keep >= `factor` z-steps per width (m = 1 for r <~ 1100 at du=0.025)."""
    return np.maximum(1, np.ceil(factor * du * np.sqrt(np.asarray(r, dtype=float) + 1.0))).astype(int)


def fine_kernel(prior, grid, z_lo, z_hi, m):
    """Kernel on the sub-lattice du/m, returned as (integer part q, cubic weights B, lw)."""
    fg = Grid(grid.u_min, grid.u_max, grid.du / m)
    kf, lw = layer_kernel(prior, fg, z_lo, z_hi)
    q = np.floor_divide(kf, m)
    f = (kf - q * m) / m
    return q.astype(np.int64), np.ascontiguousarray(_cubic_basis(f).T), lw, kf * fg.du


def advance_fine(ht, rows, kern, grid, first_layer, cut=50.0):
    """One layer for rows needing a sub-lattice.  kern: dict m -> fine_kernel(...).
    first_layer: use the closed form of ht^(0) instead of interpolating the table."""
    out = np.empty_like(ht)
    for i, r in enumerate(rows):
        m = int(substeps(r, grid.du))
        q, B, lw, z = kern[m]
        if first_layer:
            out[i] = _lattice_lse_base(grid.u, lw, z, float(r), cut)
            continue
        row = ht[i]
        padL, padR = max(0, -int(q[0])) + 2, max(0, int(q[-1])) + 3
        sL, sR = row[1] - row[0], row[-1] - row[-2]
        ext = np.concatenate([row[0] + sL * np.arange(-padL, 0), row,
                              row[-1] + sR * np.arange(1, padR + 1)])
        out[i] = _lattice_lse_fine(ext, lw, q, B, padL, grid.G, cut)
    return out


def advance(ht, k, lw, grid, chunk=256, cut=50.0):
    """One layer: ht (R, G) -> ht' (R, G).  Values read outside the grid are
    extrapolated linearly from the two edge points."""
    R, G = ht.shape
    if HAVE_NUMBA:
        k_lo, k_hi = int(k[0]), int(k[-1])
        padL, padR = max(0, -k_lo), max(0, k_hi)
        out = np.empty_like(ht)
        for r in range(R):
            row = ht[r]
            sL, sR = row[1] - row[0], row[-1] - row[-2]
            ext = np.concatenate([row[0] + sL * np.arange(-padL, 0), row,
                                  row[-1] + sR * np.arange(1, padR + 1)])
            out[r] = _lattice_lse(ext, lw, padL + k_lo, G, cut)
        return out
    k_lo, k_hi = int(k[0]), int(k[-1])
    padL, padR = max(0, -k_lo), max(0, k_hi)
    out = np.empty_like(ht)
    sw = np.lib.stride_tricks.sliding_window_view
    for r in range(R):
        row = ht[r]
        sL, sR = row[1] - row[0], row[-1] - row[-2]
        ext = np.concatenate([row[0] + sL * np.arange(-padL, 0),
                              row,
                              row[-1] + sR * np.arange(1, padR + 1)])
        # output i reads ext[padL + i + k_lo : padL + i + k_hi + 1]
        win = sw(ext, len(k))
        base = padL + k_lo
        for a in range(0, G, chunk):
            b = min(a + chunk, G)
            m = win[base + a: base + b] + lw[None, :]
            mx = m.max(axis=1)
            out[r, a:b] = mx + np.log(np.exp(m - mx[:, None]).sum(axis=1))
    return out


# ------------------------------------------------------- profile integral
def mixture_predictive(N, d, prof_r, prof_c, H, rowidx, grid, want_ratios=True, win=4.0, spline_pad=None):
    """Paper's simplex_lib.mixture_predictive with an explicit grid.
    H: (R, G) table of h_r (NOT tilted); rowidx: dict r -> row of H."""
    UG, DU, G = grid.u, grid.du, grid.G
    prof_r = np.asarray(prof_r, dtype=np.int64)
    prof_c = np.asarray(prof_c, dtype=np.float64)
    s = int(np.sum(prof_c))
    need = sorted(set([0]) | set(prof_r.tolist())
                  | (set((prof_r + 1).tolist()) | {1} if want_ratios else set()))
    sub = H[[rowidx[r] for r in need]]
    row = {r: i for i, r in enumerate(need)}

    psi_c = N * UG + (d - s) * sub[row[0]]
    for r, c in zip(prof_r, prof_c):
        psi_c = psi_c + c * sub[row[r]]
    i0 = int(np.argmax(psi_c))
    wcells = int(round(win / DU))
    ilo, ihi = max(i0 - wcells, 0), min(i0 + wcells, G - 1)
    if spline_pad is None:
        spl = CubicSpline(UG, sub, axis=1)                  # paper: spline on the whole grid
    else:
        # spline on the window plus a margin: the end-condition influence decays ~0.27
        # per knot, so values inside the window equal the full-grid spline's
        jlo, jhi = max(ilo - spline_pad, 0), min(ihi + spline_pad, G - 1)
        spl = CubicSpline(UG[jlo:jhi + 1], sub[:, jlo:jhi + 1], axis=1)
    n_sub = 40
    ud = np.linspace(UG[ilo], UG[ihi], (ihi - ilo) * n_sub + 1)
    dud = ud[1] - ud[0]
    sub_d = spl(ud)
    psi_d = N * ud + (d - s) * sub_d[row[0]]
    for r, c in zip(prof_r, prof_c):
        psi_d = psi_d + c * sub_d[row[r]]
    logw_in = np.full_like(ud, np.log(dud)); logw_in[0] -= LOG2; logw_in[-1] -= LOG2
    outside = np.ones(G, dtype=bool); outside[ilo + 1: ihi] = False
    logw_out = np.full(G, np.log(DU))
    logw_out[0] -= LOG2; logw_out[-1] -= LOG2
    logw_out[ilo] -= LOG2; logw_out[ihi] -= LOG2

    def integrate(extra_dense, extra_coarse):
        val_in = logsumexp(psi_d + extra_dense + logw_in)
        g = psi_c + extra_coarse + logw_out
        return np.logaddexp(val_in, logsumexp(g[outside]))

    logI = integrate(0.0, 0.0)
    logq = -gammaln(N) + logI
    preds = {}
    if want_ratios:
        for r in ([0] if d > s else []) + prof_r.tolist():
            dd = ud + sub_d[row[r + 1]] - sub_d[row[r]]
            dc = UG + sub[row[r + 1]] - sub[row[r]]
            preds[r] = np.exp(integrate(dd, dc) - logI - np.log(N))
    return logq, preds, (i0, UG[i0])


def untilt(ht, rows, grid):
    """h_r(u) = ht_r(u) - r u."""
    return ht - np.asarray(rows, dtype=np.float64)[:, None] * grid.u[None, :]
