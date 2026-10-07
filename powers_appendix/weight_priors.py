"""Priors nu on the per-coordinate, per-layer power w in Model B.

Model B layer factor: V = E^w, E ~ Exp(1), w ~ nu.  Z = log V has density
    g(z) = int nu_tau(tau) e^{-tau} g_G(z e^{-tau}) dtau,   g_G(y) = exp(y - e^y),
with tau = log w and nu_tau the density of tau.  Everything is evaluated in the
log domain so that log g stays accurate where g itself underflows (the recursion
multiplies g by e^{r z}, so log g ~ -1e3 still matters).

Each prior exposes a quadrature rule in tau (nodes, log weights that include the
density of tau), from which log g, moments and the Mellin transform follow.
`z_shift` is the deterministic constant c removed from Z (Z' = Z - c); theta is
invariant to it.  Bounded priors use c = Lam(1) = log E[V] so that E[Y] = 1 at
every depth, as in the paper; Exp1 has E[V] = inf and uses c = E[Z] = -gamma.
"""
import numpy as np
from scipy.special import gammaln, digamma, logsumexp

EULER = np.euler_gamma


def _log_gG(y):
    """log g_G(y) = y - e^y; -inf once e^y overflows (such points never matter:
    at a kernel peak e^y is of order r*w <= 1e5)."""
    return np.where(y > 700.0, -np.inf, y - np.exp(np.minimum(y, 700.0)))


def _tanh_sinh(a, b, n=401, T=4.0):
    """Tanh-sinh rule on [a, b]; returns nodes and log weights.  Nodes cluster
    doubly-exponentially at both ends, which resolves endpoint-dominated
    integrands (log g at large |z| is dominated by w = b)."""
    t = np.linspace(-T, T, n)
    h = t[1] - t[0]
    s = 0.5 * np.pi * np.sinh(t)
    x = np.tanh(s)
    wx = h * 0.5 * np.pi * np.cosh(t) / np.cosh(s) ** 2
    half = 0.5 * (b - a)
    keep = wx > 1e-300
    return (0.5 * (a + b) + half * x)[keep], np.log(half * wx[keep])


class WeightPrior:
    name = "base"
    bounded = True
    singular = False        # True if nu(0) > 0 (log singularity of g; see modelb_lib)
    w_max = None

    def tau_rule(self):
        """(tau nodes, log[ quadrature weight * density of tau ])."""
        raise NotImplementedError

    # ---- layer log-density -------------------------------------------------
    def log_g_raw(self, z, chunk=512, rule=None):
        """log density of Z = w log E (unshifted), vectorized over z.  `rule`
        restricts the w-integral to a sub-rule (used for the Exp1 split)."""
        tau, lw = self.tau_rule() if rule is None else rule
        z = np.atleast_1d(np.asarray(z, dtype=np.float64))
        out = np.empty_like(z)
        emt = np.exp(-tau)
        base = lw - tau
        for a in range(0, len(z), chunk):
            zz = z[a:a + chunk, None]
            y = zz * emt[None, :]
            out[a:a + chunk] = logsumexp(base[None, :] + _log_gG(y), axis=1)
        return out

    def log_g(self, z):
        """log density of the shifted layer variable Z' = Z - z_shift."""
        return self.log_g_raw(np.asarray(z, dtype=np.float64) + self.z_shift)

    # ---- moments of w and Z --------------------------------------------------
    def _p_w(self):
        tau, lw = self.tau_rule()
        p = np.exp(lw - logsumexp(lw))
        return np.exp(tau), p

    def moments(self):
        w, p = self._p_w()
        Ew, Ew2 = float(p @ w), float(p @ (w * w))
        return {"Ew": Ew, "Ew2": Ew2, "Varw": Ew2 - Ew ** 2,
                "EZ": -EULER * Ew,
                "VarZ": np.pi ** 2 / 6 * Ew2 + EULER ** 2 * (Ew2 - Ew ** 2)}

    # ---- Mellin transform of V (bounded priors only) -----------------------
    def lam(self, s):
        """Lam(s) = log E[V^s] = log E_nu[Gamma(1 + w s)], real s > -1/w_max."""
        if not self.bounded:
            return 0.0 if s == 0 else np.inf    # E[V^s] = inf for every s != 0
        tau, lw = self.tau_rule()
        return float(logsumexp(lw + gammaln(1.0 + np.exp(tau) * s)) - logsumexp(lw))

    def dlam(self, s):
        tau, lw = self.tau_rule()
        w = np.exp(tau)
        a = lw + gammaln(1.0 + w * s)
        p = np.exp(a - logsumexp(a))
        return float(p @ (w * digamma(1.0 + w * s)))

    def cstar(self):
        """Freezing coefficient c*_B = 1/(Lam'(1) - Lam(1)); 0 if Lam(1) = inf."""
        if not self.bounded:
            return 0.0
        return 1.0 / (self.dlam(1.0) - self.lam(1.0))

    shift_override = None    # set to a number to remove a different constant (theta is invariant)

    @property
    def z_shift(self):
        if self.shift_override is not None:
            return self.shift_override
        if self.bounded:
            return self.lam(1.0)
        return -EULER * self.moments()["Ew"]


class Delta(WeightPrior):
    """nu = delta_{w0}; w0 = 1 is the paper."""
    def __init__(self, w0=1.0):
        self.w0 = float(w0)
        self.name = f"delta{w0:g}"
        self.w_max = self.w0

    def tau_rule(self):
        return np.array([np.log(self.w0)]), np.array([0.0])

    def log_g_raw(self, z, chunk=None, rule=None):
        z = np.atleast_1d(np.asarray(z, dtype=np.float64))
        return _log_gG(z / self.w0) - np.log(self.w0)


class LogUniform(WeightPrior):
    """log w ~ Uniform[log(s/b), log(s b)]; s = 1 gives geometric mean 1."""
    def __init__(self, b, s=1.0, n=401):
        assert b > 1.0
        self.b, self.s, self.n = float(b), float(s), n
        self.name = f"logu_b{b:g}" + ("" if s == 1.0 else f"_s{s:g}")
        self.w_max = s * b
        lo, hi = np.log(s) - np.log(b), np.log(s) + np.log(b)
        tau, lw = _tanh_sinh(lo, hi, n)
        self._rule = (tau, lw - np.log(hi - lo))

    def tau_rule(self):
        return self._rule


def _composite_gl(a, b, panel, n):
    """Composite Gauss-Legendre on [a, b] with panels of width <= panel."""
    x, w = np.polynomial.legendre.leggauss(n)
    m = max(1, int(np.ceil((b - a) / panel)))
    edges = np.linspace(a, b, m + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    return ((lo + hi) / 2 + (hi - lo) / 2 * x).ravel(), ((hi - lo) / 2 * w).ravel()


class _SingularAtZero(WeightPrior):
    """Priors with nu(0) > 0: g then has an integrable log singularity at z = -z_shift.
    Composite Gauss-Legendre in tau with log(w_split) as an exact breakpoint, so the
    smooth (w >= w_split) and singular (w < w_split) parts used by the layer kernel
    partition the prior exactly."""
    singular = True

    def _build_rule(self, log_dens, tau_lo, tau_hi, panel, n, w_split):
        """log_dens(tau, log q) -> log(q * density of tau)."""
        self.w_split = w_split
        ts = np.log(w_split)
        t1, q1 = _composite_gl(tau_lo, ts, panel, n)
        t2, q2 = _composite_gl(ts, tau_hi, panel, n)
        tau, q = np.concatenate([t1, t2]), np.concatenate([q1, q2])
        self._rule = (tau, log_dens(tau, np.log(q)))

    def tau_rule(self):
        return self._rule

    def split(self):
        """(singular rule w < w_split, smooth rule w >= w_split)."""
        tau, lw = self._rule
        m = tau < np.log(self.w_split)
        return (tau[m], lw[m]), (tau[~m], lw[~m])


class Exp1(_SingularAtZero):
    """w ~ Exp(1), untruncated.  In tau = log w the density is exp(tau - e^tau)."""
    bounded = False

    def __init__(self, panel=0.1, n=16, tau_lo=-60.0, tau_hi=8.0, w_split=0.1):
        self.name = "exp1"
        self.w_max = np.inf
        self._build_rule(lambda t, lq: lq + t - np.exp(t), tau_lo, tau_hi, panel, n, w_split)


class Uniform(_SingularAtZero):
    """w ~ Uniform[0, a].  In tau = log w the density is e^tau / a on tau < log a.
    Bounded above (finite Mellin transform, c* > 0) but, like Exp(1), nu(0) > 0."""

    def __init__(self, a=2.0, panel=0.1, n=16, tau_lo=-60.0, w_split=0.1, n_ts=401):
        self.a = float(a)
        self.name = f"unif{a:g}"
        self.w_max = self.a
        self.w_split = w_split
        ts = np.log(w_split)
        # w < w_split: composite Gauss-Legendre (as Exp1).  w >= w_split: tanh-sinh, whose
        # nodes cluster at w = a; log g at large z is dominated by that endpoint (as for
        # LogUniform), and Gauss-Legendre there is off by O(100) nats at log g ~ -1e4.
        t1, q1 = _composite_gl(tau_lo, ts, panel, n)
        t2, lq2 = _tanh_sinh(ts, np.log(self.a), n_ts)
        tau = np.concatenate([t1, t2])
        lq = np.concatenate([np.log(q1), lq2])
        self._rule = (tau, lq + tau - np.log(self.a))


def make_prior(spec):
    """'delta', 'logu:2', 'exp1', 'unif:2' -> prior object."""
    if spec in ("delta", "paper"):
        return Delta(1.0)
    if spec.startswith("logu:"):
        return LogUniform(float(spec.split(":")[1]))
    if spec == "exp1":
        return Exp1()
    if spec.startswith("unif:"):
        return Uniform(float(spec.split(":")[1]))
    raise ValueError(spec)
