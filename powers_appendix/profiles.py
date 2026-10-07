"""Count profiles of the Zipf-spectrum experiment, regenerated with the paper's
seeds (paper/code/run_spectrum2.py), so every prior sees the same samples."""
import numpy as np

# (d, N) panels of run_spectrum2.py; pi is the paper's panel index (seeds depend on it)
PANELS = {0: (1_000, 316), 1: (10_000, 1000)}
ALPHAS = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
T = 40


def zipf(k, alpha):
    p = np.arange(1, k + 1, dtype=np.float64) ** (-alpha)
    return p / p.sum()


def profile_of(counts):
    pos = counts[counts > 0]
    r, c = np.unique(pos, return_counts=True)
    return r.astype(np.int64), c.astype(np.int64)


def spectrum_profiles(panels=(0, 1)):
    """dict (pi, ai, t) -> (d, N, prof_r, prof_c); also entropies H2[(pi, ai)]."""
    out, H2 = {}, {}
    for pi in panels:
        D, n = PANELS[pi]
        for ai, al in enumerate(ALPHAS):
            p = zipf(D, al)
            H2[(pi, ai)] = float(-np.sum(p * np.log2(p)))
            for t in range(T):
                rng = np.random.default_rng(881_000 + 7919 * pi + 101 * ai + t)
                r, c = profile_of(rng.multinomial(n, p))
                out[(pi, ai, t)] = (D, n, r, c)
    return out, H2


def rows_needed(profs, extra=(0, 1, 2, 3)):
    s = set(extra)
    for (_, _, r, _) in profs.values():
        s |= set(r.tolist())
    return np.array(sorted(s), dtype=np.int64)


if __name__ == "__main__":
    profs, _ = spectrum_profiles()
    rows = rows_needed(profs)
    print(f"{len(profs)} profiles; {len(rows)} distinct rows; max count {rows.max()}")
    print("largest rows:", rows[-15:])
