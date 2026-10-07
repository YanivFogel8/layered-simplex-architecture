"""Power spectrum on the Zipf benchmark (Experiment 6): L layers that share one fixed power w.

    python run_power.py [--procs 20] [--K 22] [--family single|geo|depthcheck] [--only w3,w1_4] [--out results7]

family 'single' (default): L = 1, w = 1..K and 1/2..1/K                      labels w<k>, w1_<k>
family 'geo': L = k layers of power w = 1/k, k = 2..K (Y = geometric mean)    labels g<k>
family 'depthcheck': L = 3, 22 with w = 1 (the paper's depth model; validates the
    Mellin-Barnes tables against the lattice results in results3/delta1.pkl)   labels d<L>

theta_j proportional to E_j^w, E_j ~ Exp(1) i.i.d. (w = 1 is the paper's L = 1, add-one).  Powers
w = 1..K and w = 1/2..1/K, K = round(c* ln d) = 22 (paper's c* = 1/(1 - gamma), d = 1e4).
Same protocol, trials and seeds as run_bench.py (Zipf set, results3/trials.pkl): the table is built
by direct quadrature for L = 1 (power_table.py) or by a Mellin-Barnes integral for L > 1
(mb_table.py), then scored by the same mixture_predictive.
Output per model: results7/pow_<label>.pkl in run_bench's format with ladder [1], plus 'w', 'L'.
"""
import argparse, os, pickle, time
import numpy as np
from protocol import guard
from multiprocessing import Pool
from scipy.special import gammaln
from modelb_lib import Grid, untilt
from power_table import ht_rows
from mb_table import ht_rows as mb_rows
import run_bench as RB

CSTAR = 1.0 / (1.0 - np.euler_gamma)


def powers(K, family="single"):
    """[(label, L, w)]."""
    if family == "geo":
        return [(f"g{k}", k, 1.0 / k) for k in range(2, K + 1)]
    if family == "depthcheck":
        return [(f"d{L}", L, 1.0) for L in (3, 22)]
    return [(f"w{i}", 1, float(i)) for i in range(1, K + 1)] + [(f"w1_{i}", 1, 1.0 / i) for i in range(2, K + 1)]


def grid_for(w, du=0.025, L=1, umin=-40.0, umax=None):
    mu = L * (np.euler_gamma * w + gammaln(1.0 + w))   # as run_chain.setup with Lmax = L
    sd = np.pi * w * np.sqrt(L / 6.0)
    return Grid(umin, max(65.0, mu + 6 * sd) if umax is None else umax, du)


def _tab(args):
    rows, u, L, w = args
    return ht_rows(rows, u, w) if L == 1 else mb_rows(rows, u, L, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=20)
    ap.add_argument("--K", type=int, default=int(round(CSTAR * np.log(RB.K))))
    ap.add_argument("--family", default="single", choices=["single", "geo", "depthcheck"])
    ap.add_argument("--only", help="comma-separated labels")
    ap.add_argument("--out", default="results7")
    ap.add_argument("--trials", default="results3/trials.pkl", help="trial file (run_bench.build_trials format)")
    ap.add_argument("--du", type=float, default=0.025)
    ap.add_argument("--umin", type=float, default=-40.0)
    ap.add_argument("--umax", type=float, help="default: max(65, mu + 6 sd) as run_chain.setup; a narrower "
                    "grid is safe while no saddle comes within 4 of an edge (checked)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(a.trials, "rb") as f:
        trials = pickle.load(f)
    rows = {0, 1}
    for (_, r, _, _, _) in trials.values():
        rows |= set(r.tolist()) | set((r + 1).tolist())
    rows = np.array(sorted(rows))
    keys = sorted(trials)
    ftr_run = f"{a.out}/trials.pkl"
    payload=pickle.dumps(trials,protocol=4)
    if os.path.exists(ftr_run):
        with open(ftr_run,"rb") as f: existing=pickle.load(f)
        if pickle.dumps(existing,protocol=4)!=payload: raise ValueError("Output trials differ; use a new directory")
    else:
        with open(ftr_run,"wb") as f:f.write(payload)
    todo = powers(a.K, a.family)
    if a.only:
        keep = set(a.only.split(","))
        todo = [p for p in todo if p[0] in keep]
    print(f"K = {a.K}: {len(todo)} powers, {len(rows)} rows (max {rows.max()}), {len(trials)} trials", flush=True)
    with Pool(a.procs, initializer=RB._init, initargs=(ftr_run,)) as pool:
        for label, L, w in todo:
            fres = f"{a.out}/pow_{label}.pkl"
            guard(a.out,"pow_"+label,{"trials":a.trials},{"L":L,"w":w,"du":a.du,"umin":a.umin,"umax":a.umax})
            if os.path.exists(fres):
                print(f"{label}: done", flush=True)
                continue
            t0 = time.time()
            grid = grid_for(w, a.du, L, a.umin, a.umax)
            # heaviest rows are not the slowest; interleave so chunks are balanced
            parts = [rows[i::a.procs * 4] for i in range(a.procs * 4)]
            ht = np.empty((len(rows), grid.G))
            for i, part in enumerate(pool.map(_tab, [(p, grid.u, L, w) for p in parts])):
                ht[i::a.procs * 4] = part
            H = untilt(ht, rows, grid)
            hfile = f"{a.out}/_H_{label}.npy"
            np.save(hfile, H)
            t_tab = time.time() - t0
            res = {"logq": {}, "q": {}, "norm": {}, "targets": "main Table 1 row order (see table1_trials.py)", "ns": RB.NS, "T": RB.T,
                   "ladder": [1], "prior": f"delta_{label}", "w": w, "L": L,
                   "grid": (grid.u_min, grid.u_max, grid.du)}
            nmax, bad = 0.0, 0
            for out in pool.imap_unordered(RB._eval, [(keys[i::a.procs * 3], hfile, rows, grid)
                                                      for i in range(a.procs * 3)]):
                for key, lq, q, nrm, at_edge in out:
                    res["logq"][key + (1,)] = lq
                    res["q"][key + (1,)] = q
                    res["norm"][key + (1,)] = nrm
                    nmax = max(nmax, abs(nrm)); bad += at_edge
            os.remove(hfile)
            if bad:
                raise SystemExit(f"{label}: {bad} trials have their saddle at the grid edge")
            with open(fres + ".tmp", "wb") as f:
                pickle.dump(res, f)
            os.replace(fres + ".tmp", fres)
            print(f"{label:6s} L={L:2d} w={w:.4f} {grid} table {t_tab:6.1f}s eval {time.time()-t0-t_tab:5.1f}s | "
                  f"max |sum of predictive - 1| {nmax:.1e}", flush=True)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
