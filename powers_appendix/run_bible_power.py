"""King James Bible compression (Experiment 4 protocol) for the fixed-power single layers of Experiment 6.

    python run_bible_power.py [--procs 20] [--K 27] [--only w3,w1_4] [--out results12]

Same tokens, alphabet d = 1e5 and checkpoints as run_bible.py; each model is one layer with a common power w
(theta_i proportional to E_i^w), w = 1..K and 1/2..1/K, K = round(c* ln d) = 27 for d = 1e5.  Tables by direct
quadrature (power_table.py; checked on these rows, up to r = 70,579, against the w = 1 closed form and by step
halving to < 1e-9), evidence by the same mixture_predictive.
Output: <out>/bible_pow_<label>.pkl with logq[checkpoint] (nats) and the saddle positions; w1 reproduces
results4/delta1.pkl at L = 1.
"""
import argparse, os, pickle, time
import numpy as np
from protocol import guard,load_tokens,TOKENS
from multiprocessing import Pool
from modelb_lib import untilt, mixture_predictive, LOG2
from power_table import ht_rows
import run_bible as RBi
import run_power as RP


def _tab(args):
    rows, u, w = args
    return ht_rows(rows, u, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=20)
    ap.add_argument("--K", type=int, default=int(round(RP.CSTAR * np.log(RBi.D))))
    ap.add_argument("--only", help="comma-separated labels")
    ap.add_argument("--out", default="results12")
    ap.add_argument("--du", type=float, default=0.025)
    ap.add_argument("--umin", type=float, default=-40.0)
    ap.add_argument("--ckpts", help="comma-separated prefix lengths (default: run_bible.py's)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    toks = load_tokens()
    cks = RBi.checkpoints(toks, [int(x) for x in a.ckpts.split(",")] if a.ckpts else None)
    rows = {0, 1}
    for ck in cks:
        rows |= set(ck["r"].tolist())
    rows = np.array(sorted(rows))
    ridx = {int(r): i for i, r in enumerate(rows)}
    todo = RP.powers(a.K, "single")
    if a.only:
        keep = set(a.only.split(","))
        todo = [p for p in todo if p[0] in keep]
    print(f"K = {a.K}: {len(todo)} models, {len(rows)} rows (max {rows.max()}), checkpoints "
          f"{[ck['n'] for ck in cks]}", flush=True)
    with Pool(a.procs) as pool:
        for label, L, w in todo:
            fres = f"{a.out}/bible_pow_{label}.pkl"
            guard(a.out,"bible_pow_"+label,{"tokens":TOKENS},{"L":L,"w":w,"du":a.du,"umin":a.umin,"ckpts":[ck["n"] for ck in cks]})
            if os.path.exists(fres):
                print(f"{label}: done", flush=True)
                continue
            t0 = time.time()
            grid = RP.grid_for(w, a.du, L, a.umin)
            nchunk = a.procs * 4
            ht = np.empty((len(rows), grid.G))
            for i, part in enumerate(pool.map(_tab, [(rows[i::nchunk], grid.u, w) for i in range(nchunk)])):
                ht[i::nchunk] = part
            H = untilt(ht, rows, grid)
            edge = 4.0 / grid.du + 1
            res = {"label": label, "w": w, "L": L, "grid": (grid.u_min, grid.u_max, grid.du),
                   "ckpts": [ck["n"] for ck in cks], "H": [ck["H"] for ck in cks],
                   "distinct": [ck["distinct"] for ck in cks], "logq": [], "saddle": []}
            for ck in cks:
                lq, _, (i0, us) = mixture_predictive(ck["n"], RBi.D, ck["r"], ck["c"], H, ridx, grid,
                                                     want_ratios=False, spline_pad=60)
                if i0 < edge or i0 > grid.G - 1 - edge:
                    raise SystemExit(f"{label}: saddle at grid edge (u={us:.1f}) for n={ck['n']}")
                res["logq"].append(lq); res["saddle"].append(us)
            res["logq"], res["saddle"] = np.array(res["logq"]), np.array(res["saddle"])
            with open(fres + ".tmp", "wb") as f:
                pickle.dump(res, f)
            os.replace(fres + ".tmp", fres)
            red = [-lq / LOG2 / ck["n"] - ck["H"] for lq, ck in zip(res["logq"], cks)]
            print(f"{label:6s} w={w:.4f} {grid} [{time.time() - t0:5.1f}s] | redundancy "
                  + " ".join(f"{v:.4f}" for v in red), flush=True)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
