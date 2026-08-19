"""Compare free-power and finite-size-corrected fits for the alpha=2 sweep."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from data_scaling_experiment import DataScalingResult, read_results_csv
from small_l_vs_log_depth_experiment import power_fit


def corrected_fit(rows: list[DataScalingResult]) -> tuple[float, float, float, float]:
    """Fit R/log(d) = A*N^-1/2 + B*N^-1; return A, B, R2, RMSE."""
    y = np.asarray([row.regret_over_ln_d for row in rows], dtype=float)
    n = np.asarray([row.N for row in rows], dtype=float)
    design = np.column_stack((n ** -0.5, n ** -1.0))
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residual = y - fitted
    ss_res = float(residual @ residual)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(coefficients[0]), float(coefficients[1]), r2, math.sqrt(ss_res / len(y))


def label_cost_fit(rows: list[DataScalingResult]) -> tuple[float, float, float, float, float]:
    """Fit sqrt(N) R = a log(d) + b log(N) + intercept across d and N."""
    target = np.asarray([math.sqrt(row.N) * row.regret_bits for row in rows])
    design = np.asarray([[math.log(row.d), math.log(row.N), 1.0] for row in rows])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ coefficients
    residual = target - fitted
    ss_res = float(residual @ residual)
    ss_tot = float(((target - target.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    a, b, intercept = map(float, coefficients)
    return a, b, intercept, r2, math.sqrt(ss_res / len(target))


def summarize(rows: list[DataScalingResult], *, fit_min_n: int) -> list[dict[str, object]]:
    usable = [row for row in rows if row.alpha == 2 and row.N >= fit_min_n
              and row.failed_profile_samples == 0 and row.regret_over_ln_d > 0]
    records = []
    for d, c_value in sorted({(row.d, row.c_value) for row in usable}):
        curve = sorted([row for row in usable if row.d == d and row.c_value == c_value],
                       key=lambda row: row.N)
        if len(curve) < 3:
            continue
        amplitude, beta, free_r2 = power_fit(
            [float(row.N) for row in curve], [row.regret_over_ln_d for row in curve]
        )
        free_errors = [row.regret_over_ln_d - amplitude * row.N ** (-beta) for row in curve]
        free_rmse = math.sqrt(sum(value * value for value in free_errors) / len(curve))
        a, b, corrected_r2, corrected_rmse = corrected_fit(curve)
        records.append({
            "experiment": "Data scaling: alpha=2 diagnostic", "d": d,
            "c_value": f"{c_value:.12g}", "L": curve[0].L,
            "point_count": len(curve), "N_min": curve[0].N, "N_max": curve[-1].N,
            "free_amplitude": f"{amplitude:.12g}", "free_beta": f"{beta:.12g}",
            "free_r2": f"{free_r2:.12g}", "free_rmse": f"{free_rmse:.12g}",
            "corrected_A": f"{a:.12g}", "corrected_B": f"{b:.12g}",
            "corrected_r2": f"{corrected_r2:.12g}",
            "corrected_rmse": f"{corrected_rmse:.12g}",
            "rmse_ratio_corrected_over_free": f"{corrected_rmse/free_rmse:.12g}",
        })
    return records


def summarize_label_cost(rows: list[DataScalingResult], *, fit_min_n: int) -> list[dict[str, object]]:
    usable = [row for row in rows if row.alpha == 2 and row.N >= fit_min_n
              and row.failed_profile_samples == 0 and row.regret_bits > 0]
    records = []
    for c_value in sorted({row.c_value for row in usable}):
        curve = [row for row in usable if row.c_value == c_value]
        if len({row.d for row in curve}) < 2 or len(curve) < 6:
            continue
        a, b, intercept, r2, rmse = label_cost_fit(curve)
        records.append({
            "experiment": "Alphabet scaling: alpha=2 label-cost diagnostic",
            "c_value": f"{c_value:.12g}", "cell_count": len(curve),
            "d_count": len({row.d for row in curve}), "N_min": min(row.N for row in curve),
            "N_max": max(row.N for row in curve), "coefficient_log_d": f"{a:.12g}",
            "coefficient_log_N": f"{b:.12g}", "intercept": f"{intercept:.12g}",
            "log_N_over_log_d_ratio": f"{b/a:.12g}", "predicted_ratio": "-0.5",
            "r2": f"{r2:.12g}", "rmse": f"{rmse:.12g}",
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--fit-min-N", type=int, default=500)
    parser.add_argument("--out", type=Path,
                        default=Path("output/scaling_law_factorial/alpha2_depth_diagnostic_summary.csv"))
    parser.add_argument("--out-label-cost", type=Path,
                        default=Path("output/scaling_law_factorial/alpha2_label_cost_summary.csv"))
    args = parser.parse_args()
    rows = [row for path in args.csv for row in read_results_csv(path)]
    records = summarize(rows, fit_min_n=args.fit_min_N)
    if not records:
        raise ValueError("no complete alpha=2 curves")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {args.out}")
    label_records = summarize_label_cost(rows, fit_min_n=args.fit_min_N)
    if label_records:
        args.out_label_cost.parent.mkdir(parents=True, exist_ok=True)
        with args.out_label_cost.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(label_records[0]))
            writer.writeheader()
            writer.writerows(label_records)
        print(f"wrote {args.out_label_cost}")


if __name__ == "__main__":
    main()
