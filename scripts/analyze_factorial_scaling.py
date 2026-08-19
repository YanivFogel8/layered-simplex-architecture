"""Summarize the independent scaling-law grid into named experiments."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

from data_scaling_experiment import (
    DataScalingResult,
    predicted_beta,
    read_results_csv,
    write_summary_csv,
)
from small_l_vs_log_depth_experiment import power_fit


def valid_rows(rows: list[DataScalingResult]) -> list[DataScalingResult]:
    return [
        row
        for row in rows
        if row.failed_profile_samples == 0
        and math.isfinite(row.regret_bits)
        and row.regret_bits > 0.0
    ]


def normalized_prefactor(row: DataScalingResult) -> float:
    beta = predicted_beta(row.alpha)
    if beta <= 0.0:
        raise ValueError("the marginal alpha <= 1 case has no predicted power")
    return row.regret_bits * row.N**beta / math.log(row.d)


def write_alphabet_summary(path: Path, rows: list[DataScalingResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for alpha in sorted({row.alpha for row in rows if row.alpha > 1.0}):
        for n in sorted({row.N for row in rows if row.alpha == alpha}):
            curve = sorted(
                [row for row in rows if row.alpha == alpha and row.N == n],
                key=lambda row: row.d,
            )
            if len(curve) < 2:
                continue
            values = [normalized_prefactor(row) for row in curve]
            _, residual_beta, residual_r2 = power_fit(
                [float(row.d) for row in curve], values
            )
            mean = statistics.fmean(values)
            records.append(
                {
                    "experiment": "Alphabet scaling",
                    "alpha": f"{alpha:g}",
                    "N": n,
                    "point_count": len(curve),
                    "d_min": curve[0].d,
                    "d_max": curve[-1].d,
                    "mean_normalized_prefactor": f"{mean:.12g}",
                    "coefficient_of_variation": f"{statistics.pstdev(values)/mean:.12g}",
                    "residual_beta_vs_d": f"{residual_beta:.12g}",
                    "residual_r2": f"{residual_r2:.12g}",
                }
            )
    fields = list(records[0]) if records else ["experiment"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_collapse_rows(path: Path, rows: list[DataScalingResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment", "d", "N", "alpha", "c_value", "L",
        "predicted_beta", "normalized_prefactor", "relative_standard_error",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.alpha, item.d, item.N)):
            if row.alpha <= 1.0:
                continue
            value = normalized_prefactor(row)
            writer.writerow(
                {
                    "experiment": "Scaling-law collapse",
                    "d": row.d,
                    "N": row.N,
                    "alpha": f"{row.alpha:g}",
                    "c_value": f"{row.c_value:.12g}",
                    "L": row.L,
                    "predicted_beta": f"{predicted_beta(row.alpha):.12g}",
                    "normalized_prefactor": f"{value:.12g}",
                    "relative_standard_error": f"{row.stderr_bits/row.regret_bits:.12g}",
                }
            )


def write_collapse_summary(path: Path, rows: list[DataScalingResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment", "alpha", "cell_count", "mean_normalized_prefactor",
        "coefficient_of_variation", "minimum", "maximum", "max_relative_deviation",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for alpha in sorted({row.alpha for row in rows if row.alpha > 1.0}):
            values = [normalized_prefactor(row) for row in rows if row.alpha == alpha]
            mean = statistics.fmean(values)
            writer.writerow(
                {
                    "experiment": "Scaling-law collapse",
                    "alpha": f"{alpha:g}",
                    "cell_count": len(values),
                    "mean_normalized_prefactor": f"{mean:.12g}",
                    "coefficient_of_variation": f"{statistics.pstdev(values)/mean:.12g}",
                    "minimum": f"{min(values):.12g}",
                    "maximum": f"{max(values):.12g}",
                    "max_relative_deviation": f"{max(abs(value/mean-1) for value in values):.12g}",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-csv", type=Path, required=True)
    parser.add_argument("--fit-min-N", type=int, default=100)
    parser.add_argument(
        "--out-data-summary",
        type=Path,
        default=Path("output/scaling_law_factorial/data_scaling_summary.csv"),
    )
    parser.add_argument(
        "--out-alphabet-summary",
        type=Path,
        default=Path("output/scaling_law_factorial/alphabet_scaling_summary.csv"),
    )
    parser.add_argument(
        "--out-collapse-csv",
        type=Path,
        default=Path("output/scaling_law_factorial/scaling_law_collapse.csv"),
    )
    parser.add_argument(
        "--out-collapse-summary",
        type=Path,
        default=Path("output/scaling_law_factorial/scaling_law_collapse_summary.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = valid_rows(read_results_csv(args.from_csv))
    if not rows:
        raise ValueError("input contains no valid rows")
    write_summary_csv(args.out_data_summary, rows, fit_min_N=args.fit_min_N)
    write_alphabet_summary(args.out_alphabet_summary, rows)
    write_collapse_rows(args.out_collapse_csv, rows)
    write_collapse_summary(args.out_collapse_summary, rows)
    for path in (
        args.out_data_summary,
        args.out_alphabet_summary,
        args.out_collapse_csv,
        args.out_collapse_summary,
    ):
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
