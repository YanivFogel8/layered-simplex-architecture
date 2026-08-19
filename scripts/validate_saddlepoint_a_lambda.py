from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from exact_online_regret import (
    integer_partitions,
    multinomial_pattern_weights_fast,
    multinomial_pattern_weights_power_sum_mp,
    zipf_distribution,
)
from lsa.pattern_weights import log_a_lambda_hybrid, log_a_lambda_saddlepoint


@dataclass(frozen=True)
class SummaryRow:
    alpha: float
    n: int
    partitions_total: int
    checked: int
    checked_mass: float
    exact_weight_sum: float
    approx_checked_mass: float
    weighted_mean_log_error: float
    weighted_abs_log_error: float
    weighted_rmse_log_error: float
    max_abs_log_error: float
    worst_partition: tuple[int, ...]
    converged: int
    failed: int


@dataclass(frozen=True)
class DetailRow:
    alpha: float
    n: int
    partition: tuple[int, ...]
    exact_weight: float
    approx_log_weight: float
    log_error: float
    converged: bool
    residual_norm: float
    top_count: int


def uniform_distribution(d: int) -> np.ndarray:
    return np.full(d, 1.0 / d, dtype=np.float64)


def distribution_for_alpha(d: int, alpha: float) -> np.ndarray:
    if alpha == 0.0:
        return uniform_distribution(d)
    return zipf_distribution(d, alpha)


def exact_uniform_pattern_weights(d: int, n: int) -> dict[tuple[int, ...], float]:
    return {
        partition: math.exp(exact_uniform_log_pattern_weight(d, partition))
        for partition in integer_partitions(n)
    }


def exact_uniform_log_pattern_weight(d: int, partition: tuple[int, ...]) -> float:
    counts = Counter(partition)
    observed_symbols = len(partition)
    log_value = (
        math.lgamma(sum(partition) + 1)
        - sum(partition) * math.log(d)
        + math.lgamma(d + 1)
        - math.lgamma(d - observed_symbols + 1)
    )
    for part, multiplicity in counts.items():
        log_value -= multiplicity * math.lgamma(part + 1)
        log_value -= math.lgamma(multiplicity + 1)
    return log_value


def exact_pattern_weights(
    p: np.ndarray,
    n: int,
    *,
    alpha: float,
) -> dict[tuple[int, ...], float]:
    if alpha == 0.0:
        return exact_uniform_pattern_weights(p.size, n)
    weights = multinomial_pattern_weights_fast(p, n)
    if any(weight <= 0.0 for weight in weights.values()):
        weights = multinomial_pattern_weights_power_sum_mp(p, n)
    return weights


def select_partitions(
    weights: dict[tuple[int, ...], float],
    *,
    exhaustive: bool,
    top_partitions: int,
) -> list[tuple[int, ...]]:
    if exhaustive:
        return list(weights)
    return [
        partition
        for partition, _ in sorted(
            weights.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_partitions]
    ]


def validate_one_n(
    *,
    p: np.ndarray,
    alpha: float,
    n: int,
    exhaustive: bool,
    top_partitions: int,
    chunk_size: int,
    method: str,
    top_count: int | None,
    min_expected_count: float,
    max_top_states: int,
) -> tuple[SummaryRow, list[DetailRow]]:
    weights = exact_pattern_weights(p, n, alpha=alpha)
    selected = select_partitions(
        weights,
        exhaustive=exhaustive,
        top_partitions=top_partitions,
    )

    details = []
    exact_selected_weights = []
    approx_selected_weights = []
    log_errors = []
    converged = 0

    for partition in selected:
        exact_weight = weights[partition]
        if exact_weight <= 0.0:
            raise RuntimeError(
                f"non-positive exact weight after high-precision fallback: "
                f"N={n}, alpha={alpha:g}, partition={partition}, "
                f"weight={exact_weight}"
            )
        if method == "saddlepoint":
            approximation = log_a_lambda_saddlepoint(
                p,
                partition,
                chunk_size=chunk_size,
            )
            residual_norm = approximation.residual_norm
            approximation_top_count = 0
        elif method == "hybrid":
            approximation = log_a_lambda_hybrid(
                p,
                partition,
                top_count=top_count,
                min_expected_count=min_expected_count,
                chunk_size=chunk_size,
                max_top_states=max_top_states,
            )
            residual_norm = approximation.max_tail_residual_norm
            approximation_top_count = approximation.top_count
        else:
            raise ValueError(f"unknown method: {method}")
        if approximation.converged:
            converged += 1
        log_error = approximation.log_weight - math.log(exact_weight)
        details.append(
            DetailRow(
                alpha=alpha,
                n=n,
                partition=partition,
                exact_weight=exact_weight,
                approx_log_weight=approximation.log_weight,
                log_error=log_error,
                converged=approximation.converged,
                residual_norm=residual_norm,
                top_count=approximation_top_count,
            )
        )
        exact_selected_weights.append(exact_weight)
        approx_selected_weights.append(math.exp(approximation.log_weight))
        log_errors.append(log_error)

    exact_selected = np.array(exact_selected_weights, dtype=np.float64)
    approx_selected = np.array(approx_selected_weights, dtype=np.float64)
    errors = np.array(log_errors, dtype=np.float64)
    checked_mass = float(np.sum(exact_selected, dtype=np.float64))
    if checked_mass > 0.0:
        normalized_weights = exact_selected / checked_mass
        weighted_mean = float(np.sum(normalized_weights * errors, dtype=np.float64))
        weighted_abs = float(
            np.sum(normalized_weights * np.abs(errors), dtype=np.float64)
        )
        weighted_rmse = float(
            math.sqrt(np.sum(normalized_weights * errors * errors, dtype=np.float64))
        )
    else:
        weighted_mean = math.nan
        weighted_abs = math.nan
        weighted_rmse = math.nan

    if errors.size:
        worst_index = int(np.argmax(np.abs(errors)))
        max_abs = float(abs(errors[worst_index]))
        worst_partition = selected[worst_index]
    else:
        max_abs = math.nan
        worst_partition = ()

    summary = SummaryRow(
        alpha=alpha,
        n=n,
        partitions_total=len(weights),
        checked=len(selected),
        checked_mass=checked_mass,
        exact_weight_sum=float(sum(weights.values())),
        approx_checked_mass=float(np.sum(approx_selected, dtype=np.float64)),
        weighted_mean_log_error=weighted_mean,
        weighted_abs_log_error=weighted_abs,
        weighted_rmse_log_error=weighted_rmse,
        max_abs_log_error=max_abs,
        worst_partition=worst_partition,
        converged=converged,
        failed=len(selected) - converged,
    )
    return summary, details


def parse_csv_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def write_summary(path: Path, rows: list[SummaryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "alpha",
                "n",
                "partitions_total",
                "checked",
                "checked_mass",
                "exact_weight_sum",
                "approx_checked_mass",
                "weighted_mean_log_error",
                "weighted_abs_log_error",
                "weighted_rmse_log_error",
                "max_abs_log_error",
                "worst_partition",
                "converged",
                "failed",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "alpha": f"{row.alpha:g}",
                    "n": row.n,
                    "partitions_total": row.partitions_total,
                    "checked": row.checked,
                    "checked_mass": f"{row.checked_mass:.12g}",
                    "exact_weight_sum": f"{row.exact_weight_sum:.12g}",
                    "approx_checked_mass": f"{row.approx_checked_mass:.12g}",
                    "weighted_mean_log_error": f"{row.weighted_mean_log_error:.12g}",
                    "weighted_abs_log_error": f"{row.weighted_abs_log_error:.12g}",
                    "weighted_rmse_log_error": f"{row.weighted_rmse_log_error:.12g}",
                    "max_abs_log_error": f"{row.max_abs_log_error:.12g}",
                    "worst_partition": "+".join(map(str, row.worst_partition)),
                    "converged": row.converged,
                    "failed": row.failed,
                }
            )


def write_details(path: Path, rows: list[DetailRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "alpha",
                "n",
                "partition",
                "exact_weight",
                "approx_log_weight",
                "log_error",
                "converged",
                "residual_norm",
                "top_count",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "alpha": f"{row.alpha:g}",
                    "n": row.n,
                    "partition": "+".join(map(str, row.partition)),
                    "exact_weight": f"{row.exact_weight:.12g}",
                    "approx_log_weight": f"{row.approx_log_weight:.12g}",
                    "log_error": f"{row.log_error:.12g}",
                    "converged": int(row.converged),
                    "residual_norm": f"{row.residual_norm:.12g}",
                    "top_count": row.top_count,
                }
            )


def print_summary(rows: list[SummaryRow], *, n_values: set[int]) -> None:
    for alpha in sorted({row.alpha for row in rows}):
        print(f"alpha={alpha:g}")
        print(
            "  N  checked/total   checked_mass  mean_err  abs_err   rmse"
            "     max_abs  failed"
        )
        for row in rows:
            if row.alpha != alpha or row.n not in n_values:
                continue
            print(
                f"  {row.n:2d} "
                f"{row.checked:7d}/{row.partitions_total:<7d} "
                f"{row.checked_mass:12.6g} "
                f"{row.weighted_mean_log_error:9.3g} "
                f"{row.weighted_abs_log_error:8.3g} "
                f"{row.weighted_rmse_log_error:8.3g} "
                f"{row.max_abs_log_error:9.3g} "
                f"{row.failed:6d}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--alphas", type=str, default="0,1,2,3")
    parser.add_argument("--n-max", type=int, default=50)
    parser.add_argument("--exhaustive-n-max", type=int, default=20)
    parser.add_argument("--top-partitions", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=65_536)
    parser.add_argument(
        "--method",
        choices=["saddlepoint", "hybrid"],
        default="saddlepoint",
    )
    parser.add_argument("--top-count", type=int, default=None)
    parser.add_argument("--min-expected-count", type=float, default=5.0)
    parser.add_argument("--max-top-states", type=int, default=200_000)
    parser.add_argument(
        "--summary-n-values",
        type=str,
        default="1,2,5,10,20,30,40,50",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=Path("output/pattern_weight_validation/a_lambda_summary.csv"),
    )
    parser.add_argument(
        "--out-details",
        type=Path,
        default=Path("output/pattern_weight_validation/a_lambda_details.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_max < 1:
        raise ValueError("--n-max must be positive")
    if args.exhaustive_n_max < 0:
        raise ValueError("--exhaustive-n-max must be non-negative")
    if args.top_partitions < 1:
        raise ValueError("--top-partitions must be positive")
    if args.top_count is not None and args.top_count < 0:
        raise ValueError("--top-count must be non-negative")
    if args.min_expected_count < 0.0:
        raise ValueError("--min-expected-count must be non-negative")
    if args.max_top_states < 1:
        raise ValueError("--max-top-states must be positive")

    summary_rows: list[SummaryRow] = []
    detail_rows: list[DetailRow] = []
    for alpha in parse_csv_floats(args.alphas):
        p = distribution_for_alpha(args.d, alpha)
        for n in range(1, args.n_max + 1):
            print(f"validating alpha={alpha:g}, N={n}", flush=True)
            summary, details = validate_one_n(
                p=p,
                alpha=alpha,
                n=n,
                exhaustive=n <= args.exhaustive_n_max,
                top_partitions=args.top_partitions,
                chunk_size=args.chunk_size,
                method=args.method,
                top_count=args.top_count,
                min_expected_count=args.min_expected_count,
                max_top_states=args.max_top_states,
            )
            summary_rows.append(summary)
            detail_rows.extend(details)

    write_summary(args.out_summary, summary_rows)
    write_details(args.out_details, detail_rows)
    print(f"wrote {args.out_summary}")
    print(f"wrote {args.out_details}")
    print_summary(
        summary_rows,
        n_values={int(n) for n in parse_csv_floats(args.summary_n_values)},
    )


if __name__ == "__main__":
    main()
