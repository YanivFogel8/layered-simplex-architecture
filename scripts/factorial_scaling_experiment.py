"""Evaluate the scaling law on an independent Cartesian (d, N) grid.

Unlike the earlier alphabet-scaling path N=d**eta, this experiment varies
alphabet size and sample size independently.  Each raw cell is evaluated by
the established data-scaling kernel, so the resulting CSV supports clean
fixed-d data slices, fixed-N alphabet slices, and a joint normalized collapse.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from data_scaling_experiment import (
    CSpec,
    DataScalingResult,
    parse_c_specs,
    run_one_n_group,
    write_results_csv,
)
from small_l_vs_log_depth_experiment import parse_floats, parse_ints


def build_jobs(
    *,
    d_values: list[int],
    n_values: list[int],
    alphas: tuple[float, ...],
    c_spec: CSpec,
    profile_samples: int,
    profile_batch_size: int,
    seed: int,
    u_min: float | None,
    u_max: float | None,
    u_points: int,
    laguerre_order: int,
    chunk_size: int,
) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for d_index, d in enumerate(d_values):
        for n_index, n in enumerate(n_values):
            jobs.append(
                {
                    "d": d,
                    "alphas": alphas,
                    "c_spec": c_spec,
                    "N": n,
                    "profile_samples": profile_samples,
                    "profile_batch_size": profile_batch_size,
                    "seed": seed + 1_000_003 * d_index + 10_007 * n_index,
                    "u_min": u_min,
                    "u_max": u_max,
                    "u_points": u_points,
                    "laguerre_order": laguerre_order,
                    "chunk_size": chunk_size,
                }
            )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--d-values", type=str, default="1000,3162,10000,31623,100000"
    )
    parser.add_argument(
        "--N-values", type=str, default="50,100,200,500,1000,2000,5000"
    )
    parser.add_argument("--alphas", type=str, default="1,2,3,4")
    parser.add_argument("--c-value", type=str, default="cstar")
    parser.add_argument("--profile-samples", type=int, default=5000)
    parser.add_argument("--profile-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--u-min", type=float, default=None)
    parser.add_argument("--u-max", type=float, default=None)
    parser.add_argument("--u-points", type=int, default=8001)
    parser.add_argument("--laguerre-order", type=int, default=96)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("output/scaling_law_factorial/factorial_results.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d_values = parse_ints(args.d_values)
    n_values = parse_ints(args.N_values)
    alphas = tuple(parse_floats(args.alphas))
    c_specs = parse_c_specs(args.c_value)
    if len(c_specs) != 1:
        raise ValueError("--c-value must select exactly one depth coefficient")
    if any(d <= 1 for d in d_values):
        raise ValueError("--d-values must be greater than one")
    if any(n <= 0 for n in n_values):
        raise ValueError("--N-values must be positive")
    if any(alpha <= 0 for alpha in alphas):
        raise ValueError("--alphas must be positive")
    if args.profile_samples <= 0 or args.profile_batch_size <= 0:
        raise ValueError("profile sample and batch sizes must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    jobs = build_jobs(
        d_values=d_values,
        n_values=n_values,
        alphas=alphas,
        c_spec=c_specs[0],
        profile_samples=args.profile_samples,
        profile_batch_size=args.profile_batch_size,
        seed=args.seed,
        u_min=args.u_min,
        u_max=args.u_max,
        u_points=args.u_points,
        laguerre_order=args.laguerre_order,
        chunk_size=args.chunk_size,
    )
    print(
        f"running {len(jobs)} independent (d, N) cells with "
        f"{args.workers} workers",
        flush=True,
    )
    results: list[DataScalingResult] = []
    if args.workers == 1:
        for job in jobs:
            results.extend(run_one_n_group(**job))
            write_results_csv(args.out_csv, results)
            print(f"wrote partial results to {args.out_csv}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_one_n_group, **job) for job in jobs]
            for future in as_completed(futures):
                results.extend(future.result())
                write_results_csv(args.out_csv, results)
                print(f"wrote partial results to {args.out_csv}", flush=True)
    write_results_csv(args.out_csv, results)
    print(f"wrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
