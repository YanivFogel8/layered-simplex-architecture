from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.polynomial.laguerre import laggauss

from lsa.mixture_weights import (
    ProductMomentTables,
    build_product_moment_tables,
    compute_log_q_by_partition as compute_log_q_by_partition_package,
)


EULER_GAMMA = 0.5772156649015329
LOG_2 = math.log(2.0)


@dataclass(frozen=True)
class ExactRegretResult:
    d: int
    alpha: float
    L: int
    N: int
    regret_bits: float
    entropy_bits: float


def zipf_distribution(d: int, alpha: float) -> np.ndarray:
    if d <= 0:
        raise ValueError("d must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    ranks = np.arange(1, d + 1, dtype=np.float64)
    weights = ranks ** (-alpha)
    return weights / weights.sum(dtype=np.float64)


def default_l_values(d: int) -> list[int]:
    log_d = math.log(d)
    return [
        1,
        max(1, round(log_d)),
        max(1, round(log_d / (1.0 - EULER_GAMMA))),
        max(1, round(4.0 * log_d)),
    ]


def entropy_bits(p: np.ndarray) -> float:
    positive = p > 0
    return float(-np.sum(p[positive] * np.log2(p[positive]), dtype=np.float64))


def integer_partitions(n: int, *, max_part: int | None = None) -> list[tuple[int, ...]]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return [()]
    if max_part is None or max_part > n:
        max_part = n

    partitions = []
    for first in range(max_part, 0, -1):
        for rest in integer_partitions(n - first, max_part=min(first, n - first)):
            partitions.append((first, *rest))
    return partitions


def multinomial_pattern_weights(p: np.ndarray, N: int) -> dict[tuple[int, ...], float]:
    """Return P(sort(nonzero count vector)=lambda) for M~Multinomial(N,p)."""

    dp: dict[tuple[int, tuple[int, ...]], float] = {(0, ()): 1.0}
    inv_factorials = [1.0 / math.factorial(r) for r in range(N + 1)]
    for prob in p:
        next_dp: dict[tuple[int, tuple[int, ...]], float] = defaultdict(float)
        powers = [1.0]
        for r in range(1, N + 1):
            powers.append(powers[-1] * float(prob))

        for (total, parts), coeff in dp.items():
            remaining = N - total
            for r in range(remaining + 1):
                if r == 0:
                    next_parts = parts
                else:
                    next_parts = tuple(sorted((*parts, r), reverse=True))
                next_dp[(total + r, next_parts)] += coeff * powers[r] * inv_factorials[r]
        dp = dict(next_dp)

    factor = math.factorial(N)
    weights = {
        parts: factor * coeff
        for (total, parts), coeff in dp.items()
        if total == N
    }
    return weights


def multinomial_pattern_weights_power_sum(
    p: np.ndarray,
    N: int,
) -> dict[tuple[int, ...], float]:
    """Return multinomial count-pattern weights using symmetric sums.

    This computes the same values as :func:`multinomial_pattern_weights`, but
    avoids iterating over the alphabet for every partition state.  Let
    ``tilde_m_lambda`` be the augmented monomial symmetric sum, where equal
    parts of ``lambda`` are treated as labeled.  If ``lambda = (a, mu)``, then

        tilde_m_lambda = power_a * tilde_m_mu
                         - sum_b count_mu(b) * tilde_m_{mu-b+a+b}.

    The ordinary monomial symmetric sum is obtained by dividing by the
    factorials of repeated part multiplicities.
    """

    if N < 0:
        raise ValueError("N must be non-negative")
    if N == 0:
        return {(): 1.0}

    power_sums = np.empty(N + 1, dtype=np.float64)
    power_sums[0] = float(p.size)
    current_power = np.ones_like(p, dtype=np.float64)
    for r in range(1, N + 1):
        current_power = current_power * p
        power_sums[r] = float(np.sum(current_power, dtype=np.float64))

    augmented: dict[tuple[int, ...], float] = {(): 1.0}
    partitions_by_n = {n: integer_partitions(n) for n in range(1, N + 1)}
    for n in range(1, N + 1):
        for partition in sorted(partitions_by_n[n], key=len):
            a = partition[-1]
            mu = partition[:-1]
            value = power_sums[a] * augmented[mu]

            counts = Counter(mu)
            for b, multiplicity in counts.items():
                mu_without_b = list(mu)
                mu_without_b.remove(b)
                merged = tuple(sorted((*mu_without_b, a + b), reverse=True))
                value -= multiplicity * augmented[merged]

            if value < 0.0 and abs(value) < 1e-13:
                value = 0.0
            augmented[partition] = value

    factor = math.factorial(N)
    weights = {}
    for partition in partitions_by_n[N]:
        denominator = 1
        for part in partition:
            denominator *= math.factorial(part)
        for multiplicity in Counter(partition).values():
            denominator *= math.factorial(multiplicity)
        weights[partition] = factor * augmented[partition] / denominator
    return weights


def multinomial_pattern_weights_power_sum_mp(
    p: np.ndarray,
    N: int,
    *,
    dps: int = 90,
) -> dict[tuple[int, ...], float]:
    """High-precision version of ``multinomial_pattern_weights_power_sum``."""

    try:
        import mpmath as mp
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "mpmath is required for high-precision pattern weights"
        ) from exc

    if N < 0:
        raise ValueError("N must be non-negative")
    if N == 0:
        return {(): 1.0}

    old_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        power_sums = [mp.mpf("0")] * (N + 1)
        power_sums[0] = mp.mpf(int(p.size))
        for prob in p:
            prob_mp = mp.mpf(str(float(prob)))
            power = mp.mpf("1")
            for r in range(1, N + 1):
                power *= prob_mp
                power_sums[r] += power

        augmented: dict[tuple[int, ...], mp.mpf] = {(): mp.mpf("1")}
        partitions_by_n = {n: integer_partitions(n) for n in range(1, N + 1)}
        for n in range(1, N + 1):
            for partition in sorted(partitions_by_n[n], key=len):
                a = partition[-1]
                mu = partition[:-1]
                value = power_sums[a] * augmented[mu]

                counts = Counter(mu)
                for b, multiplicity in counts.items():
                    mu_without_b = list(mu)
                    mu_without_b.remove(b)
                    merged = tuple(sorted((*mu_without_b, a + b), reverse=True))
                    value -= multiplicity * augmented[merged]

                augmented[partition] = value

        factor = mp.mpf(math.factorial(N))
        weights = {}
        for partition in partitions_by_n[N]:
            denominator = 1
            for part in partition:
                denominator *= math.factorial(part)
            for multiplicity in Counter(partition).values():
                denominator *= math.factorial(multiplicity)
            probability = float(factor * augmented[partition] / denominator)
            if probability < 0.0 and abs(probability) < 1e-12:
                probability = 0.0
            weights[partition] = probability
        return weights
    finally:
        mp.mp.dps = old_dps


def multinomial_pattern_weights_fast(
    p: np.ndarray,
    N: int,
    *,
    sum_tol: float = 1e-8,
) -> dict[tuple[int, ...], float]:
    """Return pattern weights, falling back to high precision when necessary."""

    weights = multinomial_pattern_weights_power_sum(p, N)
    weight_sum = sum(weights.values())
    if math.isclose(weight_sum, 1.0, rel_tol=sum_tol, abs_tol=sum_tol):
        return weights
    return multinomial_pattern_weights_power_sum_mp(p, N)


def logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    finite = np.isfinite(max_values)
    shifted_sum = np.sum(np.exp(values - max_values), axis=axis, keepdims=True)
    out = max_values + np.log(shifted_sum)
    out = np.where(finite, out, -np.inf)
    if axis is None:
        return np.asarray(out).reshape(())
    return np.squeeze(out, axis=axis)


def build_log_phi_tables(
    *,
    max_L: int,
    max_r: int,
    u_grid: np.ndarray,
    laguerre_order: int,
    chunk_size: int,
) -> dict[tuple[int, int], np.ndarray]:
    """Compute log phi_r^(L)(exp(u)) on a grid in u=log(t).

    The recurrence is
        phi_r^(ell)(t) = E[E^r phi_r^(ell-1)(t E)],
    with E~Exp(1), evaluated by Gauss-Laguerre quadrature.
    """

    nodes, weights = laggauss(laguerre_order)
    if np.any(nodes <= 0) or np.any(weights <= 0):
        raise ValueError("Laguerre quadrature returned non-positive nodes/weights")

    log_nodes = np.log(nodes)
    log_quadrature_weights = np.log(weights)
    tables: dict[tuple[int, int], np.ndarray] = {}

    for r in range(max_r + 1):
        left_log_moments = {1: math.lgamma(r + 1)}
        current = math.lgamma(r + 1) - (r + 1) * np.logaddexp(0.0, u_grid)
        tables[(1, r)] = current.copy()

        for ell in range(2, max_L + 1):
            next_values = np.empty_like(u_grid)
            quadrature_prefactor = log_quadrature_weights + r * log_nodes
            left_value = left_log_moments[ell - 1]

            for start in range(0, u_grid.size, chunk_size):
                stop = min(start + chunk_size, u_grid.size)
                interpolation_points = (
                    u_grid[start:stop, None] + log_nodes[None, :]
                )
                interpolated = np.interp(
                    interpolation_points.ravel(),
                    u_grid,
                    current,
                    left=left_value,
                    right=-np.inf,
                ).reshape(interpolation_points.shape)
                values = quadrature_prefactor[None, :] + interpolated
                next_values[start:stop] = logsumexp(values, axis=1)

            current = next_values
            left_log_moments[ell] = ell * math.lgamma(r + 1)
            tables[(ell, r)] = current.copy()

    return tables


def log_trapezoid_integral(log_values: np.ndarray, x_grid: np.ndarray) -> float:
    if log_values.size != x_grid.size:
        raise ValueError("log_values and x_grid must have the same length")
    if log_values.size < 2:
        raise ValueError("at least two grid points are required")

    max_value = float(np.max(log_values))
    if not math.isfinite(max_value):
        return -math.inf
    integral = np.trapezoid(np.exp(log_values - max_value), x_grid)
    return max_value + math.log(float(integral))


def log_q_for_partition_numeric(
    *,
    partition: tuple[int, ...],
    d: int,
    L: int,
    N: int,
    u_grid: np.ndarray,
    log_phi_tables: dict[tuple[int, int], np.ndarray],
) -> tuple[float, float, float]:
    """Return log(q_lambda) and endpoint gaps for a partition."""

    nonzero_count = len(partition)
    log_integrand = (
        N * u_grid
        - math.lgamma(N)
        + (d - nonzero_count) * log_phi_tables[(L, 0)]
    )
    for part in partition:
        log_integrand = log_integrand + log_phi_tables[(L, part)]

    log_q = log_trapezoid_integral(log_integrand, u_grid)
    peak = float(np.max(log_integrand))
    left_gap = peak - float(log_integrand[0])
    right_gap = peak - float(log_integrand[-1])
    return log_q, left_gap, right_gap


def log_q_for_partition_closed_l1(
    *,
    partition: tuple[int, ...],
    d: int,
    N: int,
) -> float:
    return (
        math.lgamma(d)
        - math.lgamma(d + N)
        + sum(math.lgamma(1 + part) for part in partition)
    )


def compute_log_q_by_partition(
    *,
    d: int,
    L: int,
    N: int,
    u_grid: np.ndarray,
    log_phi_tables: dict[tuple[int, int], np.ndarray] | None,
) -> dict[tuple[int, ...], float]:
    partitions = integer_partitions(N)
    if L == 1:
        return {
            partition: log_q_for_partition_closed_l1(
                partition=partition,
                d=d,
                N=N,
            )
            for partition in partitions
        }

    if log_phi_tables is None:
        raise ValueError("log_phi_tables are required for L > 1")

    log_q = {}
    for partition in partitions:
        value, left_gap, right_gap = log_q_for_partition_numeric(
            partition=partition,
            d=d,
            L=L,
            N=N,
            u_grid=u_grid,
            log_phi_tables=log_phi_tables,
        )
        if left_gap < 20.0 or right_gap < 20.0:
            print(
                "warning: integral endpoint may be too close to peak "
                f"(d={d}, L={L}, N={N}, partition={partition}, "
                f"left_gap={left_gap:.3g}, right_gap={right_gap:.3g})",
                file=sys.stderr,
            )
        log_q[partition] = value
    return log_q


def exact_online_regret(
    *,
    p: np.ndarray,
    d: int,
    L: int,
    N: int,
    log_q_by_partition: dict[tuple[int, ...], float],
) -> float:
    pattern_weights = multinomial_pattern_weights(p, N)
    missing = set(pattern_weights) - set(log_q_by_partition)
    if missing:
        raise ValueError(f"missing q-values for partitions: {sorted(missing)}")

    expected_log_q = sum(
        weight * log_q_by_partition[partition]
        for partition, weight in pattern_weights.items()
    )
    pattern_weight_sum = sum(pattern_weights.values())
    if not math.isclose(pattern_weight_sum, 1.0, rel_tol=1e-11, abs_tol=1e-11):
        raise ValueError(f"pattern weights sum to {pattern_weight_sum}, not 1")

    target_log_prob_per_symbol = float(np.sum(p * np.log(p), dtype=np.float64))
    return (target_log_prob_per_symbol - expected_log_q / N) / LOG_2


def exact_online_regret_from_pattern_weights(
    *,
    p: np.ndarray,
    N: int,
    pattern_weights: dict[tuple[int, ...], float],
    log_q_by_partition: dict[tuple[int, ...], float],
) -> float:
    missing = set(pattern_weights) - set(log_q_by_partition)
    if missing:
        raise ValueError(f"missing q-values for partitions: {sorted(missing)}")

    expected_log_q = sum(
        weight * log_q_by_partition[partition]
        for partition, weight in pattern_weights.items()
    )
    pattern_weight_sum = sum(pattern_weights.values())
    if not math.isclose(pattern_weight_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"pattern weights sum to {pattern_weight_sum}, not 1")

    target_log_prob_per_symbol = float(np.sum(p * np.log(p), dtype=np.float64))
    return (target_log_prob_per_symbol - expected_log_q / N) / LOG_2


def parse_csv_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def write_results(path: Path, results: list[ExactRegretResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["d", "alpha", "L", "N", "regret_bits", "entropy_bits"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "d": result.d,
                    "alpha": f"{result.alpha:g}",
                    "L": result.L,
                    "N": result.N,
                    "regret_bits": f"{result.regret_bits:.12g}",
                    "entropy_bits": f"{result.entropy_bits:.12g}",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--alphas", type=str, default="1,2,3")
    parser.add_argument("--n-values", type=str, default="1,5")
    parser.add_argument(
        "--l-values",
        type=str,
        default="default",
        help="comma-separated L values, or 'default' for 1, round(ln d), "
        "round(ln d/(1-gamma)), round(4 ln d)",
    )
    parser.add_argument("--u-min", type=float, default=-70.0)
    parser.add_argument("--u-max", type=float, default=35.0)
    parser.add_argument("--u-points", type=int, default=16_001)
    parser.add_argument("--laguerre-order", type=int, default=96)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument(
        "--q-method",
        choices=("auto", "grid", "laplace"),
        default="auto",
        help="method for q_lambda: auto uses the closed L=1 formula and "
        "Laplace otherwise; grid is the older validation integral",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("output/exact_online_regret/exact_online_regret.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alphas = parse_csv_floats(args.alphas)
    n_values = parse_csv_ints(args.n_values)
    l_values = (
        default_l_values(args.d)
        if args.l_values == "default"
        else parse_csv_ints(args.l_values)
    )
    l_values = sorted(dict.fromkeys(l_values))
    n_values = sorted(dict.fromkeys(n_values))

    if args.u_points < 2:
        raise ValueError("--u-points must be at least 2")
    u_grid = np.linspace(args.u_min, args.u_max, args.u_points, dtype=np.float64)
    max_L = max(l_values)
    max_N = max(n_values)

    moment_tables: ProductMomentTables | None = None
    if max_L > 1 and max_N > 1:
        moment_tables = build_product_moment_tables(
            max_L=max_L,
            max_r=max_N + (2 if args.q_method in {"auto", "laplace"} else 0),
            u_grid=u_grid,
            laguerre_order=args.laguerre_order,
            chunk_size=args.chunk_size,
        )

    log_q_cache: dict[tuple[int, int], dict[tuple[int, ...], float]] = {}
    for N in n_values:
        for L in l_values:
            if N == 1:
                log_q_cache[(N, L)] = {(1,): -math.log(args.d)}
            else:
                log_q_cache[(N, L)] = compute_log_q_by_partition_package(
                    d=args.d,
                    L=L,
                    N=N,
                    method=args.q_method,
                    tables=moment_tables,
                    laguerre_order=args.laguerre_order,
                    chunk_size=args.chunk_size,
                )

    results = []
    for alpha in alphas:
        p = zipf_distribution(args.d, alpha)
        h_bits = entropy_bits(p)
        for N in n_values:
            pattern_weights = multinomial_pattern_weights_fast(p, N)
            for L in l_values:
                regret = exact_online_regret_from_pattern_weights(
                    p=p,
                    N=N,
                    pattern_weights=pattern_weights,
                    log_q_by_partition=log_q_cache[(N, L)],
                )
                results.append(
                    ExactRegretResult(
                        d=args.d,
                        alpha=alpha,
                        L=L,
                        N=N,
                        regret_bits=regret,
                        entropy_bits=h_bits,
                    )
                )

    write_results(args.out_csv, results)

    print(f"wrote {args.out_csv}")
    for alpha in alphas:
        print(f"alpha={alpha:g}")
        for N in n_values:
            print(f"  N={N}")
            for result in results:
                if result.alpha == alpha and result.N == N:
                    print(
                        f"    L={result.L:2d} "
                        f"regret_bits={result.regret_bits:.10g}"
                    )


if __name__ == "__main__":
    main()
