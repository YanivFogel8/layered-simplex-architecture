from __future__ import annotations

import argparse
import csv
import math
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas

from lsa import (
    build_selected_product_moment_tables,
    entropy,
    log_q_lambda_closed_l1,
    log_q_lambda_laplace,
)


EULER_GAMMA = 0.5772156649015329
C_STAR = 1.0 / (1.0 - EULER_GAMMA)
LOG_2 = math.log(2.0)


@dataclass(frozen=True)
class DepthSpec:
    label: str
    kind: Literal["fixed", "log"]
    value: float

    def L_for_d(self, d: int) -> int:
        if self.kind == "fixed":
            return max(1, int(round(self.value)))
        return max(1, int(round(self.value * math.log(d))))

    def c_for_d(self, d: int) -> float:
        return self.L_for_d(d) / math.log(d)

    def plot_label(self) -> str:
        if self.kind == "fixed":
            return f"L={int(round(self.value))}"
        if self.label == "log_cstar":
            return "L=round(c* ln d)"
        return f"L=round({self.value:.3g} ln d)"


@dataclass(frozen=True)
class ExperimentResult:
    d: int
    alpha: float
    eta: float
    N: int
    depth_label: str
    depth_kind: str
    depth_value: float
    L: int
    L_over_ln_d: float
    profile_samples: int
    unique_profiles: int
    distinct_r_values: int
    max_profile_part: int
    entropy_bits: float
    mean_log_q_nats: float
    model_penalty_bits: float
    regret_bits: float
    regret_over_ln_d: float
    stderr_bits: float
    stderr_over_ln_d: float
    nonconverged_profile_samples: int
    failed_profile_samples: int
    min_left_gap: float
    min_right_gap: float
    u_min: float
    u_max: float
    elapsed_seconds: float


def zipf_distribution(d: int, alpha: float) -> np.ndarray:
    if d <= 0:
        raise ValueError("d must be positive")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    ranks = np.arange(1, d + 1, dtype=np.float64)
    weights = ranks ** (-alpha)
    return weights / weights.sum(dtype=np.float64)


def sample_target_profiles(
    *,
    p: np.ndarray,
    N: int,
    profile_samples: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Counter[tuple[int, ...]]:
    profiles: Counter[tuple[int, ...]] = Counter()
    remaining = profile_samples
    while remaining:
        current = min(batch_size, remaining)
        draws = rng.choice(p.size, size=(current, N), replace=True, p=p)
        for row in draws:
            _, counts = np.unique(row, return_counts=True)
            profile = tuple(sorted((int(count) for count in counts), reverse=True))
            profiles[profile] += 1
        remaining -= current
    return profiles


def needed_r_values(profiles: Counter[tuple[int, ...]]) -> tuple[int, ...]:
    r_values = {0, 1, 2}
    for partition in profiles:
        for part in partition:
            r_values.add(part)
            r_values.add(part + 1)
            r_values.add(part + 2)
    return tuple(sorted(r_values))


def run_one_group(
    *,
    d: int,
    alpha: float,
    eta: float,
    depth_specs: tuple[DepthSpec, ...],
    profile_samples: int,
    profile_batch_size: int,
    seed: int,
    u_min: float | None,
    u_max: float | None,
    u_points: int,
    laguerre_order: int,
    chunk_size: int,
) -> list[ExperimentResult]:
    start = time.perf_counter()
    N = max(1, int(round(d**eta)))
    rng = np.random.default_rng(seed)
    p = zipf_distribution(d, alpha)
    entropy_bits = entropy(p, validate=False)
    profiles = sample_target_profiles(
        p=p,
        N=N,
        profile_samples=profile_samples,
        batch_size=profile_batch_size,
        rng=rng,
    )
    r_values = needed_r_values(profiles)
    max_profile_part = max(max(profile) for profile in profiles)
    l_by_label = {spec.label: spec.L_for_d(d) for spec in depth_specs}
    unique_l_values = sorted(set(l_by_label.values()))
    max_L = max(unique_l_values)
    actual_u_min = (
        u_min
        if u_min is not None
        else min(-80.0, -max_L * math.log(max(r_values) + 1.0) - 30.0)
    )
    actual_u_max = (
        u_max
        if u_max is not None
        else max(45.0, EULER_GAMMA * max_L - 0.5 * math.log(d) + 20.0)
    )

    print(
        f"d={d} alpha={alpha:g} eta={eta:g} N={N} "
        f"samples={profile_samples} unique_profiles={len(profiles)} "
        f"r_values={len(r_values)} max_part={max_profile_part} "
        f"L_values={unique_l_values} u=[{actual_u_min:.4g},{actual_u_max:.4g}]",
        flush=True,
    )

    tables = None
    if max_L > 1:
        table_start = time.perf_counter()
        tables = build_selected_product_moment_tables(
            max_L=max_L,
            r_values=r_values,
            u_min=actual_u_min,
            u_max=actual_u_max,
            u_points=u_points,
            laguerre_order=laguerre_order,
            chunk_size=chunk_size,
        )
        print(
            f"  built moment tables in {time.perf_counter() - table_start:.2f}s",
            flush=True,
        )

    values_by_l = {}
    for L in unique_l_values:
        l_start = time.perf_counter()
        values = {}
        for partition in profiles:
            if L == 1:
                q_result = log_q_lambda_closed_l1(d=d, partition=partition)
            else:
                if tables is None:
                    raise RuntimeError("moment tables were not built")
                q_result = log_q_lambda_laplace(
                    d=d,
                    L=L,
                    partition=partition,
                    tables=tables,
                )
            values[partition] = (
                q_result.log_q,
                q_result.converged,
                math.isfinite(q_result.log_q),
                q_result.left_gap if q_result.left_gap is not None else math.inf,
                q_result.right_gap if q_result.right_gap is not None else math.inf,
            )
        values_by_l[L] = values
        print(
            f"  computed q_lambda for L={L} over {len(profiles)} profiles "
            f"in {time.perf_counter() - l_start:.2f}s",
            flush=True,
        )

    results = []
    for spec in depth_specs:
        L = l_by_label[spec.label]
        result = summarize_depth(
            d=d,
            alpha=alpha,
            eta=eta,
            N=N,
            spec=spec,
            L=L,
            profile_samples=profile_samples,
            profiles=profiles,
            q_values=values_by_l[L],
            entropy_bits=entropy_bits,
            distinct_r_values=len(r_values),
            max_profile_part=max_profile_part,
            u_min=actual_u_min,
            u_max=actual_u_max,
            elapsed_seconds=time.perf_counter() - start,
        )
        results.append(result)
        print(
            f"  {spec.label}: L={L} regret={result.regret_bits:.6g} "
            f"+/- {result.stderr_bits:.2g} bits "
            f"failed={result.failed_profile_samples} "
            f"nonconv={result.nonconverged_profile_samples}",
            flush=True,
        )
    return results


def summarize_depth(
    *,
    d: int,
    alpha: float,
    eta: float,
    N: int,
    spec: DepthSpec,
    L: int,
    profile_samples: int,
    profiles: Counter[tuple[int, ...]],
    q_values: dict[tuple[int, ...], tuple[float, bool, bool, float, float]],
    entropy_bits: float,
    distinct_r_values: int,
    max_profile_part: int,
    u_min: float,
    u_max: float,
    elapsed_seconds: float,
) -> ExperimentResult:
    weighted_sum = 0.0
    finite_count = 0
    nonconverged = 0
    failed = 0
    min_left_gap = math.inf
    min_right_gap = math.inf

    for partition, count in profiles.items():
        log_q, converged, finite, left_gap, right_gap = q_values[partition]
        if not finite:
            failed += count
            continue
        if not converged:
            nonconverged += count
        weighted_sum += count * log_q
        finite_count += count
        min_left_gap = min(min_left_gap, left_gap)
        min_right_gap = min(min_right_gap, right_gap)

    if finite_count != profile_samples:
        return ExperimentResult(
            d=d,
            alpha=alpha,
            eta=eta,
            N=N,
            depth_label=spec.label,
            depth_kind=spec.kind,
            depth_value=spec.value,
            L=L,
            L_over_ln_d=L / math.log(d),
            profile_samples=profile_samples,
            unique_profiles=len(profiles),
            distinct_r_values=distinct_r_values,
            max_profile_part=max_profile_part,
            entropy_bits=entropy_bits,
            mean_log_q_nats=math.nan,
            model_penalty_bits=math.nan,
            regret_bits=math.nan,
            regret_over_ln_d=math.nan,
            stderr_bits=math.nan,
            stderr_over_ln_d=math.nan,
            nonconverged_profile_samples=nonconverged,
            failed_profile_samples=failed,
            min_left_gap=min_left_gap,
            min_right_gap=min_right_gap,
            u_min=u_min,
            u_max=u_max,
            elapsed_seconds=elapsed_seconds,
        )

    mean_log_q = weighted_sum / profile_samples
    weighted_squares = 0.0
    for partition, count in profiles.items():
        log_q = q_values[partition][0]
        weighted_squares += count * (log_q - mean_log_q) ** 2
    sample_variance = (
        weighted_squares / (profile_samples - 1) if profile_samples > 1 else 0.0
    )
    stderr_bits = math.sqrt(sample_variance / profile_samples) / (N * LOG_2)
    model_penalty_bits = -mean_log_q / (N * LOG_2)
    regret_bits = model_penalty_bits - entropy_bits
    return ExperimentResult(
        d=d,
        alpha=alpha,
        eta=eta,
        N=N,
        depth_label=spec.label,
        depth_kind=spec.kind,
        depth_value=spec.value,
        L=L,
        L_over_ln_d=L / math.log(d),
        profile_samples=profile_samples,
        unique_profiles=len(profiles),
        distinct_r_values=distinct_r_values,
        max_profile_part=max_profile_part,
        entropy_bits=entropy_bits,
        mean_log_q_nats=mean_log_q,
        model_penalty_bits=model_penalty_bits,
        regret_bits=regret_bits,
        regret_over_ln_d=regret_bits / math.log(d),
        stderr_bits=stderr_bits,
        stderr_over_ln_d=stderr_bits / math.log(d),
        nonconverged_profile_samples=nonconverged,
        failed_profile_samples=failed,
        min_left_gap=min_left_gap,
        min_right_gap=min_right_gap,
        u_min=u_min,
        u_max=u_max,
        elapsed_seconds=elapsed_seconds,
    )


def write_results_csv(path: Path, results: list[ExperimentResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "d",
        "alpha",
        "eta",
        "N",
        "depth_label",
        "depth_kind",
        "depth_value",
        "c_star",
        "L",
        "L_over_ln_d",
        "profile_samples",
        "unique_profiles",
        "distinct_r_values",
        "max_profile_part",
        "entropy_bits",
        "mean_log_q_nats",
        "model_penalty_bits",
        "regret_bits",
        "regret_over_ln_d",
        "stderr_bits",
        "stderr_over_ln_d",
        "nonconverged_profile_samples",
        "failed_profile_samples",
        "min_left_gap",
        "min_right_gap",
        "u_min",
        "u_max",
        "elapsed_seconds",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sort_results(results):
            writer.writerow(
                {
                    "d": row.d,
                    "alpha": f"{row.alpha:g}",
                    "eta": f"{row.eta:.12g}",
                    "N": row.N,
                    "depth_label": row.depth_label,
                    "depth_kind": row.depth_kind,
                    "depth_value": f"{row.depth_value:.12g}",
                    "c_star": f"{C_STAR:.12g}",
                    "L": row.L,
                    "L_over_ln_d": f"{row.L_over_ln_d:.12g}",
                    "profile_samples": row.profile_samples,
                    "unique_profiles": row.unique_profiles,
                    "distinct_r_values": row.distinct_r_values,
                    "max_profile_part": row.max_profile_part,
                    "entropy_bits": f"{row.entropy_bits:.12g}",
                    "mean_log_q_nats": f"{row.mean_log_q_nats:.12g}",
                    "model_penalty_bits": f"{row.model_penalty_bits:.12g}",
                    "regret_bits": f"{row.regret_bits:.12g}",
                    "regret_over_ln_d": f"{row.regret_over_ln_d:.12g}",
                    "stderr_bits": f"{row.stderr_bits:.12g}",
                    "stderr_over_ln_d": f"{row.stderr_over_ln_d:.12g}",
                    "nonconverged_profile_samples": row.nonconverged_profile_samples,
                    "failed_profile_samples": row.failed_profile_samples,
                    "min_left_gap": f"{row.min_left_gap:.12g}",
                    "min_right_gap": f"{row.min_right_gap:.12g}",
                    "u_min": f"{row.u_min:.12g}",
                    "u_max": f"{row.u_max:.12g}",
                    "elapsed_seconds": f"{row.elapsed_seconds:.12g}",
                }
            )


def read_results_csv(path: Path) -> list[ExperimentResult]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            ExperimentResult(
                d=int(row["d"]),
                alpha=float(row["alpha"]),
                eta=float(row["eta"]),
                N=int(row["N"]),
                depth_label=row["depth_label"],
                depth_kind=row["depth_kind"],
                depth_value=float(row["depth_value"]),
                L=int(row["L"]),
                L_over_ln_d=float(row["L_over_ln_d"]),
                profile_samples=int(row["profile_samples"]),
                unique_profiles=int(row["unique_profiles"]),
                distinct_r_values=int(row["distinct_r_values"]),
                max_profile_part=int(row["max_profile_part"]),
                entropy_bits=float(row["entropy_bits"]),
                mean_log_q_nats=float(row["mean_log_q_nats"]),
                model_penalty_bits=float(row["model_penalty_bits"]),
                regret_bits=float(row["regret_bits"]),
                regret_over_ln_d=float(row["regret_over_ln_d"]),
                stderr_bits=float(row["stderr_bits"]),
                stderr_over_ln_d=float(row["stderr_over_ln_d"]),
                nonconverged_profile_samples=int(row["nonconverged_profile_samples"]),
                failed_profile_samples=int(row["failed_profile_samples"]),
                min_left_gap=float(row["min_left_gap"]),
                min_right_gap=float(row["min_right_gap"]),
                u_min=float(row["u_min"]),
                u_max=float(row["u_max"]),
                elapsed_seconds=float(row["elapsed_seconds"]),
            )
            for row in reader
        ]


def write_summary_csv(path: Path, results: list[ExperimentResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "alpha",
        "eta",
        "depth_label",
        "depth_kind",
        "point_count",
        "d_min",
        "d_max",
        "fit_regret_amplitude",
        "fit_regret_beta",
        "fit_regret_r2",
        "fit_regret_over_ln_d_amplitude",
        "fit_regret_over_ln_d_beta",
        "fit_regret_over_ln_d_r2",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for alpha in sorted({row.alpha for row in results}):
            for eta in sorted({row.eta for row in results if row.alpha == alpha}):
                rows = [
                    row
                    for row in results
                    if row.alpha == alpha
                    and row.eta == eta
                    and row.failed_profile_samples == 0
                    and math.isfinite(row.regret_bits)
                    and row.regret_bits > 0.0
                ]
                for depth_label in depth_order([row.depth_label for row in rows]):
                    curve = sorted(
                        [row for row in rows if row.depth_label == depth_label],
                        key=lambda row: row.d,
                    )
                    if len(curve) < 2:
                        continue
                    fit_r = power_fit(
                        [float(row.d) for row in curve],
                        [row.regret_bits for row in curve],
                    )
                    fit_norm = power_fit(
                        [float(row.d) for row in curve],
                        [row.regret_over_ln_d for row in curve],
                    )
                    writer.writerow(
                        {
                            "alpha": f"{alpha:g}",
                            "eta": f"{eta:.12g}",
                            "depth_label": depth_label,
                            "depth_kind": curve[0].depth_kind,
                            "point_count": len(curve),
                            "d_min": curve[0].d,
                            "d_max": curve[-1].d,
                            "fit_regret_amplitude": f"{fit_r[0]:.12g}",
                            "fit_regret_beta": f"{fit_r[1]:.12g}",
                            "fit_regret_r2": f"{fit_r[2]:.12g}",
                            "fit_regret_over_ln_d_amplitude": f"{fit_norm[0]:.12g}",
                            "fit_regret_over_ln_d_beta": f"{fit_norm[1]:.12g}",
                            "fit_regret_over_ln_d_r2": f"{fit_norm[2]:.12g}",
                        }
                    )


def power_fit(x_values: list[float], y_values: list[float]) -> tuple[float, float, float]:
    log_x = np.log(np.asarray(x_values, dtype=np.float64))
    log_y = np.log(np.asarray(y_values, dtype=np.float64))
    slope, intercept = np.polyfit(log_x, log_y, deg=1)
    fitted = intercept + slope * log_x
    residual = float(np.sum((log_y - fitted) ** 2))
    total = float(np.sum((log_y - float(np.mean(log_y))) ** 2))
    r2 = 1.0 - residual / total if total > 0.0 else 1.0
    return float(math.exp(intercept)), float(-slope), r2


def write_plots(
    pdf_path: Path,
    png_dir: Path,
    results: list[ExperimentResult],
    depth_specs: tuple[DepthSpec, ...],
) -> None:
    del png_dir
    finite = [
        row
        for row in results
        if row.failed_profile_samples == 0
        and math.isfinite(row.regret_bits)
        and row.regret_bits > 0.0
    ]
    if not finite:
        return
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    spec_by_label = {spec.label: spec for spec in depth_specs}
    labels = depth_order([row.depth_label for row in finite])
    styles = plot_styles(labels)
    pdf = canvas.Canvas(str(pdf_path), pagesize=landscape(letter))
    width, height = landscape(letter)
    pdf.setTitle("Small L versus logarithmic depth")

    for alpha in sorted({row.alpha for row in finite}):
        alpha_rows = [row for row in finite if row.alpha == alpha]
        for eta in sorted({row.eta for row in alpha_rows}):
            eta_rows = [row for row in alpha_rows if row.eta == eta]
            draw_scaling_page(
                pdf,
                width=width,
                height=height,
                rows=eta_rows,
                alpha=alpha,
                eta=eta,
                labels=labels,
                spec_by_label=spec_by_label,
                styles=styles,
            )
            pdf.showPage()

    draw_exponent_page(
        pdf,
        width=width,
        height=height,
        rows=finite,
        labels=labels,
        spec_by_label=spec_by_label,
        styles=styles,
    )
    pdf.save()


def draw_scaling_page(
    pdf: canvas.Canvas,
    *,
    width: float,
    height: float,
    rows: list[ExperimentResult],
    alpha: float,
    eta: float,
    labels: list[str],
    spec_by_label: dict[str, DepthSpec],
    styles: dict[str, tuple[colors.Color, tuple[int, ...] | None, str]],
) -> None:
    title = f"Small L versus logarithmic depth: alpha={alpha:g}, N=round(d^{eta:g})"
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(44, height - 42, title)
    sample_counts = sorted({row.profile_samples for row in rows})
    n_values = sorted({row.N for row in rows})
    pdf.setFont("Helvetica", 8.5)
    pdf.drawString(
        44,
        height - 58,
        f"profile samples={sample_counts}; N values={n_values}; c*={C_STAR:.6g}.",
    )
    pdf.drawString(
        44,
        height - 72,
        "Curves are distinguished by color, dash pattern, and marker. "
        "Error bars show profile-sampling standard error.",
    )
    panel_w = (width - 120) / 2
    panel_h = height - 165
    draw_log_panel(
        pdf,
        x0=58,
        y0=88,
        w=panel_w,
        h=panel_h,
        rows=rows,
        labels=labels,
        spec_by_label=spec_by_label,
        styles=styles,
        metric="regret_bits",
        stderr_metric="stderr_bits",
        y_label="regret R_N(Q,p) [bits]",
    )
    draw_log_panel(
        pdf,
        x0=82 + panel_w,
        y0=88,
        w=panel_w,
        h=panel_h,
        rows=rows,
        labels=labels,
        spec_by_label=spec_by_label,
        styles=styles,
        metric="regret_over_ln_d",
        stderr_metric="stderr_over_ln_d",
        y_label="regret / ln(d) [bits]",
        legend=True,
    )


def draw_log_panel(
    pdf: canvas.Canvas,
    *,
    x0: float,
    y0: float,
    w: float,
    h: float,
    rows: list[ExperimentResult],
    labels: list[str],
    spec_by_label: dict[str, DepthSpec],
    styles: dict[str, tuple[colors.Color, tuple[int, ...] | None, str]],
    metric: str,
    stderr_metric: str,
    y_label: str,
    legend: bool = False,
) -> None:
    finite = [
        row
        for row in rows
        if math.isfinite(getattr(row, metric)) and getattr(row, metric) > 0.0
    ]
    if not finite:
        return
    log_x_values = [math.log10(row.d) for row in finite]
    log_y_values = [math.log10(getattr(row, metric)) for row in finite]
    x_min, x_max = padded_log_range(log_x_values)
    y_min, y_max = padded_log_range(log_y_values)

    def sx(d_value: float) -> float:
        return x0 + (math.log10(d_value) - x_min) / (x_max - x_min) * w

    def sy(y_value: float) -> float:
        return y0 + (math.log10(y_value) - y_min) / (y_max - y_min) * h

    draw_log_axes(pdf, x0, y0, w, h, x_min, x_max, y_min, y_max)
    pdf.setFont("Helvetica", 8.5)
    pdf.drawCentredString(x0 + w / 2, y0 - 36, "alphabet size d")
    pdf.saveState()
    pdf.translate(x0 - 44, y0 + h / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, y_label)
    pdf.restoreState()

    for label in labels:
        curve = sorted([row for row in finite if row.depth_label == label], key=lambda r: r.d)
        if not curve:
            continue
        color, dash, marker = styles[label]
        pdf.saveState()
        pdf.setStrokeColor(color)
        pdf.setFillColor(color)
        pdf.setLineWidth(1.5)
        if dash is not None:
            pdf.setDash(list(dash))
        previous = None
        for row in curve:
            point = (sx(row.d), sy(getattr(row, metric)))
            if previous is not None:
                pdf.line(previous[0], previous[1], point[0], point[1])
            previous = point
        pdf.setDash()
        for row in curve:
            x = sx(row.d)
            y = sy(getattr(row, metric))
            err = getattr(row, stderr_metric)
            if math.isfinite(err) and err > 0.0 and getattr(row, metric) > err:
                y_lo = sy(getattr(row, metric) - err)
                y_hi = sy(getattr(row, metric) + err)
                pdf.line(x, y_lo, x, y_hi)
                pdf.line(x - 2.5, y_lo, x + 2.5, y_lo)
                pdf.line(x - 2.5, y_hi, x + 2.5, y_hi)
            draw_marker(pdf, x, y, marker, 3.5)
        pdf.restoreState()

    if legend:
        legend_x = x0 + w - 150
        legend_y = y0 + h - 16
        pdf.setFont("Helvetica", 7.8)
        for index, label in enumerate(labels):
            curve = sorted(
                [row for row in finite if row.depth_label == label],
                key=lambda r: r.d,
            )
            if not curve:
                continue
            color, dash, marker = styles[label]
            y = legend_y - 16 * index
            pdf.saveState()
            pdf.setStrokeColor(color)
            pdf.setFillColor(color)
            pdf.setLineWidth(1.4)
            if dash is not None:
                pdf.setDash(list(dash))
            pdf.line(legend_x, y, legend_x + 18, y)
            pdf.setDash()
            draw_marker(pdf, legend_x + 9, y, marker, 3.3)
            pdf.restoreState()
            legend_label = spec_by_label[label].plot_label()
            if len(curve) >= 2:
                _, beta, r2 = power_fit(
                    [float(row.d) for row in curve],
                    [getattr(row, metric) for row in curve],
                )
                legend_label += f"; beta={beta:.3g}"
                if r2 < 0.95:
                    legend_label += f", R2={r2:.2f}"
            pdf.setFillColor(colors.black)
            pdf.drawString(legend_x + 24, y - 3, legend_label)


def draw_exponent_page(
    pdf: canvas.Canvas,
    *,
    width: float,
    height: float,
    rows: list[ExperimentResult],
    labels: list[str],
    spec_by_label: dict[str, DepthSpec],
    styles: dict[str, tuple[colors.Color, tuple[int, ...] | None, str]],
) -> None:
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(44, height - 44, "Estimated alphabet-scaling exponents")
    pdf.setFont("Helvetica", 9)
    pdf.drawString(
        44,
        height - 62,
        "Fits use R/ln(d) ~ A d^{-beta}; positive beta means normalized regret decreases with d.",
    )
    points_by_label: dict[str, list[tuple[float, float]]] = {}
    for label in labels:
        points = []
        for alpha in sorted({row.alpha for row in rows}):
            curve = sorted(
                [row for row in rows if row.alpha == alpha and row.depth_label == label],
                key=lambda row: row.d,
            )
            if len(curve) >= 2:
                _, beta, _ = power_fit(
                    [float(row.d) for row in curve],
                    [row.regret_over_ln_d for row in curve],
                )
                points.append((alpha, beta))
        points_by_label[label] = points
    all_points = [point for points in points_by_label.values() for point in points]
    if not all_points:
        return
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    x_min, x_max = padded_linear_range(x_values, floor=None)
    y_min, y_max = padded_linear_range(y_values, floor=0.0)
    x0, y0, w, h = 86, 96, width - 180, height - 190

    def sx(x: float) -> float:
        return x0 + (x - x_min) / (x_max - x_min) * w

    def sy(y: float) -> float:
        return y0 + (y - y_min) / (y_max - y_min) * h

    draw_linear_axes(pdf, x0, y0, w, h, x_min, x_max, y_min, y_max)
    zero_y = sy(0.0)
    if y0 <= zero_y <= y0 + h:
        pdf.saveState()
        pdf.setStrokeColor(colors.HexColor("#666666"))
        pdf.setLineWidth(0.8)
        pdf.setDash(3, 2)
        pdf.line(x0, zero_y, x0 + w, zero_y)
        pdf.restoreState()
    pdf.setFont("Helvetica", 9)
    pdf.drawCentredString(x0 + w / 2, y0 - 38, "Zipf exponent alpha")
    pdf.saveState()
    pdf.translate(x0 - 52, y0 + h / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, "beta in R/ln(d) ~ A d^{-beta}")
    pdf.restoreState()

    for label in labels:
        points = points_by_label[label]
        if not points:
            continue
        color, dash, marker = styles[label]
        pdf.saveState()
        pdf.setStrokeColor(color)
        pdf.setFillColor(color)
        pdf.setLineWidth(1.5)
        if dash is not None:
            pdf.setDash(list(dash))
        previous = None
        for x_value, y_value in points:
            point = (sx(x_value), sy(y_value))
            if previous is not None:
                pdf.line(previous[0], previous[1], point[0], point[1])
            previous = point
        pdf.setDash()
        for x_value, y_value in points:
            draw_marker(pdf, sx(x_value), sy(y_value), marker, 3.6)
        pdf.restoreState()

    legend_x = x0 + w - 175
    legend_y = y0 + h - 18
    pdf.setFont("Helvetica", 8.2)
    for index, label in enumerate(labels):
        if not points_by_label[label]:
            continue
        color, dash, marker = styles[label]
        y = legend_y - 17 * index
        pdf.saveState()
        pdf.setStrokeColor(color)
        pdf.setFillColor(color)
        if dash is not None:
            pdf.setDash(list(dash))
        pdf.line(legend_x, y, legend_x + 20, y)
        pdf.setDash()
        draw_marker(pdf, legend_x + 10, y, marker, 3.5)
        pdf.restoreState()
        pdf.setFillColor(colors.black)
        pdf.drawString(legend_x + 27, y - 3, spec_by_label[label].plot_label())


def draw_log_axes(
    pdf: canvas.Canvas,
    x0: float,
    y0: float,
    w: float,
    h: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    pdf.saveState()
    pdf.setLineWidth(0.8)
    pdf.setStrokeColor(colors.black)
    pdf.rect(x0, y0, w, h, stroke=1, fill=0)
    pdf.setFont("Helvetica", 7.4)
    for value in log_ticks(x_min, x_max):
        x = x0 + (value - x_min) / (x_max - x_min) * w
        pdf.setStrokeColor(colors.HexColor("#DDDDDD"))
        pdf.line(x, y0, x, y0 + h)
        pdf.setStrokeColor(colors.black)
        pdf.line(x, y0, x, y0 - 3)
        pdf.drawCentredString(x, y0 - 13, format_power10(value))
    for value in log_ticks(y_min, y_max):
        y = y0 + (value - y_min) / (y_max - y_min) * h
        pdf.setStrokeColor(colors.HexColor("#DDDDDD"))
        pdf.line(x0, y, x0 + w, y)
        pdf.setStrokeColor(colors.black)
        pdf.line(x0 - 3, y, x0, y)
        pdf.drawRightString(x0 - 6, y - 2.5, format_power10(value))
    pdf.restoreState()


def draw_linear_axes(
    pdf: canvas.Canvas,
    x0: float,
    y0: float,
    w: float,
    h: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    pdf.saveState()
    pdf.setStrokeColor(colors.black)
    pdf.setLineWidth(0.8)
    pdf.rect(x0, y0, w, h, stroke=1, fill=0)
    pdf.setFont("Helvetica", 7.5)
    for index in range(6):
        frac = index / 5
        x_value = x_min + frac * (x_max - x_min)
        y_value = y_min + frac * (y_max - y_min)
        x = x0 + frac * w
        y = y0 + frac * h
        pdf.setStrokeColor(colors.HexColor("#DDDDDD"))
        pdf.line(x, y0, x, y0 + h)
        pdf.line(x0, y, x0 + w, y)
        pdf.setStrokeColor(colors.black)
        pdf.drawCentredString(x, y0 - 13, f"{x_value:.3g}")
        pdf.drawRightString(x0 - 6, y - 2.5, f"{y_value:.3g}")
    pdf.restoreState()


def log_ticks(lower: float, upper: float) -> list[float]:
    start = math.ceil(lower)
    stop = math.floor(upper)
    ticks = [float(value) for value in range(start, stop + 1)]
    if len(ticks) < 3:
        ticks = [lower + index * (upper - lower) / 4 for index in range(5)]
    return ticks


def format_power10(log_value: float) -> str:
    rounded = round(log_value)
    if abs(log_value - rounded) < 1e-9:
        if -2 <= rounded <= 4:
            return f"{10.0**rounded:g}"
        return f"1e{rounded}"
    return f"{10.0**log_value:.2g}"


def padded_log_range(values: list[float]) -> tuple[float, float]:
    lower = min(values)
    upper = max(values)
    if lower == upper:
        lower -= 0.5
        upper += 0.5
    span = upper - lower
    return lower - 0.08 * span, upper + 0.10 * span


def padded_linear_range(values: list[float], *, floor: float | None) -> tuple[float, float]:
    lower = min(values)
    upper = max(values)
    if floor is not None:
        lower = min(lower, floor)
    if lower == upper:
        padding = max(0.1, abs(lower) * 0.1)
        lower -= padding
        upper += padding
    span = upper - lower
    return lower - 0.08 * span, upper + 0.12 * span


def plot_styles(
    labels: list[str],
) -> dict[str, tuple[colors.Color, tuple[int, ...] | None, str]]:
    palette = {
        "L1": colors.HexColor("#0072B2"),
        "L2": colors.HexColor("#D55E00"),
        "L5": colors.HexColor("#009E73"),
        "log_cstar": colors.HexColor("#CC79A7"),
    }
    dashes = {
        "L1": None,
        "L2": (6, 3),
        "L5": (2, 2),
        "log_cstar": (8, 2, 2, 2),
    }
    markers = {
        "L1": "circle",
        "L2": "square",
        "L5": "triangle",
        "log_cstar": "diamond",
    }
    return {
        label: (
            palette.get(label, colors.black),
            dashes.get(label),
            markers.get(label, "circle"),
        )
        for label in labels
    }


def draw_marker(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    marker: str,
    size: float,
) -> None:
    if marker == "circle":
        pdf.circle(x, y, size, stroke=1, fill=1)
    elif marker == "square":
        pdf.rect(x - size, y - size, 2 * size, 2 * size, stroke=1, fill=1)
    elif marker == "triangle":
        pdf.line(x, y + size, x - size, y - size)
        pdf.line(x - size, y - size, x + size, y - size)
        pdf.line(x + size, y - size, x, y + size)
    elif marker == "diamond":
        pdf.line(x, y + size, x - size, y)
        pdf.line(x - size, y, x, y - size)
        pdf.line(x, y - size, x + size, y)
        pdf.line(x + size, y, x, y + size)
    else:
        pdf.line(x - size, y, x + size, y)
        pdf.line(x, y - size, x, y + size)


def depth_order(labels: list[str]) -> list[str]:
    preferred = ["L1", "L2", "L5", "log_cstar"]
    present = set(labels)
    ordered = [label for label in preferred if label in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def sort_results(results: list[ExperimentResult]) -> list[ExperimentResult]:
    order = {label: index for index, label in enumerate(depth_order([r.depth_label for r in results]))}
    return sorted(
        results,
        key=lambda row: (
            row.eta,
            row.alpha,
            row.d,
            order.get(row.depth_label, 999),
            row.L,
        ),
    )


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_log_depths(raw: str) -> list[DepthSpec]:
    specs = []
    for token in [part.strip() for part in raw.split(",") if part.strip()]:
        if token.lower() in {"cstar", "c*", "star"}:
            specs.append(DepthSpec("log_cstar", "log", C_STAR))
        else:
            value = float(token)
            label = f"log_c{value:g}".replace(".", "p")
            specs.append(DepthSpec(label, "log", value))
    return specs


def make_depth_specs(fixed_l_values: list[int], log_depths: list[DepthSpec]) -> tuple[DepthSpec, ...]:
    fixed_specs = tuple(DepthSpec(f"L{value}", "fixed", float(value)) for value in fixed_l_values)
    return (*fixed_specs, *tuple(log_depths))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-csv", type=Path, default=None)
    parser.add_argument("--d-values", type=str, default="100,1000,10000,100000")
    parser.add_argument("--alphas", type=str, default="1,2,3")
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--fixed-l-values", type=str, default="1,2,5")
    parser.add_argument("--log-depths", type=str, default="cstar")
    parser.add_argument("--profile-samples", type=int, default=2000)
    parser.add_argument("--profile-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--u-min", type=float, default=None)
    parser.add_argument("--u-max", type=float, default=None)
    parser.add_argument("--u-points", type=int, default=12001)
    parser.add_argument("--laguerre-order", type=int, default=96)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--parallel-backend",
        choices=("thread", "process"),
        default="process",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("output/small_l_vs_log_depth/small_l_vs_log_depth.csv"),
    )
    parser.add_argument(
        "--out-summary-csv",
        type=Path,
        default=Path("output/small_l_vs_log_depth/small_l_vs_log_depth_summary.csv"),
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=Path("output/small_l_vs_log_depth/small_l_vs_log_depth.pdf"),
    )
    parser.add_argument(
        "--out-png-dir",
        type=Path,
        default=Path("output/small_l_vs_log_depth/png"),
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixed_l_values = parse_ints(args.fixed_l_values)
    depth_specs = make_depth_specs(fixed_l_values, parse_log_depths(args.log_depths))
    if args.from_csv is not None:
        results = read_results_csv(args.from_csv)
        write_summary_csv(args.out_summary_csv, results)
        if not args.no_plots:
            write_plots(args.out_pdf, args.out_png_dir, results, depth_specs)
        return

    d_values = parse_ints(args.d_values)
    alphas = parse_floats(args.alphas)
    if args.profile_samples <= 0:
        raise ValueError("--profile-samples must be positive")
    if args.profile_batch_size <= 0:
        raise ValueError("--profile-batch-size must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    jobs = []
    for d_index, d in enumerate(d_values):
        for alpha_index, alpha in enumerate(alphas):
            seed = args.seed + 1_000_003 * d_index + 10_007 * alpha_index
            jobs.append(
                {
                    "d": d,
                    "alpha": alpha,
                    "eta": args.eta,
                    "depth_specs": depth_specs,
                    "profile_samples": args.profile_samples,
                    "profile_batch_size": args.profile_batch_size,
                    "seed": seed,
                    "u_min": args.u_min,
                    "u_max": args.u_max,
                    "u_points": args.u_points,
                    "laguerre_order": args.laguerre_order,
                    "chunk_size": args.chunk_size,
                }
            )

    results: list[ExperimentResult] = []
    if args.workers == 1:
        for job in jobs:
            results.extend(run_one_group(**job))
            write_results_csv(args.out_csv, results)
            write_summary_csv(args.out_summary_csv, results)
            if not args.no_plots:
                write_plots(args.out_pdf, args.out_png_dir, results, depth_specs)
            print(f"wrote partial results to {args.out_csv}", flush=True)
    else:
        executor_class = ThreadPoolExecutor if args.parallel_backend == "thread" else ProcessPoolExecutor
        print(
            f"running {len(jobs)} groups with {args.workers} "
            f"{args.parallel_backend} workers",
            flush=True,
        )
        with executor_class(max_workers=args.workers) as executor:
            futures = [executor.submit(run_one_group, **job) for job in jobs]
            for future in as_completed(futures):
                results.extend(future.result())
                write_results_csv(args.out_csv, results)
                write_summary_csv(args.out_summary_csv, results)
                if not args.no_plots:
                    write_plots(args.out_pdf, args.out_png_dir, results, depth_specs)
                print(f"wrote partial results to {args.out_csv}", flush=True)

    results = sort_results(results)
    write_results_csv(args.out_csv, results)
    write_summary_csv(args.out_summary_csv, results)
    if not args.no_plots:
        write_plots(args.out_pdf, args.out_png_dir, results, depth_specs)
    print(f"wrote {args.out_csv}", flush=True)
    print(f"wrote {args.out_summary_csv}", flush=True)
    if not args.no_plots:
        print(f"wrote {args.out_pdf}", flush=True)


if __name__ == "__main__":
    main()
