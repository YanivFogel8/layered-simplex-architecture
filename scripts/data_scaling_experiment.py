from __future__ import annotations

import argparse
import csv
import math
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

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
from small_l_vs_log_depth_experiment import (
    C_STAR,
    EULER_GAMMA,
    LOG_2,
    draw_log_axes,
    draw_marker,
    needed_r_values,
    padded_log_range,
    parse_floats,
    parse_ints,
    power_fit,
    sample_target_profiles,
    zipf_distribution,
)


@dataclass(frozen=True)
class CSpec:
    label: str
    value: float

    def L_for_d(self, d: int) -> int:
        return max(1, int(round(self.value * math.log(d))))

    def plot_label(self) -> str:
        if self.label == "cstar":
            return "c = c-star"
        return f"c = {self.value:g}"

    def tex_label(self) -> str:
        if self.label == "cstar":
            return r"$c^\star$"
        return f"${self.value:g}$"


@dataclass(frozen=True)
class DataScalingResult:
    d: int
    alpha: float
    c_label: str
    c_value: float
    L: int
    N: int
    profile_samples: int
    unique_profiles: int
    union_unique_profiles: int
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


@dataclass(frozen=True)
class FitResult:
    d: int
    alpha: float
    c_label: str
    c_value: float
    L: int
    fit_min_N: int
    point_count: int
    N_min: int
    N_max: int
    amplitude: float
    fitted_beta: float
    r2: float
    predicted_beta: float


def run_one_n_group(
    *,
    d: int,
    alphas: tuple[float, ...],
    c_spec: CSpec,
    N: int,
    profile_samples: int,
    profile_batch_size: int,
    seed: int,
    u_min: float | None,
    u_max: float | None,
    u_points: int,
    laguerre_order: int,
    chunk_size: int,
) -> list[DataScalingResult]:
    start = time.perf_counter()
    L = c_spec.L_for_d(d)
    profiles_by_alpha: dict[float, Counter[tuple[int, ...]]] = {}
    entropy_by_alpha: dict[float, float] = {}
    union_profiles: Counter[tuple[int, ...]] = Counter()

    for alpha_index, alpha in enumerate(alphas):
        profile_start = time.perf_counter()
        p = zipf_distribution(d, alpha)
        entropy_by_alpha[alpha] = entropy(p, validate=False)
        rng_seed = seed + 10_007 * alpha_index
        rng = np.random.default_rng(rng_seed)
        profiles = sample_target_profiles(
            p=p,
            N=N,
            profile_samples=profile_samples,
            batch_size=profile_batch_size,
            rng=rng,
        )
        profiles_by_alpha[alpha] = profiles
        union_profiles.update(profiles)
        print(
            f"d={d} {c_spec.label} L={L} N={N} alpha={alpha:g}: "
            f"sampled {profile_samples} profiles "
            f"({len(profiles)} unique) in {time.perf_counter() - profile_start:.2f}s",
            flush=True,
        )

    r_values = needed_r_values(union_profiles)
    max_profile_part = max(max(profile) for profile in union_profiles)
    actual_u_min = (
        u_min
        if u_min is not None
        else min(-80.0, -L * math.log(max(r_values) + 1.0) - 30.0)
    )
    actual_u_max = (
        u_max
        if u_max is not None
        else max(45.0, EULER_GAMMA * L - 0.5 * math.log(d) + 20.0)
    )
    print(
        f"d={d} {c_spec.label} L={L} N={N}: "
        f"union_profiles={len(union_profiles)} r_values={len(r_values)} "
        f"max_part={max_profile_part} u=[{actual_u_min:.4g},{actual_u_max:.4g}]",
        flush=True,
    )

    tables = None
    if L > 1:
        table_start = time.perf_counter()
        tables = build_selected_product_moment_tables(
            max_L=L,
            r_values=r_values,
            u_min=actual_u_min,
            u_max=actual_u_max,
            u_points=u_points,
            laguerre_order=laguerre_order,
            chunk_size=chunk_size,
        )
        print(
            f"d={d} {c_spec.label} L={L} N={N}: "
            f"built moment tables in {time.perf_counter() - table_start:.2f}s",
            flush=True,
        )

    q_values: dict[tuple[int, ...], tuple[float, bool, bool, float, float]] = {}
    q_start = time.perf_counter()
    for partition in union_profiles:
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
        q_values[partition] = (
            q_result.log_q,
            q_result.converged,
            math.isfinite(q_result.log_q),
            q_result.left_gap if q_result.left_gap is not None else math.inf,
            q_result.right_gap if q_result.right_gap is not None else math.inf,
        )
    print(
        f"d={d} {c_spec.label} L={L} N={N}: "
        f"computed q_lambda for {len(q_values)} profiles "
        f"in {time.perf_counter() - q_start:.2f}s",
        flush=True,
    )

    results = []
    for alpha in alphas:
        results.append(
            summarize_alpha(
                d=d,
                alpha=alpha,
                c_spec=c_spec,
                L=L,
                N=N,
                profile_samples=profile_samples,
                profiles=profiles_by_alpha[alpha],
                union_unique_profiles=len(union_profiles),
                q_values=q_values,
                entropy_bits=entropy_by_alpha[alpha],
                distinct_r_values=len(r_values),
                max_profile_part=max_profile_part,
                u_min=actual_u_min,
                u_max=actual_u_max,
                elapsed_seconds=time.perf_counter() - start,
            )
        )
    return results


def summarize_alpha(
    *,
    d: int,
    alpha: float,
    c_spec: CSpec,
    L: int,
    N: int,
    profile_samples: int,
    profiles: Counter[tuple[int, ...]],
    union_unique_profiles: int,
    q_values: dict[tuple[int, ...], tuple[float, bool, bool, float, float]],
    entropy_bits: float,
    distinct_r_values: int,
    max_profile_part: int,
    u_min: float,
    u_max: float,
    elapsed_seconds: float,
) -> DataScalingResult:
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
        return DataScalingResult(
            d=d,
            alpha=alpha,
            c_label=c_spec.label,
            c_value=c_spec.value,
            L=L,
            N=N,
            profile_samples=profile_samples,
            unique_profiles=len(profiles),
            union_unique_profiles=union_unique_profiles,
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
    return DataScalingResult(
        d=d,
        alpha=alpha,
        c_label=c_spec.label,
        c_value=c_spec.value,
        L=L,
        N=N,
        profile_samples=profile_samples,
        unique_profiles=len(profiles),
        union_unique_profiles=union_unique_profiles,
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


def write_results_csv(path: Path, results: list[DataScalingResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "d",
        "alpha",
        "c_label",
        "c_value",
        "L",
        "N",
        "profile_samples",
        "unique_profiles",
        "union_unique_profiles",
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
                    "c_label": row.c_label,
                    "c_value": f"{row.c_value:.12g}",
                    "L": row.L,
                    "N": row.N,
                    "profile_samples": row.profile_samples,
                    "unique_profiles": row.unique_profiles,
                    "union_unique_profiles": row.union_unique_profiles,
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


def read_results_csv(path: Path) -> list[DataScalingResult]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            DataScalingResult(
                d=int(row["d"]),
                alpha=float(row["alpha"]),
                c_label=row["c_label"],
                c_value=float(row["c_value"]),
                L=int(row["L"]),
                N=int(row["N"]),
                profile_samples=int(row["profile_samples"]),
                unique_profiles=int(row["unique_profiles"]),
                union_unique_profiles=int(row["union_unique_profiles"]),
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


def write_summary_csv(
    path: Path,
    results: list[DataScalingResult],
    *,
    fit_min_N: int,
) -> list[FitResult]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits = fit_results(results, fit_min_N=fit_min_N)
    with path.open("w", newline="") as handle:
        fieldnames = [
            "d",
            "alpha",
            "c_label",
            "c_value",
            "L",
            "fit_min_N",
            "point_count",
            "N_min",
            "N_max",
            "amplitude",
            "fitted_beta",
            "r2",
            "predicted_beta",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for fit in fits:
            writer.writerow(
                {
                    "d": fit.d,
                    "alpha": f"{fit.alpha:g}",
                    "c_label": fit.c_label,
                    "c_value": f"{fit.c_value:.12g}",
                    "L": fit.L,
                    "fit_min_N": fit.fit_min_N,
                    "point_count": fit.point_count,
                    "N_min": fit.N_min,
                    "N_max": fit.N_max,
                    "amplitude": f"{fit.amplitude:.12g}",
                    "fitted_beta": f"{fit.fitted_beta:.12g}",
                    "r2": f"{fit.r2:.12g}",
                    "predicted_beta": f"{fit.predicted_beta:.12g}",
                }
            )
    return fits


def fit_results(
    results: list[DataScalingResult],
    *,
    fit_min_N: int,
) -> list[FitResult]:
    finite = [
        row
        for row in results
        if row.failed_profile_samples == 0
        and math.isfinite(row.regret_over_ln_d)
        and row.regret_over_ln_d > 0.0
        and row.N >= fit_min_N
    ]
    fits = []
    for d in sorted({row.d for row in finite}):
        d_rows = [row for row in finite if row.d == d]
        for alpha in sorted({row.alpha for row in d_rows}):
            alpha_rows = [row for row in d_rows if row.alpha == alpha]
            for c_label in c_order([row.c_label for row in alpha_rows]):
                curve = sorted(
                    [row for row in alpha_rows if row.c_label == c_label],
                    key=lambda row: row.N,
                )
                if len(curve) < 2:
                    continue
                amplitude, beta, r2 = power_fit(
                    [float(row.N) for row in curve],
                    [row.regret_over_ln_d for row in curve],
                )
                fits.append(
                    FitResult(
                        d=d,
                        alpha=alpha,
                        c_label=c_label,
                        c_value=curve[0].c_value,
                        L=curve[0].L,
                        fit_min_N=fit_min_N,
                        point_count=len(curve),
                        N_min=curve[0].N,
                        N_max=curve[-1].N,
                        amplitude=amplitude,
                        fitted_beta=beta,
                        r2=r2,
                        predicted_beta=predicted_beta(alpha),
                    )
                )
    return fits


def write_plots(
    pdf_path: Path,
    results: list[DataScalingResult],
    *,
    c_specs: tuple[CSpec, ...],
    fit_min_N: int,
) -> None:
    finite = [
        row
        for row in results
        if row.failed_profile_samples == 0
        and math.isfinite(row.regret_over_ln_d)
        and row.regret_over_ln_d > 0.0
    ]
    if not finite:
        return
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(pdf_path), pagesize=landscape(letter))
    pdf.setTitle("Data scaling for logarithmic depth")
    width, height = landscape(letter)
    draw_data_scaling_page(
        pdf,
        width=width,
        height=height,
        rows=finite,
        c_specs=c_specs,
        fit_min_N=fit_min_N,
    )
    pdf.save()


def draw_data_scaling_page(
    pdf: canvas.Canvas,
    *,
    width: float,
    height: float,
    rows: list[DataScalingResult],
    c_specs: tuple[CSpec, ...],
    fit_min_N: int,
) -> None:
    alphas = sorted({row.alpha for row in rows})
    panel_alphas = alphas[:4]
    c_labels = c_order([row.c_label for row in rows])
    spec_by_label = {spec.label: spec for spec in c_specs}
    styles = c_styles(c_labels)

    margin_x = 62.0
    margin_y = 62.0
    gutter_x = 54.0
    gutter_y = 52.0
    panel_w = (width - 2 * margin_x - gutter_x) / 2
    panel_h = (height - 2 * margin_y - gutter_y - 12) / 2
    positions = [
        (margin_x, margin_y + panel_h + gutter_y),
        (margin_x + panel_w + gutter_x, margin_y + panel_h + gutter_y),
        (margin_x, margin_y),
        (margin_x + panel_w + gutter_x, margin_y),
    ]

    for index, alpha in enumerate(panel_alphas):
        x0, y0 = positions[index]
        draw_alpha_panel(
            pdf,
            x0=x0,
            y0=y0,
            w=panel_w,
            h=panel_h,
            rows=[row for row in rows if row.alpha == alpha],
            alpha=alpha,
            c_labels=c_labels,
            spec_by_label=spec_by_label,
            styles=styles,
            fit_min_N=fit_min_N,
            draw_x_label=index >= 2,
            draw_y_label=index % 2 == 0,
        )

    draw_shared_legend(
        pdf,
        x=width - 220,
        y=height - 28,
        c_labels=c_labels,
        spec_by_label=spec_by_label,
        styles=styles,
    )


def draw_alpha_panel(
    pdf: canvas.Canvas,
    *,
    x0: float,
    y0: float,
    w: float,
    h: float,
    rows: list[DataScalingResult],
    alpha: float,
    c_labels: list[str],
    spec_by_label: dict[str, CSpec],
    styles: dict[str, tuple[colors.Color, tuple[int, ...] | None, str]],
    fit_min_N: int,
    draw_x_label: bool,
    draw_y_label: bool,
) -> None:
    if not rows:
        return
    log_x_values = [math.log10(row.N) for row in rows]
    log_y_values = [math.log10(row.regret_over_ln_d) for row in rows]
    x_min = math.floor(min(log_x_values))
    x_max = math.ceil(max(log_x_values))
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    y_min, y_max = padded_log_range(log_y_values)

    def sx(N: float) -> float:
        return x0 + (math.log10(N) - x_min) / (x_max - x_min) * w

    def sy(value: float) -> float:
        return y0 + (math.log10(value) - y_min) / (y_max - y_min) * h

    draw_log_axes(pdf, x0, y0, w, h, x_min, x_max, y_min, y_max)
    pdf.setFont("Helvetica-Bold", 9.5)
    pdf.drawCentredString(x0 + w / 2, y0 + h + 10, f"Zipf alpha = {alpha:g}")
    pdf.setFont("Helvetica", 8.2)
    if draw_x_label:
        pdf.drawCentredString(x0 + w / 2, y0 - 34, "sample size N")
    if draw_y_label:
        pdf.saveState()
        pdf.translate(x0 - 46, y0 + h / 2)
        pdf.rotate(90)
        pdf.drawCentredString(0, 0, "normalized regret R_N / ln(d)")
        pdf.restoreState()

    pred_beta = predicted_beta(alpha)
    if pred_beta > 0.0:
        anchor = sorted(rows, key=lambda row: abs(math.log(row.N) - (x_min + x_max) * math.log(10.0) / 2.0))[0]
        x_left = 10.0**x_min
        x_right = 10.0**x_max
        y_left = anchor.regret_over_ln_d * (x_left / anchor.N) ** (-pred_beta)
        y_right = anchor.regret_over_ln_d * (x_right / anchor.N) ** (-pred_beta)
        if y_left > 0.0 and y_right > 0.0:
            pdf.saveState()
            pdf.setStrokeColor(colors.HexColor("#777777"))
            pdf.setLineWidth(1.0)
            pdf.setDash(4, 3)
            pdf.line(sx(x_left), sy(y_left), sx(x_right), sy(y_right))
            pdf.setDash()
            pdf.setFillColor(colors.HexColor("#555555"))
            pdf.setFont("Helvetica", 7.4)
            pdf.drawRightString(
                x0 + w - 6,
                y0 + h - 12,
                f"predicted beta={pred_beta:.3g}",
            )
            pdf.restoreState()
    else:
        pdf.saveState()
        pdf.setFillColor(colors.HexColor("#555555"))
        pdf.setFont("Helvetica", 7.4)
        pdf.drawRightString(x0 + w - 6, y0 + h - 12, "marginal")
        pdf.restoreState()

    for c_label in c_labels:
        curve = sorted([row for row in rows if row.c_label == c_label], key=lambda row: row.N)
        if not curve:
            continue
        color, dash, marker = styles[c_label]
        pdf.saveState()
        pdf.setStrokeColor(color)
        pdf.setFillColor(color)
        pdf.setLineWidth(1.45)
        if dash is not None:
            pdf.setDash(list(dash))
        previous = None
        for row in curve:
            point = (sx(row.N), sy(row.regret_over_ln_d))
            if previous is not None:
                pdf.line(previous[0], previous[1], point[0], point[1])
            previous = point
        pdf.setDash()
        for row in curve:
            x = sx(row.N)
            y = sy(row.regret_over_ln_d)
            err = row.stderr_over_ln_d
            if math.isfinite(err) and err > 0.0 and row.regret_over_ln_d > err:
                y_lo = sy(row.regret_over_ln_d - err)
                y_hi = sy(row.regret_over_ln_d + err)
                pdf.line(x, y_lo, x, y_hi)
                pdf.line(x - 2.0, y_lo, x + 2.0, y_lo)
                pdf.line(x - 2.0, y_hi, x + 2.0, y_hi)
            draw_marker(pdf, x, y, marker, 3.1)
        fit_curve = [row for row in curve if row.N >= fit_min_N]
        if len(fit_curve) >= 2:
            amplitude, beta, _ = power_fit(
                [float(row.N) for row in fit_curve],
                [row.regret_over_ln_d for row in fit_curve],
            )
            x_left = fit_curve[0].N
            x_right = fit_curve[-1].N
            y_left = amplitude * x_left ** (-beta)
            y_right = amplitude * x_right ** (-beta)
            pdf.setStrokeColor(color)
            pdf.setLineWidth(0.9)
            pdf.setDash(1, 3)
            pdf.line(sx(x_left), sy(y_left), sx(x_right), sy(y_right))
            pdf.setDash()
        pdf.restoreState()


def draw_shared_legend(
    pdf: canvas.Canvas,
    *,
    x: float,
    y: float,
    c_labels: list[str],
    spec_by_label: dict[str, CSpec],
    styles: dict[str, tuple[colors.Color, tuple[int, ...] | None, str]],
) -> None:
    pdf.setFont("Helvetica", 8.0)
    for index, c_label in enumerate(c_labels):
        color, dash, marker = styles[c_label]
        row_y = y - 15 * index
        pdf.saveState()
        pdf.setStrokeColor(color)
        pdf.setFillColor(color)
        pdf.setLineWidth(1.4)
        if dash is not None:
            pdf.setDash(list(dash))
        pdf.line(x, row_y, x + 22, row_y)
        pdf.setDash()
        draw_marker(pdf, x + 11, row_y, marker, 3.0)
        pdf.restoreState()
        pdf.setFillColor(colors.black)
        pdf.drawString(x + 28, row_y - 3, spec_by_label[c_label].plot_label())
    row_y = y - 15 * len(c_labels)
    pdf.saveState()
    pdf.setStrokeColor(colors.HexColor("#777777"))
    pdf.setDash(4, 3)
    pdf.line(x, row_y, x + 22, row_y)
    pdf.restoreState()
    pdf.setFillColor(colors.black)
    pdf.drawString(x + 28, row_y - 3, "predicted slope")


def c_styles(
    labels: list[str],
) -> dict[str, tuple[colors.Color, tuple[int, ...] | None, str]]:
    palette = [
        colors.HexColor("#0072B2"),
        colors.HexColor("#D55E00"),
        colors.HexColor("#009E73"),
        colors.HexColor("#CC79A7"),
        colors.HexColor("#000000"),
    ]
    dashes = [None, (6, 3), (2, 2), (8, 2, 2, 2), (3, 2, 1, 2)]
    markers = ["circle", "square", "triangle", "diamond", "cross"]
    return {
        label: (
            palette[index % len(palette)],
            dashes[index % len(dashes)],
            markers[index % len(markers)],
        )
        for index, label in enumerate(labels)
    }


def parse_c_specs(raw: str) -> tuple[CSpec, ...]:
    specs = []
    for token in [part.strip() for part in raw.split(",") if part.strip()]:
        if token.lower() in {"cstar", "c*", "star"}:
            specs.append(CSpec("cstar", C_STAR))
        else:
            value = float(token)
            label = f"c{value:g}".replace(".", "p")
            specs.append(CSpec(label, value))
    return tuple(specs)


def predicted_beta(alpha: float) -> float:
    if alpha <= 1.0:
        return 0.0
    return 1.0 - 1.0 / alpha


def c_order(labels: list[str]) -> list[str]:
    preferred = ["c1", "cstar", "c4"]
    present = set(labels)
    ordered = [label for label in preferred if label in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def sort_results(results: list[DataScalingResult]) -> list[DataScalingResult]:
    c_rank = {label: index for index, label in enumerate(c_order([row.c_label for row in results]))}
    return sorted(
        results,
        key=lambda row: (
            row.d,
            row.alpha,
            c_rank.get(row.c_label, 999),
            row.N,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-csv", type=Path, default=None)
    parser.add_argument("--d", type=int, default=100000)
    parser.add_argument("--alphas", type=str, default="1,2,3,4")
    parser.add_argument("--c-values", type=str, default="cstar")
    parser.add_argument("--n-values", type=str, default="20,50,100,200,500,1000,2000,5000")
    parser.add_argument("--fit-min-n", type=int, default=50)
    parser.add_argument("--profile-samples", type=int, default=5000)
    parser.add_argument("--profile-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--u-min", type=float, default=None)
    parser.add_argument("--u-max", type=float, default=None)
    parser.add_argument("--u-points", type=int, default=8001)
    parser.add_argument("--laguerre-order", type=int, default=96)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("output/data_scaling/data_scaling.csv"),
    )
    parser.add_argument(
        "--out-summary-csv",
        type=Path,
        default=Path("output/data_scaling/data_scaling_summary.csv"),
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=Path("output/data_scaling/data_scaling.pdf"),
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    c_specs = parse_c_specs(args.c_values)
    if args.from_csv is not None:
        results = read_results_csv(args.from_csv)
        write_summary_csv(args.out_summary_csv, results, fit_min_N=args.fit_min_n)
        if not args.no_plots:
            write_plots(args.out_pdf, results, c_specs=c_specs, fit_min_N=args.fit_min_n)
        print(f"wrote {args.out_summary_csv}", flush=True)
        if not args.no_plots:
            print(f"wrote {args.out_pdf}", flush=True)
        return

    alphas = tuple(parse_floats(args.alphas))
    n_values = parse_ints(args.n_values)
    if args.d <= 0:
        raise ValueError("--d must be positive")
    if any(N <= 0 for N in n_values):
        raise ValueError("--n-values must be positive")
    if args.profile_samples <= 0:
        raise ValueError("--profile-samples must be positive")
    if args.profile_batch_size <= 0:
        raise ValueError("--profile-batch-size must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    jobs = []
    for c_index, c_spec in enumerate(c_specs):
        for n_index, N in enumerate(n_values):
            jobs.append(
                {
                    "d": args.d,
                    "alphas": alphas,
                    "c_spec": c_spec,
                    "N": N,
                    "profile_samples": args.profile_samples,
                    "profile_batch_size": args.profile_batch_size,
                    "seed": args.seed + 1_000_003 * c_index + 20_011 * n_index,
                    "u_min": args.u_min,
                    "u_max": args.u_max,
                    "u_points": args.u_points,
                    "laguerre_order": args.laguerre_order,
                    "chunk_size": args.chunk_size,
                }
            )

    results: list[DataScalingResult] = []
    if args.workers == 1:
        for job in jobs:
            results.extend(run_one_n_group(**job))
            write_results_csv(args.out_csv, results)
            write_summary_csv(args.out_summary_csv, results, fit_min_N=args.fit_min_n)
            if not args.no_plots:
                write_plots(args.out_pdf, results, c_specs=c_specs, fit_min_N=args.fit_min_n)
            print(f"wrote partial results to {args.out_csv}", flush=True)
    else:
        print(f"running {len(jobs)} N-groups with {args.workers} workers", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_one_n_group, **job) for job in jobs]
            for future in as_completed(futures):
                results.extend(future.result())
                write_results_csv(args.out_csv, results)
                write_summary_csv(args.out_summary_csv, results, fit_min_N=args.fit_min_n)
                if not args.no_plots:
                    write_plots(args.out_pdf, results, c_specs=c_specs, fit_min_N=args.fit_min_n)
                print(f"wrote partial results to {args.out_csv}", flush=True)

    results = sort_results(results)
    write_results_csv(args.out_csv, results)
    write_summary_csv(args.out_summary_csv, results, fit_min_N=args.fit_min_n)
    if not args.no_plots:
        write_plots(args.out_pdf, results, c_specs=c_specs, fit_min_N=args.fit_min_n)
    print(f"wrote {args.out_csv}", flush=True)
    print(f"wrote {args.out_summary_csv}", flush=True)
    if not args.no_plots:
        print(f"wrote {args.out_pdf}", flush=True)


if __name__ == "__main__":
    main()
