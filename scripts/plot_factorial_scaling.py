"""Create the three consistently named figures for the factorial scaling grid."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas

from analyze_factorial_scaling import normalized_prefactor, valid_rows
from data_scaling_experiment import DataScalingResult, predicted_beta, read_results_csv


PALETTE = (colors.HexColor("#2166ac"), colors.HexColor("#b2182b"),
           colors.HexColor("#1b7837"), colors.HexColor("#762a83"),
           colors.HexColor("#e08214"), colors.HexColor("#4d4d4d"),
           colors.HexColor("#35978f"))


def _map(value: float, lo: float, hi: float, start: float, stop: float) -> float:
    return start + (value - lo) * (stop - start) / (hi - lo)


def _panel(pdf: canvas.Canvas, *, x: float, y: float, w: float, h: float,
           title: str, xlabel: str, ylabel: str, curves: list[tuple[str, list[tuple[float, float]]]],
           log_x: bool = True, log_y: bool = True) -> None:
    transformed = []
    for label, points in curves:
        transformed.append((label, [((math.log10(a) if log_x else a),
                                     (math.log10(b) if log_y else b)) for a, b in points]))
    xs = [a for _, points in transformed for a, _ in points]
    ys = [b for _, points in transformed for _, b in points]
    xlo, xhi, ylo, yhi = min(xs), max(xs), min(ys), max(ys)
    data_xlo, data_xhi, data_ylo, data_yhi = xlo, xhi, ylo, yhi
    padx = max((xhi - xlo) * .06, 1e-9); pady = max((yhi - ylo) * .09, 1e-9)
    xlo -= padx; xhi += padx; ylo -= pady; yhi += pady
    left, bottom, right, top = x + 46, y + 38, x + w - 10, y + h - 25
    pdf.setFillColor(colors.black)
    pdf.setStrokeColor(colors.HexColor("#444444")); pdf.setLineWidth(.7)
    pdf.line(left, bottom, left, top); pdf.line(left, bottom, right, bottom)
    pdf.setFont("Helvetica-Bold", 9); pdf.drawString(left, top + 20, title)
    pdf.setFont("Helvetica", 8); pdf.drawCentredString((left + right) / 2, y + 8, xlabel)
    pdf.saveState(); pdf.translate(x + 10, (bottom + top) / 2); pdf.rotate(90)
    pdf.drawCentredString(0, 0, ylabel); pdf.restoreState()
    pdf.setFont("Helvetica", 6.5); pdf.setFillColor(colors.HexColor("#444444"))
    for fraction in (0.0, 0.5, 1.0):
        xv = data_xlo + fraction * (data_xhi - data_xlo); px = _map(xv, xlo, xhi, left, right)
        yv = data_ylo + fraction * (data_yhi - data_ylo); py = _map(yv, ylo, yhi, bottom, top)
        x_label = f"{10**xv:.3g}" if log_x else f"{xv:.3g}"
        y_label = f"{10**yv:.3g}" if log_y else f"{yv:.3g}"
        pdf.drawCentredString(px, bottom - 11, x_label)
        pdf.drawRightString(left - 4, py - 2, y_label)
    for idx, (label, points) in enumerate(transformed):
        color = PALETTE[idx % len(PALETTE)]; pdf.setStrokeColor(color); pdf.setFillColor(color)
        coords = [(_map(a, xlo, xhi, left, right), _map(b, ylo, yhi, bottom, top)) for a, b in points]
        if len(coords) > 1:
            path = pdf.beginPath(); path.moveTo(*coords[0])
            for point in coords[1:]: path.lineTo(*point)
            pdf.drawPath(path)
        for px, py in coords: pdf.circle(px, py, 2.1, fill=1, stroke=0)
        legend_x = left + (idx % 3) * (right - left) / 3
        legend_y = top - 7 - (idx // 3) * 8
        pdf.circle(legend_x, legend_y, 1.8, fill=1, stroke=0)
        pdf.setFont("Helvetica", 6.5); pdf.drawString(legend_x + 4, legend_y - 2, label)


def _pages(rows: list[DataScalingResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    alphas = [a for a in sorted({r.alpha for r in rows}) if a > 1]
    specs = [
        ("data_scaling.pdf", "Data scaling", "N", "R_N / ln d",
         lambda a: [(f"d={d:g}", [(r.N, r.regret_over_ln_d) for r in sorted(rows, key=lambda q:q.N)
                                   if r.alpha == a and r.d == d]) for d in sorted({r.d for r in rows})]),
        ("alphabet_scaling.pdf", "Alphabet scaling", "d", "normalized prefactor",
         lambda a: [(f"N={n:g}", [(r.d, normalized_prefactor(r)) for r in sorted(rows, key=lambda q:q.d)
                                   if r.alpha == a and r.N == n]) for n in (100, 500, 2000)]),
        ("scaling_law_collapse.pdf", "Scaling-law collapse", "N", "R_N N^beta / ln d",
         lambda a: [(f"d={d:g}", [(r.N, normalized_prefactor(r)) for r in sorted(rows, key=lambda q:q.N)
                                   if r.alpha == a and r.d == d]) for d in sorted({r.d for r in rows})]),
    ]
    for filename, heading, xlabel, ylabel, make_curves in specs:
        pdf = canvas.Canvas(str(out_dir / filename), pagesize=landscape(letter))
        pdf.setTitle(heading); width, height = landscape(letter)
        pdf.setFont("Helvetica-Bold", 14); pdf.drawString(42, height - 28, heading)
        panel_w = (width - 102) / 3
        for i, alpha in enumerate(alphas):
            _panel(pdf, x=34 + i * (panel_w + 17), y=45, w=panel_w, h=height - 92,
                   title=f"alpha={alpha:g} (beta={predicted_beta(alpha):.3f})",
                   xlabel=xlabel, ylabel=ylabel, curves=make_curves(alpha),
                   log_x=True, log_y=filename == "data_scaling.pdf")
        pdf.save(); print(f"wrote {out_dir / filename}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("output/scaling_law_factorial"))
    args = parser.parse_args()
    rows = valid_rows(read_results_csv(args.from_csv))
    if not rows: raise ValueError("input contains no valid rows")
    _pages(rows, args.out_dir)


if __name__ == "__main__":
    main()
