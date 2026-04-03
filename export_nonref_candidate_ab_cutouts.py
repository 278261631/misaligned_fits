from pathlib import Path
import argparse
import csv
import json
from datetime import datetime
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch

from alignment_common import eval_poly


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export A/B cutout comparison PNGs for each row in "
            "variable_candidates_nonref_only_inner_border.csv and overlay stars.all.npz info."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("variable_candidates_nonref_only_inner_border.csv"),
        help="Input candidate CSV path.",
    )
    parser.add_argument("--a-fits", type=Path, required=True, help="Reference A FITS path.")
    parser.add_argument("--b-fits", type=Path, required=True, help="Target B FITS path.")
    parser.add_argument("--a-stars-all", type=Path, required=True, help="A stars.all NPZ path (xy, flux).")
    parser.add_argument("--b-stars-all", type=Path, required=True, help="B stars.all NPZ path (xy, flux).")
    parser.add_argument(
        "--align-npz",
        type=Path,
        default=None,
        help="Optional A->B alignment NPZ (cx/cy/fit_degree). Required unless --b-aligned-to-a is set.",
    )
    parser.add_argument(
        "--b-aligned-to-a",
        action="store_true",
        help="Treat B as already aligned to A pixel grid (use same center x/y for B cutout).",
    )
    parser.add_argument("--cutout-size", type=int, default=128, help="Square cutout size in pixels.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input_csv_dir>/output).",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Maximum candidate rows to export (<=0 means all).")
    parser.add_argument(
        "--annotate-top-n",
        type=int,
        default=8,
        help="Annotate top-N brightest stars (per panel) by flux.",
    )
    parser.add_argument("--dpi", type=int, default=140, help="Output PNG DPI.")
    parser.add_argument(
        "--timing-jsonl",
        type=Path,
        default=None,
        help="Optional timing JSONL path (default: <input_csv_dir>/timing.jsonl).",
    )
    parser.add_argument(
        "--timing-plot",
        type=Path,
        default=None,
        help="Output path for timing summary PNG (default: <out_dir>/timing_summary.png).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs; default skips when cutout outputs already exist.",
    )
    return parser.parse_args()


def parse_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def resolve_path(input_csv: Path, maybe_path: Path | None):
    if maybe_path is None:
        return None
    if maybe_path.is_absolute():
        return maybe_path
    return input_csv.parent / maybe_path


def load_stars_npz(path: Path):
    dat = np.load(path, allow_pickle=True)
    if "xy" not in dat or "flux" not in dat:
        raise RuntimeError(f"Invalid stars NPZ (need xy/flux): {path}")
    xy = np.asarray(dat["xy"], dtype=np.float64)
    flux = np.asarray(dat["flux"], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise RuntimeError(f"Invalid xy shape in {path}: {xy.shape}")
    if flux.ndim != 1 or len(flux) != len(xy):
        raise RuntimeError(f"Invalid flux shape in {path}: {flux.shape}, xy={xy.shape}")
    return xy, flux


def load_alignment(path: Path | None):
    if path is None:
        return None, None, None
    sol = np.load(path, allow_pickle=True)
    if "cx" not in sol or "cy" not in sol or "fit_degree" not in sol:
        raise RuntimeError(f"Invalid align NPZ (need cx/cy/fit_degree): {path}")
    cx = np.asarray(sol["cx"], dtype=np.float64)
    cy = np.asarray(sol["cy"], dtype=np.float64)
    fit_degree = int(np.asarray(sol["fit_degree"]).ravel()[0])
    return cx, cy, fit_degree


def extract_cutout(img, cx, cy, size):
    h, w = img.shape
    half = int(size) // 2
    x0 = int(np.round(float(cx))) - half
    y0 = int(np.round(float(cy))) - half
    x1 = x0 + int(size)
    y1 = y0 + int(size)

    out = np.full((int(size), int(size)), np.nan, dtype=np.float32)

    sx0 = max(0, x0)
    sx1 = min(w, x1)
    sy0 = max(0, y0)
    sy1 = min(h, y1)

    if sx1 > sx0 and sy1 > sy0:
        dx0 = sx0 - x0
        dx1 = dx0 + (sx1 - sx0)
        dy0 = sy0 - y0
        dy1 = dy0 + (sy1 - sy0)
        out[dy0:dy1, dx0:dx1] = img[sy0:sy1, sx0:sx1].astype(np.float32)
    return out, x0, y0


def stars_in_cutout(xy, flux, x0, y0, size):
    x = np.asarray(xy[:, 0], dtype=np.float64)
    y = np.asarray(xy[:, 1], dtype=np.float64)
    m = (x >= x0) & (x < (x0 + size)) & (y >= y0) & (y < (y0 + size))
    if not np.any(m):
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)
    p = np.column_stack([x[m] - x0, y[m] - y0]).astype(np.float64)
    f = np.asarray(flux[m], dtype=np.float64)
    return p, f


def safe_norm(img):
    finite = np.isfinite(img)
    fill = float(np.nanmedian(img[finite])) if np.any(finite) else 0.0
    view = np.where(finite, img, fill)
    norm = ImageNormalize(view, interval=PercentileInterval(99.5), stretch=SqrtStretch())
    return view, norm


def short_flux_summary(flux, n_show=3):
    if len(flux) == 0:
        return "n=0"
    order = np.argsort(flux)[::-1]
    top = [f"{float(flux[i]):.2f}" for i in order[: max(0, int(n_show))]]
    return f"n={len(flux)}, top={';'.join(top)}"


def annotate_star_flux(ax, points, flux, top_n):
    if len(points) == 0 or int(top_n) <= 0:
        return
    order = np.argsort(flux)[::-1]
    k = min(int(top_n), len(order))
    for i in order[:k]:
        q = points[i]
        ax.text(
            float(q[0]) + 1.2,
            float(q[1]) + 1.2,
            f"{float(flux[i]):.1f}",
            color="#00E5FF",
            fontsize=5,
            ha="left",
            va="bottom",
            alpha=0.95,
        )


def draw_crosshair(ax, x, y, gap=2.6, arm=8.0, color="#FFD400", lw=1.2):
    x = float(x)
    y = float(y)
    gap = float(gap)
    arm = float(arm)
    # Four-segment reticle: leave center empty so the target stays visible.
    ax.plot([x - arm, x - gap], [y, y], color=color, linewidth=lw)
    ax.plot([x + gap, x + arm], [y, y], color=color, linewidth=lw)
    ax.plot([x, x], [y - arm, y - gap], color=color, linewidth=lw)
    ax.plot([x, x], [y + gap, y + arm], color=color, linewidth=lw)


def load_timing_events(path: Path, run_id: str | None = None, script: str | None = None):
    if not path.exists():
        return []
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                item = json.loads(s)
            except Exception:
                continue
            if run_id is not None and str(item.get("run_id", "")) != str(run_id):
                continue
            if script is not None and str(item.get("script", "")) != str(script):
                continue
            duration_ms = parse_float(item.get("duration_ms"))
            if duration_ms is None:
                continue
            events.append(
                {
                    "ts": str(item.get("ts", "")),
                    "script": str(item.get("script", "")),
                    "step": str(item.get("step", "")),
                    "duration_ms": float(duration_ms),
                    "status": str(item.get("status", "")),
                    "run_id": str(item.get("run_id", "")),
                }
            )
    return events


def save_timing_summary_png(events, out_png: Path):
    if len(events) == 0:
        return False

    def parse_iso_ts(ts: str):
        s = str(ts).strip()
        if not s:
            return None
        try:
            # JSONL uses trailing "Z" for UTC.
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None

    timeline = []
    for idx, ev in enumerate(events):
        ts = parse_iso_ts(ev.get("ts", ""))
        dur_sec = max(0.0, float(ev.get("duration_ms", 0.0)) / 1000.0)
        timeline.append(
            {
                "idx": idx,
                "ts": ts,
                "dur_sec": dur_sec,
                "step": str(ev.get("step", "")),
                "script": str(ev.get("script", "")),
                "status": str(ev.get("status", "")),
            }
        )

    timeline.sort(key=lambda it: (it["ts"] is None, it["ts"], it["idx"]))

    first_ts = None
    for it in timeline:
        if it["ts"] is not None:
            first_ts = it["ts"]
            break

    cursor_sec = 0.0
    for it in timeline:
        if it["ts"] is not None and first_ts is not None:
            end_sec = max(0.0, (it["ts"] - first_ts).total_seconds())
            start_sec = max(0.0, end_sec - it["dur_sec"])
        else:
            # Fallback for invalid/missing timestamps: pack sequentially.
            start_sec = cursor_sec
            end_sec = start_sec + it["dur_sec"]
        it["start_sec"] = start_sec
        it["end_sec"] = end_sec
        cursor_sec = max(cursor_sec, end_sec)

    n_error = sum(1 for it in timeline if str(it["status"]) != "ok")
    total_sec = float(sum(it["dur_sec"] for it in timeline))
    labels = [f"{i + 1:03d} {it['script']}::{it['step']}" for i, it in enumerate(timeline)]
    starts = [it["start_sec"] for it in timeline]
    widths = [it["dur_sec"] for it in timeline]
    colors = ["#4C78A8" if str(it["status"]) == "ok" else "#E45756" for it in timeline]

    fig_h = min(36.0, max(8.0, 0.26 * len(labels) + 2.5))
    fig = plt.figure(figsize=(14, fig_h))
    ax = fig.add_subplot(111)
    y = np.arange(len(labels))
    ax.barh(y, widths, left=starts, color=colors, edgecolor="none")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Timeline (seconds from first event)")
    ax.set_title("Timing Timeline (All events)")
    ax.grid(True, axis="x", alpha=0.25, linestyle="--")
    ax.invert_yaxis()

    script_totals = defaultdict(float)
    for it in timeline:
        script_totals[it["script"]] += it["dur_sec"]
    script_items = sorted(script_totals.items(), key=lambda kv: kv[1], reverse=True)
    top_script_txt = ", ".join([f"{k}:{v:.2f}s" for k, v in script_items[:5]])
    span_sec = max([it["end_sec"] for it in timeline], default=0.0)
    fig.text(
        0.01,
        0.01,
        f"events={len(events)}, errors={n_error}, sum_durations={total_sec:.2f}s, timeline_span={span_sec:.2f}s, top_scripts={top_script_txt}",
        fontsize=8,
        ha="left",
        va="bottom",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    input_csv = args.input_csv
    n_ok = 0
    n_skipped = 0
    if not input_csv.exists():
        raise RuntimeError(f"Input CSV not found: {input_csv}")

    out_dir = args.out_dir if args.out_dir is not None else (input_csv.parent / "output")
    out_dir = resolve_path(input_csv, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timing_plot_path = resolve_path(input_csv, args.timing_plot) if args.timing_plot is not None else (out_dir / "timing_summary.png")
    timing_path = resolve_path(input_csv, args.timing_jsonl) if args.timing_jsonl is not None else (input_csv.parent / "timing.jsonl")
    existing_cutouts = list(out_dir.glob("rank_*_ab.png"))
    if (not args.overwrite) and len(existing_cutouts) > 0 and timing_plot_path.exists():
        print("SKIP export_nonref_candidate_ab_cutouts.py: outputs already exist (use --overwrite to regenerate)")
        print(f"EXISTS {timing_plot_path}")
        print(f"EXISTS cutouts={len(existing_cutouts)} in {out_dir}")
        return

    a_fits = resolve_path(input_csv, args.a_fits)
    b_fits = resolve_path(input_csv, args.b_fits)
    a_stars_all = resolve_path(input_csv, args.a_stars_all)
    b_stars_all = resolve_path(input_csv, args.b_stars_all)
    align_npz = resolve_path(input_csv, args.align_npz) if args.align_npz is not None else None

    if a_fits is None or not a_fits.exists():
        raise RuntimeError(f"A FITS not found: {a_fits}")
    if b_fits is None or not b_fits.exists():
        raise RuntimeError(f"B FITS not found: {b_fits}")
    if a_stars_all is None or not a_stars_all.exists():
        raise RuntimeError(f"A stars.all NPZ not found: {a_stars_all}")
    if b_stars_all is None or not b_stars_all.exists():
        raise RuntimeError(f"B stars.all NPZ not found: {b_stars_all}")
    if (not bool(args.b_aligned_to_a)) and align_npz is None:
        raise RuntimeError("Need --align-npz unless --b-aligned-to-a is set.")
    if align_npz is not None and (not align_npz.exists()):
        raise RuntimeError(f"Align NPZ not found: {align_npz}")

    cutout_size = int(args.cutout_size)
    if cutout_size <= 0:
        raise RuntimeError("--cutout-size must be > 0.")

    a_img = fits.getdata(a_fits).astype(np.float32)
    b_img = fits.getdata(b_fits).astype(np.float32)
    a_xy, a_flux = load_stars_npz(a_stars_all)
    b_xy, b_flux = load_stars_npz(b_stars_all)
    cx, cy, fit_degree = load_alignment(align_npz)

    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if int(args.max_rows) > 0:
        rows = rows[: int(args.max_rows)]

    for row_idx, row in enumerate(rows, start=1):
            rank_raw = row.get("rank", "")
            x = parse_float(row.get("x"))
            y = parse_float(row.get("y"))
            if x is None or y is None:
                n_skipped += 1
                continue

            if bool(args.b_aligned_to_a):
                bx = float(x)
                by = float(y)
            else:
                bx_arr, by_arr = eval_poly(
                    np.asarray([float(x)], dtype=np.float64),
                    np.asarray([float(y)], dtype=np.float64),
                    cx,
                    cy,
                    degree=int(fit_degree),
                )
                bx = float(bx_arr[0])
                by = float(by_arr[0])

            a_patch, a_x0, a_y0 = extract_cutout(a_img, x, y, cutout_size)
            b_patch, b_x0, b_y0 = extract_cutout(b_img, bx, by, cutout_size)

            a_pts, a_f = stars_in_cutout(a_xy, a_flux, a_x0, a_y0, cutout_size)
            b_pts, b_f = stars_in_cutout(b_xy, b_flux, b_x0, b_y0, cutout_size)

            a_view, a_norm = safe_norm(a_patch)
            b_view, b_norm = safe_norm(b_patch)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=int(args.dpi))
            ax_a, ax_b = axes
            ax_a.imshow(a_view, origin="lower", cmap="gray", norm=a_norm, interpolation="nearest")
            ax_b.imshow(b_view, origin="lower", cmap="gray", norm=b_norm, interpolation="nearest")

            draw_crosshair(ax_a, float(x) - a_x0, float(y) - a_y0)
            draw_crosshair(ax_b, float(bx) - b_x0, float(by) - b_y0)

            if len(a_pts) > 0:
                ax_a.scatter(
                    a_pts[:, 0], a_pts[:, 1], s=16, facecolors="none", edgecolors="#00E5FF", linewidths=0.8, alpha=0.9
                )
            if len(b_pts) > 0:
                ax_b.scatter(
                    b_pts[:, 0], b_pts[:, 1], s=16, facecolors="none", edgecolors="#00E5FF", linewidths=0.8, alpha=0.9
                )

            annotate_star_flux(ax_a, a_pts, a_f, top_n=int(args.annotate_top_n))
            annotate_star_flux(ax_b, b_pts, b_f, top_n=int(args.annotate_top_n))

            ax_a.text(
                0.02,
                0.98,
                short_flux_summary(a_f),
                transform=ax_a.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                color="white",
                bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=2),
            )
            ax_b.text(
                0.02,
                0.98,
                short_flux_summary(b_f),
                transform=ax_b.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                color="white",
                bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=2),
            )

            ax_a.set_title("A cutout + A stars.all")
            ax_b.set_title("B cutout + B stars.all")
            for ax in axes:
                ax.set_xlim(-0.5, cutout_size - 0.5)
                ax.set_ylim(cutout_size - 0.5, -0.5)
                ax.set_axis_off()

            rank_label = str(rank_raw).strip()
            try:
                rank_int = int(float(rank_label))
                rank_tag = f"{rank_int:04d}"
            except Exception:
                rank_tag = f"r{row_idx:04d}"
            out_name = f"rank_{rank_tag}_x{float(x):.2f}_y{float(y):.2f}_ab.png"
            out_path = out_dir / out_name
            fig.tight_layout()
            fig.savefig(out_path, dpi=int(args.dpi))
            plt.close(fig)
            n_ok += 1

    events = load_timing_events(
        timing_path,
        run_id=None,
        script="rank_variable_candidates.py",
    )
    wrote_plot = save_timing_summary_png(events, timing_plot_path)

    print(f"Input CSV: {input_csv}")
    print(f"A FITS: {a_fits}")
    print(f"B FITS: {b_fits}")
    print(f"A stars.all: {a_stars_all}")
    print(f"B stars.all: {b_stars_all}")
    if bool(args.b_aligned_to_a):
        print("B center source: same as A (--b-aligned-to-a)")
    else:
        print(f"B center source: align NPZ {align_npz}, fit_degree={fit_degree}")
    print(f"cutout_size={cutout_size}")
    print(f"timing_jsonl={timing_path}")
    if wrote_plot:
        print(f"WROTE {timing_plot_path}")
    else:
        print(f"SKIP timing summary PNG (no rank events matched): {timing_plot_path}")
    print(f"WROTE DIR {out_dir}")
    print(f"exported_png_count={n_ok}, skipped_rows={n_skipped}")


if __name__ == "__main__":
    main()
