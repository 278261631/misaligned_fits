from pathlib import Path
import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.path import Path as MplPath
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch
from scipy.spatial import cKDTree

from alignment_common import build_matches, detect_stars, estimate_translation_from_stars, eval_poly


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank variable-star candidates from a FITS sequence using star-catalog matching."
    )
    parser.add_argument("--base", type=Path, default=Path("."), help="Base directory for relative paths.")
    parser.add_argument(
        "--pattern",
        nargs="+",
        default=["*.fits", "*.fit", "*.FITS", "*.FIT"],
        help="Glob patterns used to find FITS files.",
    )
    parser.add_argument("--ref", type=Path, default=None, help="Reference FITS path or filename (default: first match).")
    parser.add_argument("--out-csv", type=Path, default=None, help="Output ranking CSV path.")
    parser.add_argument(
        "--out-csv-nonref",
        type=Path,
        default=None,
        help="Output CSV for stars detected in non-reference frames only.",
    )
    parser.add_argument(
        "--out-csv-ref-missing",
        type=Path,
        default=None,
        help="Output CSV for stars detected in reference but missing in all used target frames.",
    )
    parser.add_argument(
        "--out-csv-nonref-inner-border",
        type=Path,
        default=None,
        help="Output CSV for non-reference-only stars inside ref-target overlap border only.",
    )
    parser.add_argument(
        "--out-overlap-expr",
        type=Path,
        default=None,
        help="Output JSON with ref-target overlap polygon expressions (per used frame).",
    )
    parser.add_argument(
        "--out-overlap-expr-png",
        type=Path,
        default=None,
        help="Optional PNG path to visualize final overlap polygons.",
    )
    parser.add_argument("--out-png", type=Path, default=None, help="Output candidate scatter PNG path.")
    parser.add_argument("--max-stars", type=int, default=5000, help="Maximum stars detected per frame.")
    parser.add_argument("--match-radius", type=float, default=24.0, help="Star matching radius in pixels.")
    parser.add_argument(
        "--ref-stars-all",
        type=Path,
        default=None,
        help="Reference all-stars NPZ (xy, flux). If set, enables NPZ mode with --target-stars-all/--target-align.",
    )
    parser.add_argument(
        "--target-stars-all",
        nargs="*",
        type=Path,
        default=None,
        help="Target all-stars NPZ list (xy, flux), one per target/alignment.",
    )
    parser.add_argument(
        "--target-align",
        nargs="*",
        type=Path,
        default=None,
        help="Alignment NPZ list, one per target stars NPZ.",
    )
    parser.add_argument(
        "--ref-image",
        type=Path,
        default=None,
        help="Optional image FITS path used only as plot background.",
    )
    parser.add_argument(
        "--ref-valid-region",
        type=Path,
        default=None,
        help="Optional effective-region JSON exported by export_fits_stars.py; points outside polygons are filtered.",
    )
    parser.add_argument("--min-observations", type=int, default=5, help="Minimum matched frames required per star.")
    parser.add_argument("--top-k", type=int, default=200, help="Top K candidates to highlight in the scatter plot.")
    parser.add_argument(
        "--mirror-vertical-png",
        dest="mirror_vertical_png",
        action="store_true",
        help="Vertically mirror output candidate PNG overlays (enabled by default).",
    )
    parser.add_argument(
        "--no-mirror-vertical-png",
        dest="mirror_vertical_png",
        action="store_false",
        help="Disable vertical mirroring for output candidate PNG overlays.",
    )
    parser.set_defaults(mirror_vertical_png=True)
    parser.add_argument(
        "--top-k-nonref",
        type=int,
        default=0,
        help="Top K non-reference-only stars to mark on plots (<=0 means show all).",
    )
    parser.add_argument(
        "--top-k-ref-missing",
        type=int,
        default=0,
        help="Top K reference-only missing stars to mark on plots (<=0 means show all).",
    )
    parser.add_argument(
        "--top-k-nonref-inner-border-csv",
        type=int,
        default=400,
        help="Top K rows to write into nonref inner-border CSV (<=0 means no limit).",
    )
    return parser.parse_args()


def resolve_path(base: Path | None, maybe_path: Path | None):
    if maybe_path is None:
        return None
    if maybe_path.is_absolute():
        return maybe_path
    if base is None:
        return maybe_path
    return base / maybe_path


def list_inputs(base: Path, patterns):
    files = []
    seen = set()
    for pattern in patterns:
        for p in sorted(base.glob(pattern)):
            if p.is_file() and p not in seen:
                seen.add(p)
                files.append(p)
    return files


def robust_mad(x):
    med = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - med))


def infer_frame_size_from_xy(xy):
    if len(xy) == 0:
        return 1, 1
    w = max(int(np.ceil(np.nanmax(xy[:, 0]) + 1.0)), 1)
    h = max(int(np.ceil(np.nanmax(xy[:, 1]) + 1.0)), 1)
    return h, w


def compute_overlap_rect_xy_bounds(w_ref, h_ref, w_tgt, h_tgt, dx0, dy0):
    ref_x_min, ref_x_max = -0.5, float(w_ref) - 0.5
    ref_y_min, ref_y_max = -0.5, float(h_ref) - 0.5
    tgt_x_min, tgt_x_max = float(dx0) - 0.5, float(dx0) + float(w_tgt) - 0.5
    tgt_y_min, tgt_y_max = float(dy0) - 0.5, float(dy0) + float(h_tgt) - 0.5
    x_min = max(ref_x_min, tgt_x_min)
    x_max = min(ref_x_max, tgt_x_max)
    y_min = max(ref_y_min, tgt_y_min)
    y_max = min(ref_y_max, tgt_y_max)
    if x_min > x_max or y_min > y_max:
        return None
    return x_min, x_max, y_min, y_max


def point_in_overlap_rect(x, y, rect):
    if rect is None:
        return False
    x_min, x_max, y_min, y_max = rect
    return (x >= x_min) and (x <= x_max) and (y >= y_min) and (y <= y_max)


def _polygon_area_abs(poly_xy):
    if poly_xy.shape[0] < 3:
        return 0.0
    x = poly_xy[:, 0]
    y = poly_xy[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _close_ring(poly_xy):
    if poly_xy.shape[0] == 0:
        return poly_xy
    if np.allclose(poly_xy[0], poly_xy[-1]):
        return poly_xy
    return np.vstack([poly_xy, poly_xy[0]])


def load_valid_region_polygons(path: Path):
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    polys = payload.get("polygons_xy", [])
    if not isinstance(polys, list) or len(polys) == 0:
        raise RuntimeError(f"Invalid valid-region JSON (polygons_xy missing/empty): {path}")
    polygons = []
    for poly in polys:
        arr = np.asarray(poly, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
            continue
        arr = _close_ring(arr)
        if _polygon_area_abs(arr[:-1]) <= 0.0:
            continue
        polygons.append(arr[:-1])
    if len(polygons) == 0:
        raise RuntimeError(f"Invalid valid-region JSON (no usable polygons): {path}")
    return polygons


def polygons_to_paths(polygons_xy):
    paths = []
    for poly in polygons_xy:
        arr = np.asarray(poly, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
            continue
        paths.append(MplPath(_close_ring(arr)))
    return paths


def point_in_any_region(x, y, region_paths):
    return any(p.contains_point((float(x), float(y)), radius=1e-9) for p in region_paths)


def _clip_poly_halfspace(poly, keep_fn, intersect_fn):
    if len(poly) == 0:
        return np.empty((0, 2), dtype=np.float64)
    out = []
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        ina = keep_fn(a)
        inb = keep_fn(b)
        if ina and inb:
            out.append(b)
        elif ina and not inb:
            out.append(intersect_fn(a, b))
        elif (not ina) and inb:
            out.append(intersect_fn(a, b))
            out.append(b)
    if len(out) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def clip_polygon_with_rect(poly, rect):
    x_min, x_max, y_min, y_max = rect
    out = np.asarray(poly, dtype=np.float64)
    if len(out) == 0:
        return out

    def inter_vertical(a, b, x0):
        dx = b[0] - a[0]
        if abs(dx) < 1e-12:
            return np.array([x0, a[1]], dtype=np.float64)
        t = (x0 - a[0]) / dx
        return np.array([x0, a[1] + t * (b[1] - a[1])], dtype=np.float64)

    def inter_horizontal(a, b, y0):
        dy = b[1] - a[1]
        if abs(dy) < 1e-12:
            return np.array([a[0], y0], dtype=np.float64)
        t = (y0 - a[1]) / dy
        return np.array([a[0] + t * (b[0] - a[0]), y0], dtype=np.float64)

    out = _clip_poly_halfspace(out, lambda p: p[0] >= x_min - 1e-12, lambda a, b: inter_vertical(a, b, x_min))
    out = _clip_poly_halfspace(out, lambda p: p[0] <= x_max + 1e-12, lambda a, b: inter_vertical(a, b, x_max))
    out = _clip_poly_halfspace(out, lambda p: p[1] >= y_min - 1e-12, lambda a, b: inter_horizontal(a, b, y_min))
    out = _clip_poly_halfspace(out, lambda p: p[1] <= y_max + 1e-12, lambda a, b: inter_horizontal(a, b, y_max))
    if len(out) < 3:
        return np.empty((0, 2), dtype=np.float64)
    return out


def save_overlap_expr_png(overlap_by_frame, out_png: Path, h_ref, w_ref, mirror_vertical=False):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.5, float(w_ref) - 0.5)
    if mirror_vertical:
        ax.set_ylim(float(h_ref) - 0.5, -0.5)
    else:
        ax.set_ylim(-0.5, float(h_ref) - 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#111111")
    cmap = plt.get_cmap("tab20")
    for k, frame_name in enumerate(sorted(overlap_by_frame.keys())):
        polys = overlap_by_frame[frame_name]
        if len(polys) == 0:
            continue
        color = cmap(k % 20)
        for j, poly in enumerate(polys):
            ring = _close_ring(poly)
            ax.plot(ring[:, 0], ring[:, 1], color=color, linewidth=1.2, alpha=0.9, label=frame_name if j == 0 else None)
    ax.set_title("Final overlap polygons (effective ∩ ref-target overlap)")
    ax.set_axis_off()
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0 and len(handles) <= 20:
        ax.legend(loc="upper right", framealpha=0.7, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def load_stars_npz(path: Path, return_meta=False):
    dat = np.load(path, allow_pickle=True)
    if "xy" not in dat or "flux" not in dat:
        raise RuntimeError(f"Invalid stars NPZ (need xy/flux): {path}")
    xy = np.asarray(dat["xy"], dtype=np.float64)
    flux = np.asarray(dat["flux"], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise RuntimeError(f"Invalid xy shape in {path}: {xy.shape}")
    if flux.ndim != 1 or len(flux) != len(xy):
        raise RuntimeError(f"Invalid flux shape in {path}: {flux.shape}, xy={xy.shape}")
    if return_meta:
        meta = {}
        for k in ("source_fits", "height", "width"):
            if k in dat:
                meta[k] = dat[k]
        return xy, flux, meta
    return xy, flux


def build_matches_from_alignment(xy_a, xy_b, cx, cy, fit_degree, match_radius):
    px, py = eval_poly(xy_a[:, 0], xy_a[:, 1], cx, cy, degree=fit_degree)
    pred = np.column_stack([px, py])
    tree = cKDTree(xy_b)
    d, bi = tree.query(pred, distance_upper_bound=match_radius)
    good = np.isfinite(d) & (bi < len(xy_b))
    ai_good = np.where(good)[0]
    best_by_b = {}
    for ai_src, bj, dist in zip(ai_good, bi[good], d[good]):
        if (bj not in best_by_b) or (dist < best_by_b[bj][1]):
            best_by_b[bj] = (int(ai_src), float(dist))
    if not best_by_b:
        return np.array([], dtype=int), np.array([], dtype=int)
    bi_idx = np.array(sorted(best_by_b.keys()), dtype=int)
    ai_idx = np.array([best_by_b[bj][0] for bj in bi_idx], dtype=int)
    return ai_idx, bi_idx


def save_candidate_scatter(
    ref_img, x, y, score, top_k, out_png: Path, nonref_xy=None, ref_missing_xy=None, mirror_vertical=True
):
    finite = np.isfinite(ref_img)
    fill = np.nanmedian(ref_img[finite]) if np.any(finite) else 0.0
    view = np.where(finite, ref_img, fill)

    norm = ImageNormalize(view, interval=PercentileInterval(99.5), stretch=SqrtStretch())
    order = np.argsort(score)[::-1]
    keep = order[: min(top_k, len(order))]

    plt.figure(figsize=(12, 8))
    plt.imshow(view, origin="lower", cmap="gray", norm=norm)
    sc = plt.scatter(
        x[keep],
        y[keep],
        c=score[keep],
        s=24,
        cmap="turbo",
        alpha=0.9,
        edgecolors="white",
        linewidths=0.3,
    )
    if nonref_xy is not None and len(nonref_xy) > 0:
        plt.scatter(
            nonref_xy[:, 0],
            nonref_xy[:, 1],
            marker="x",
            s=40,
            c="#00E5FF",
            alpha=0.95,
            linewidths=1.1,
            label="Non-reference-only stars",
        )
    if ref_missing_xy is not None and len(ref_missing_xy) > 0:
        plt.scatter(
            ref_missing_xy[:, 0],
            ref_missing_xy[:, 1],
            marker="^",
            s=34,
            c="#FF3EA5",
            alpha=0.9,
            linewidths=0.6,
            edgecolors="white",
            label="Reference-only missing stars",
        )
    if (nonref_xy is not None and len(nonref_xy) > 0) or (ref_missing_xy is not None and len(ref_missing_xy) > 0):
        plt.legend(loc="upper right", framealpha=0.7)
    plt.colorbar(sc, label="Variability score")
    plt.title(f"Top {len(keep)} variable candidates")
    if mirror_vertical:
        plt.gca().invert_yaxis()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_candidate_overlay_exact(
    ref_img,
    x,
    y,
    score,
    top_k,
    out_png: Path,
    nonref_xy=None,
    ref_missing_xy=None,
    mirror_vertical=True,
    annotate_rank=False,
    nonref_ranks=None,
    ref_missing_ranks=None,
):
    finite = np.isfinite(ref_img)
    fill = np.nanmedian(ref_img[finite]) if np.any(finite) else 0.0
    view = np.where(finite, ref_img, fill)

    h, w = view.shape
    dpi = 100
    order = np.argsort(score)[::-1]
    keep = order[: min(top_k, len(order))]

    # Export an exact-size overlay image aligned to A pixel grid.
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, frameon=False)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    norm = ImageNormalize(view, interval=PercentileInterval(99.5), stretch=SqrtStretch())
    ax.imshow(view, origin="lower", cmap="gray", norm=norm, interpolation="nearest")
    ax.scatter(
        x[keep],
        y[keep],
        c=score[keep],
        s=18,
        cmap="turbo",
        alpha=0.9,
        edgecolors="white",
        linewidths=0.25,
    )
    if annotate_rank and len(keep) > 0:
        score_keep = np.asarray(score[keep], dtype=np.float64)
        if np.nanmax(score_keep) > np.nanmin(score_keep):
            cmap_norm = plt.Normalize(vmin=float(np.nanmin(score_keep)), vmax=float(np.nanmax(score_keep)))
        else:
            cmap_norm = plt.Normalize(vmin=float(np.nanmin(score_keep)) - 1.0, vmax=float(np.nanmax(score_keep)) + 1.0)
        cmap = plt.get_cmap("turbo")
        for j, idx in enumerate(keep):
            rank_txt = str(j + 1)
            color = cmap(cmap_norm(float(score[idx])))
            ax.text(
                float(x[idx]) + 1.8,
                float(y[idx]) + 1.8,
                rank_txt,
                color=color,
                fontsize=6,
                ha="left",
                va="bottom",
                alpha=0.95,
            )
    if nonref_xy is not None and len(nonref_xy) > 0:
        ax.scatter(
            nonref_xy[:, 0],
            nonref_xy[:, 1],
            marker="x",
            s=32,
            c="#00E5FF",
            alpha=0.95,
            linewidths=1.0,
        )
        if annotate_rank:
            if nonref_ranks is None:
                nonref_ranks = np.arange(1, len(nonref_xy) + 1, dtype=int)
            for q, r in zip(nonref_xy, nonref_ranks):
                ax.text(
                    float(q[0]) + 1.8,
                    float(q[1]) + 1.8,
                    str(int(r)),
                    color="#00E5FF",
                    fontsize=6,
                    ha="left",
                    va="bottom",
                    alpha=0.95,
                )
    if ref_missing_xy is not None and len(ref_missing_xy) > 0:
        ax.scatter(
            ref_missing_xy[:, 0],
            ref_missing_xy[:, 1],
            marker="^",
            s=26,
            c="#FF3EA5",
            alpha=0.9,
            linewidths=0.5,
            edgecolors="white",
        )
        if annotate_rank:
            if ref_missing_ranks is None:
                ref_missing_ranks = np.arange(1, len(ref_missing_xy) + 1, dtype=int)
            for q, r in zip(ref_missing_xy, ref_missing_ranks):
                ax.text(
                    float(q[0]) + 1.8,
                    float(q[1]) + 1.8,
                    str(int(r)),
                    color="#FF3EA5",
                    fontsize=6,
                    ha="left",
                    va="bottom",
                    alpha=0.95,
                )
    ax.set_xlim(-0.5, w - 0.5)
    if mirror_vertical:
        ax.set_ylim(h - 0.5, -0.5)
    else:
        ax.set_ylim(-0.5, h - 0.5)
    ax.set_axis_off()
    fig.savefig(out_png, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def main():
    args = parse_args()
    base = args.base if args.base is not None else Path(".")
    npz_mode = (
        args.ref_stars_all is not None
        or (args.target_stars_all is not None and len(args.target_stars_all) > 0)
        or (args.target_align is not None and len(args.target_align) > 0)
    )

    if npz_mode:
        ref_path = resolve_path(base, args.ref) if args.ref is not None else None
        targets = []
    else:
        inputs = list_inputs(base, args.pattern)
        if len(inputs) < 2:
            raise RuntimeError(f"Need at least 2 FITS files in {base}.")
        ref_path = resolve_path(base, args.ref) if args.ref is not None else inputs[0]
        if ref_path not in inputs:
            inputs = [ref_path] + [p for p in inputs if p != ref_path]
        targets = [p for p in inputs if p != ref_path]
        if len(targets) == 0:
            raise RuntimeError("No target files found after selecting reference.")

    out_csv = args.out_csv if args.out_csv is not None else (base / "variable_candidates_rank.csv")
    out_csv_nonref = (
        args.out_csv_nonref if args.out_csv_nonref is not None else (base / "variable_candidates_nonref_only.csv")
    )
    out_csv_ref_missing = (
        args.out_csv_ref_missing
        if args.out_csv_ref_missing is not None
        else (base / "variable_candidates_ref_only_missing_in_targets.csv")
    )
    out_csv_nonref_inner_border = (
        args.out_csv_nonref_inner_border
        if args.out_csv_nonref_inner_border is not None
        else (base / "variable_candidates_nonref_only_inner_border.csv")
    )
    out_overlap_expr = (
        args.out_overlap_expr if args.out_overlap_expr is not None else (base / "ref_target_overlap_polygon_expr.json")
    )
    out_overlap_expr_png = args.out_overlap_expr_png if args.out_overlap_expr_png is not None else None
    out_png = args.out_png if args.out_png is not None else (base / "variable_candidates_rank.png")
    out_png_aligned = out_png.with_name(f"{out_png.stem}_aligned_to_a{out_png.suffix}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv_nonref.parent.mkdir(parents=True, exist_ok=True)
    out_csv_ref_missing.parent.mkdir(parents=True, exist_ok=True)
    out_csv_nonref_inner_border.parent.mkdir(parents=True, exist_ok=True)
    out_overlap_expr.parent.mkdir(parents=True, exist_ok=True)
    if out_overlap_expr_png is not None:
        out_overlap_expr_png.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    ref_data = None
    ref_image_path = resolve_path(base, args.ref_image) if args.ref_image is not None else None
    ref_valid_region_path = resolve_path(base, args.ref_valid_region) if args.ref_valid_region is not None else None
    ref_valid_region_polygons = None
    ref_valid_region_paths = None
    if ref_valid_region_path is not None:
        if not ref_valid_region_path.exists():
            raise RuntimeError(f"Reference valid-region JSON not found: {ref_valid_region_path}")
        ref_valid_region_polygons = load_valid_region_polygons(ref_valid_region_path)
        ref_valid_region_paths = polygons_to_paths(ref_valid_region_polygons)
    if ref_image_path is not None:
        if not ref_image_path.exists():
            raise RuntimeError(f"Reference image not found: {ref_image_path}")
        ref_data = fits.getdata(ref_image_path).astype(float)
    if npz_mode:
        if args.ref_stars_all is None or args.target_stars_all is None or args.target_align is None:
            raise RuntimeError("NPZ mode requires --ref-stars-all, --target-stars-all and --target-align together.")
        if len(args.target_stars_all) == 0:
            raise RuntimeError("NPZ mode requires at least one --target-stars-all entry.")
        if len(args.target_stars_all) != len(args.target_align):
            raise RuntimeError(
                "In NPZ mode, --target-stars-all and --target-align must have the same number of entries."
            )
        ref_stars_all = resolve_path(base, args.ref_stars_all)
        xy_ref, flux_ref, ref_meta = load_stars_npz(ref_stars_all, return_meta=True)
        if len(xy_ref) == 0:
            raise RuntimeError(f"No stars in reference stars NPZ: {ref_stars_all}")
        if ref_data is None:
            if "height" in ref_meta and "width" in ref_meta:
                h = int(np.asarray(ref_meta["height"]).ravel()[0])
                w = int(np.asarray(ref_meta["width"]).ravel()[0])
            else:
                x_max = int(np.ceil(np.nanmax(xy_ref[:, 0]) + 1.0)) if len(xy_ref) > 0 else 1
                y_max = int(np.ceil(np.nanmax(xy_ref[:, 1]) + 1.0)) if len(xy_ref) > 0 else 1
                w = max(x_max, 1)
                h = max(y_max, 1)
            ref_data = np.full((h, w), 0.5, dtype=np.float32)
    else:
        if ref_path is None or not ref_path.exists():
            raise RuntimeError(f"Reference FITS not found: {ref_path}")
        ref_detect_data = fits.getdata(ref_path).astype(float)
        xy_ref, flux_ref = detect_stars(ref_detect_data, max_stars=int(args.max_stars))
        if len(xy_ref) == 0:
            raise RuntimeError(f"No stars detected in reference: {ref_path}")
        if ref_data is None:
            ref_data = np.full(ref_detect_data.shape, 0.5, dtype=np.float32)

    n_ref = len(xy_ref)
    # First column uses reference flux as baseline.
    measurements = [np.asarray(flux_ref, dtype=np.float64)]
    used_files = [ref_path if ref_path is not None else resolve_path(base, args.ref_stars_all)]
    matched_counts = []
    failed_files = []
    h_ref, w_ref = ref_data.shape
    nonref_xy = []
    nonref_flux_samples = []
    nonref_n_detections = []
    nonref_frame_sets = []
    overlap_rect_by_frame = {}

    target_entries = []
    if npz_mode:
        for stars_p, align_p in zip(args.target_stars_all, args.target_align):
            target_entries.append((resolve_path(base, stars_p), resolve_path(base, align_p), None))
    else:
        for p in targets:
            target_entries.append((None, None, p))

    for stars_npz_path, align_npz_path, p in target_entries:
        try:
            if npz_mode:
                xy_b, flux_b, tgt_meta = load_stars_npz(stars_npz_path, return_meta=True)
                if len(xy_b) == 0:
                    raise RuntimeError("No stars detected in target stars NPZ.")
                if "height" in tgt_meta and "width" in tgt_meta:
                    h_b = int(np.asarray(tgt_meta["height"]).ravel()[0])
                    w_b = int(np.asarray(tgt_meta["width"]).ravel()[0])
                else:
                    h_b, w_b = infer_frame_size_from_xy(xy_b)
                sol = np.load(align_npz_path, allow_pickle=True)
                if "cx" not in sol or "cy" not in sol or "fit_degree" not in sol:
                    raise RuntimeError(f"Invalid align NPZ (need cx/cy/fit_degree): {align_npz_path}")
                cx = np.asarray(sol["cx"], dtype=float)
                cy = np.asarray(sol["cy"], dtype=float)
                fit_degree = int(np.asarray(sol["fit_degree"]).ravel()[0])
                ai_idx, bi_idx = build_matches_from_alignment(
                    xy_ref, xy_b, cx, cy, fit_degree=fit_degree, match_radius=float(args.match_radius)
                )
                if "dx0" in sol and "dy0" in sol:
                    dx0 = float(np.asarray(sol["dx0"]).ravel()[0])
                    dy0 = float(np.asarray(sol["dy0"]).ravel()[0])
                else:
                    dx0 = float(np.nanmedian(xy_ref[ai_idx, 0] - xy_b[bi_idx, 0])) if len(ai_idx) > 0 else 0.0
                    dy0 = float(np.nanmedian(xy_ref[ai_idx, 1] - xy_b[bi_idx, 1])) if len(ai_idx) > 0 else 0.0
                frame_label = stars_npz_path.name
            else:
                data = fits.getdata(p).astype(float)
                h_b, w_b = data.shape
                xy_b, flux_b = detect_stars(data, max_stars=int(args.max_stars))
                if len(xy_b) == 0:
                    raise RuntimeError("No stars detected.")

                dx0, dy0 = estimate_translation_from_stars(xy_ref, xy_b, top_n=300, bin_size=2.0)
                ai_idx, bi_idx = build_matches(xy_ref, xy_b, dx0, dy0, match_radius=float(args.match_radius))
                frame_label = p.name
            if len(ai_idx) < 10:
                raise RuntimeError(f"Too few matches ({len(ai_idx)}).")

            # Normalize frame-to-frame transparency/exposure scale with robust median ratio.
            ratio = np.asarray(flux_b[bi_idx], dtype=np.float64) / np.maximum(
                np.asarray(flux_ref[ai_idx], dtype=np.float64), 1e-12
            )
            scale = float(np.nanmedian(ratio))
            if not np.isfinite(scale) or scale <= 0.0:
                scale = 1.0

            vals = np.full(n_ref, np.nan, dtype=np.float64)
            vals[ai_idx] = np.asarray(flux_b[bi_idx], dtype=np.float64) / scale
            measurements.append(vals)
            used_files.append(stars_npz_path if npz_mode else p)
            matched_counts.append((frame_label, len(ai_idx)))
            overlap_rect_by_frame[frame_label] = compute_overlap_rect_xy_bounds(w_ref, h_ref, w_b, h_b, dx0, dy0)

            # Collect stars only detected in non-reference frames.
            unmatched = np.ones(len(xy_b), dtype=bool)
            unmatched[bi_idx] = False
            if np.any(unmatched):
                extra_xy = xy_b[unmatched] + np.array([dx0, dy0], dtype=float)
                extra_flux = np.asarray(flux_b[unmatched], dtype=np.float64) / scale
                in_bounds = (
                    (extra_xy[:, 0] >= -0.5)
                    & (extra_xy[:, 0] <= (w_ref - 0.5))
                    & (extra_xy[:, 1] >= -0.5)
                    & (extra_xy[:, 1] <= (h_ref - 0.5))
                    & np.isfinite(extra_flux)
                    & (extra_flux > 0.0)
                )
                extra_xy = extra_xy[in_bounds]
                extra_flux = extra_flux[in_bounds]
                if len(extra_xy) > 0:
                    merge_r = max(float(args.match_radius) * 0.6, 2.0)
                    if len(nonref_xy) > 0:
                        tree = cKDTree(np.asarray(nonref_xy, dtype=np.float64))
                        d, idx = tree.query(extra_xy, distance_upper_bound=merge_r)
                    else:
                        d = np.full(len(extra_xy), np.inf, dtype=np.float64)
                        idx = np.full(len(extra_xy), -1, dtype=int)

                    for q, fq, dq, jq in zip(extra_xy, extra_flux, d, idx):
                        if np.isfinite(dq) and int(jq) < len(nonref_xy):
                            j = int(jq)
                            n_old = nonref_n_detections[j]
                            nonref_xy[j] = (np.asarray(nonref_xy[j], dtype=np.float64) * n_old + q) / (n_old + 1)
                            nonref_n_detections[j] = n_old + 1
                            nonref_flux_samples[j].append(float(fq))
                            nonref_frame_sets[j].add(frame_label)
                        else:
                            nonref_xy.append(np.asarray(q, dtype=np.float64))
                            nonref_n_detections.append(1)
                            nonref_flux_samples.append([float(fq)])
                            nonref_frame_sets.append({frame_label})
        except Exception as exc:
            label = stars_npz_path.name if npz_mode else p.name
            failed_files.append((label, str(exc)))

    if ref_valid_region_polygons is not None:
        source_region_polygons = ref_valid_region_polygons
    else:
        source_region_polygons = [
            np.array(
                [
                    [-0.5, -0.5],
                    [float(w_ref) - 0.5, -0.5],
                    [float(w_ref) - 0.5, float(h_ref) - 0.5],
                    [-0.5, float(h_ref) - 0.5],
                ],
                dtype=np.float64,
            )
        ]
    final_overlap_polygons_by_frame = {}
    final_overlap_paths_by_frame = {}
    for frame_name, rect in overlap_rect_by_frame.items():
        frame_polys = []
        if rect is not None:
            for poly in source_region_polygons:
                clipped = clip_polygon_with_rect(poly, rect)
                if len(clipped) < 3:
                    continue
                if _polygon_area_abs(clipped) <= 1e-9:
                    continue
                frame_polys.append(clipped)
        final_overlap_polygons_by_frame[frame_name] = frame_polys
        final_overlap_paths_by_frame[frame_name] = polygons_to_paths(frame_polys)

    flux_mat = np.vstack(measurements).T  # [n_ref, n_frames_used]
    n_obs = np.sum(np.isfinite(flux_mat), axis=1)
    med_flux = np.nanmedian(flux_mat, axis=1)
    ref_region_ok = np.ones(n_ref, dtype=bool)
    if ref_valid_region_paths is not None:
        ref_region_ok = np.array(
            [point_in_any_region(xy_ref[i, 0], xy_ref[i, 1], ref_valid_region_paths) for i in range(n_ref)],
            dtype=bool,
        )

    valid = (n_obs >= int(args.min_observations)) & np.isfinite(med_flux) & (med_flux > 0.0) & ref_region_ok
    if np.count_nonzero(valid) == 0:
        raise RuntimeError("No stars meet min observation requirement.")

    rel = flux_mat / np.maximum(med_flux[:, None], 1e-12)
    mad_rel = np.array([robust_mad(row[np.isfinite(row)]) for row in rel], dtype=np.float64)
    p95 = np.nanpercentile(rel, 95, axis=1)
    p05 = np.nanpercentile(rel, 5, axis=1)
    amp_rel = p95 - p05

    # Composite variability score. Higher means more variable-like.
    score = mad_rel + 0.5 * amp_rel
    score[~valid] = np.nan

    idx = np.where(valid)[0]
    order = idx[np.argsort(score[idx])[::-1]]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "x",
                "y",
                "variability_score",
                "mad_rel",
                "amp_rel_p95_p05",
                "n_observations",
                "median_flux_norm",
            ]
        )
        for r, i in enumerate(order, start=1):
            writer.writerow(
                [
                    r,
                    f"{xy_ref[i, 0]:.4f}",
                    f"{xy_ref[i, 1]:.4f}",
                    f"{score[i]:.8f}",
                    f"{mad_rel[i]:.8f}",
                    f"{amp_rel[i]:.8f}",
                    int(n_obs[i]),
                    f"{med_flux[i]:.8f}",
                ]
            )

    nonref_plot_xy = np.empty((0, 2), dtype=np.float64)
    nonref_plot_xy_rank = np.empty((0, 2), dtype=np.float64)
    nonref_plot_xy_inner_all = np.empty((0, 2), dtype=np.float64)
    nonref_plot_ranks_rank = np.empty((0,), dtype=np.int32)
    nonref_count = len(nonref_xy)
    if nonref_count > 0:
        nonref_xy_arr = np.asarray(nonref_xy, dtype=np.float64)
        nonref_n_frames = np.asarray([len(s) for s in nonref_frame_sets], dtype=np.int32)
        nonref_median_flux = np.asarray([float(np.median(v)) for v in nonref_flux_samples], dtype=np.float64)
        nonref_n_det = np.asarray(nonref_n_detections, dtype=np.int32)
        # Primary key: median flux (desc), tie-breakers: detections (desc), frames (desc).
        nonref_order = np.lexsort((-nonref_n_frames, -nonref_n_det, -nonref_median_flux))
        nonref_inside = np.zeros(nonref_count, dtype=bool)
        for i in range(nonref_count):
            xi = float(nonref_xy_arr[i, 0])
            yi = float(nonref_xy_arr[i, 1])
            inside_overlap = any(
                point_in_any_region(xi, yi, final_overlap_paths_by_frame.get(frame_name, []))
                for frame_name in nonref_frame_sets[i]
            )
            nonref_inside[i] = inside_overlap

        with out_csv_nonref.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "rank",
                    "x",
                    "y",
                    "n_frames_detected",
                    "n_detections",
                    "median_flux_norm",
                    "frames",
                ]
            )
            for r, i in enumerate(nonref_order, start=1):
                frame_names = ";".join(sorted(nonref_frame_sets[i]))
                writer.writerow(
                    [
                        r,
                        f"{nonref_xy_arr[i, 0]:.4f}",
                        f"{nonref_xy_arr[i, 1]:.4f}",
                        int(nonref_n_frames[i]),
                        int(nonref_n_det[i]),
                        f"{nonref_median_flux[i]:.8f}",
                        frame_names,
                    ]
                )

        with out_csv_nonref_inner_border.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "rank",
                    "x",
                    "y",
                    "n_frames_detected",
                    "n_detections",
                    "median_flux_norm",
                    "frames",
                ]
            )
            max_inner_csv = int(args.top_k_nonref_inner_border_csv)
            rank_inner = 1
            for i in nonref_order:
                if not nonref_inside[i]:
                    continue
                if max_inner_csv > 0 and rank_inner > max_inner_csv:
                    break
                frame_names = ";".join(sorted(nonref_frame_sets[i]))
                writer.writerow(
                    [
                        rank_inner,
                        f"{nonref_xy_arr[i, 0]:.4f}",
                        f"{nonref_xy_arr[i, 1]:.4f}",
                        int(nonref_n_frames[i]),
                        int(nonref_n_det[i]),
                        f"{nonref_median_flux[i]:.8f}",
                        frame_names,
                    ]
                )
                rank_inner += 1

        top_k_nonref = int(args.top_k_nonref)
        nonref_order_inside = nonref_order[nonref_inside[nonref_order]]
        nonref_plot_xy_inner_all = nonref_xy_arr[nonref_order_inside]
        keep_n = len(nonref_order_inside) if top_k_nonref <= 0 else min(top_k_nonref, len(nonref_order_inside))
        nonref_plot_xy = nonref_xy_arr[nonref_order_inside[:keep_n]]
        keep_n_rank = min(400, len(nonref_order_inside))
        nonref_plot_xy_rank = nonref_xy_arr[nonref_order_inside[:keep_n_rank]]
        nonref_plot_ranks_rank = np.arange(1, keep_n_rank + 1, dtype=np.int32)
    else:
        with out_csv_nonref.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "rank",
                    "x",
                    "y",
                    "n_frames_detected",
                    "n_detections",
                    "median_flux_norm",
                    "frames",
                ]
            )
        with out_csv_nonref_inner_border.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "rank",
                    "x",
                    "y",
                    "n_frames_detected",
                    "n_detections",
                    "median_flux_norm",
                    "frames",
                ]
            )

    # Stars detected in reference but never matched in any successfully used target frame.
    n_used_targets = max(len(used_files) - 1, 0)
    n_target_obs = np.maximum(n_obs - 1, 0).astype(np.int32)
    ref_missing_mask = (n_target_obs == 0) & ref_region_ok
    ref_missing_idx = np.where(ref_missing_mask)[0]
    ref_missing_plot_xy = np.empty((0, 2), dtype=np.float64)
    ref_missing_plot_xy_rank = np.empty((0, 2), dtype=np.float64)
    ref_missing_plot_ranks_rank = np.empty((0,), dtype=np.int32)
    if len(ref_missing_idx) > 0:
        ref_only_flux = np.asarray(flux_ref[ref_missing_idx], dtype=np.float64)
        ref_missing_order = ref_missing_idx[np.argsort(ref_only_flux)[::-1]]
    else:
        ref_missing_order = ref_missing_idx

    with out_csv_ref_missing.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "x",
                "y",
                "reference_flux",
                "n_target_observations",
                "n_used_target_frames",
                "missing_in_all_used_targets",
            ]
        )
        for r, i in enumerate(ref_missing_order, start=1):
            writer.writerow(
                [
                    r,
                    f"{xy_ref[i, 0]:.4f}",
                    f"{xy_ref[i, 1]:.4f}",
                    f"{float(flux_ref[i]):.8f}",
                    int(n_target_obs[i]),
                    int(n_used_targets),
                    1,
                ]
            )
    if len(ref_missing_order) > 0:
        top_k_ref_missing = int(args.top_k_ref_missing)
        keep_m = len(ref_missing_order) if top_k_ref_missing <= 0 else min(top_k_ref_missing, len(ref_missing_order))
        ref_missing_plot_xy = xy_ref[ref_missing_order[:keep_m], :]
        keep_m_rank = min(200, len(ref_missing_order))
        ref_missing_plot_xy_rank = xy_ref[ref_missing_order[:keep_m_rank], :]
        ref_missing_plot_ranks_rank = np.arange(1, keep_m_rank + 1, dtype=np.int32)

    overlap_payload = {
        "coordinate_system": "reference_image_xy",
        "note": "Final geometric intersection polygons: (reference effective region) ∩ (ref-target overlap by dx0/dy0).",
        "frames": [],
    }
    for frame_name in sorted(final_overlap_polygons_by_frame.keys()):
        polys = final_overlap_polygons_by_frame[frame_name]
        if len(polys) == 0:
            overlap_payload["frames"].append({"frame": frame_name, "has_overlap": False})
            continue
        overlap_payload["frames"].append(
            {
                "frame": frame_name,
                "has_overlap": True,
                "polygon_count": int(len(polys)),
                "polygons_xy": [_close_ring(p).tolist() for p in polys],
                "areas": [float(_polygon_area_abs(p)) for p in polys],
            }
        )
    with out_overlap_expr.open("w", encoding="utf-8") as f:
        json.dump(overlap_payload, f, ensure_ascii=False, indent=2)
    if out_overlap_expr_png is not None:
        save_overlap_expr_png(
            final_overlap_polygons_by_frame,
            out_overlap_expr_png,
            h_ref=h_ref,
            w_ref=w_ref,
            mirror_vertical=bool(args.mirror_vertical_png),
        )

    save_candidate_scatter(
        ref_data,
        xy_ref[:, 0],
        xy_ref[:, 1],
        np.nan_to_num(score, nan=-1.0),
        400,
        out_png,
        nonref_xy=nonref_plot_xy_rank,
        ref_missing_xy=ref_missing_plot_xy_rank,
        mirror_vertical=bool(args.mirror_vertical_png),
    )
    save_candidate_overlay_exact(
        ref_data,
        xy_ref[:, 0],
        xy_ref[:, 1],
        np.nan_to_num(score, nan=-1.0),
        400,
        out_png_aligned,
        nonref_xy=nonref_plot_xy_rank,
        ref_missing_xy=ref_missing_plot_xy_rank,
        mirror_vertical=bool(args.mirror_vertical_png),
        annotate_rank=True,
        nonref_ranks=nonref_plot_ranks_rank,
        ref_missing_ranks=ref_missing_plot_ranks_rank,
    )

    print(f"Reference: {ref_path}")
    if ref_image_path is not None:
        print(f"Reference image: {ref_image_path}")
    if ref_valid_region_path is not None:
        print(f"Reference valid region: {ref_valid_region_path}")
    if npz_mode:
        print(f"Reference stars NPZ: {resolve_path(base, args.ref_stars_all)}")
    print(f"Frames used: {len(used_files)}")
    print(f"Stars in reference: {n_ref}")
    print(f"Candidates ranked: {len(order)}")
    print(f"WROTE {out_csv}")
    print(f"WROTE {out_csv_nonref}")
    print(f"WROTE {out_csv_nonref_inner_border}")
    print(f"WROTE {out_csv_ref_missing}")
    print(f"WROTE {out_overlap_expr}")
    if out_overlap_expr_png is not None:
        print(f"WROTE {out_overlap_expr_png}")
    print(f"WROTE {out_png}")
    print(f"WROTE {out_png_aligned}")
    print(f"Non-reference-only stars: {nonref_count}")
    print(f"Reference-only (missing in all used targets): {len(ref_missing_order)}")

    if matched_counts:
        print("Match counts per frame:")
        for name, cnt in matched_counts:
            print(f"  {name}: {cnt}")
    if failed_files:
        print("Failed frames:")
        for name, reason in failed_files:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
