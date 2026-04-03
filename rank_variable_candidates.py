from pathlib import Path
import argparse
import csv
import json
import time

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib.path import Path as MplPath
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch
from scipy.spatial import cKDTree

from alignment_common import build_matches, detect_stars, estimate_translation_from_stars, eval_poly
from timing_logger import TimingLogger, run_script_with_timing


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
        "--out-csv-nonref-inner-border-pre-knee",
        type=Path,
        default=None,
        help="Output CSV for nonref inner-border stars before flux-threshold/top-k truncation (still excludes ref-nearby points).",
    )
    parser.add_argument(
        "--csv-detail",
        "--csv_detail",
        dest="csv_detail",
        action="store_true",
        help=(
            "Enable detailed CSV outputs. When disabled (default), skip writing "
            "nonref_only and nonref_inner_border_pre_knee CSV files."
        ),
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
    parser.add_argument(
        "--nonref-ref-check-radius",
        type=float,
        default=10.0,
        help="Radius in pixels to check whether a nonref point already exists in ref-stars-all (<=0 disables).",
    )
    parser.add_argument(
        "--timing-jsonl",
        type=Path,
        default=None,
        help="Optional timing JSONL path (default: output/timing.jsonl or MISALIGNED_FITS_TIMING_PATH).",
    )
    parser.add_argument(
        "--timing-run-id",
        type=str,
        default=None,
        help="Optional run_id used when writing timing events (default: MISALIGNED_FITS_RUN_ID or auto-generated).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs; default skips when all expected outputs already exist.",
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


def detect_knee_keep_count_desc(values_desc, min_count=8):
    arr = np.asarray(values_desc, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return 0
    if n < int(min_count):
        return n
    x = np.arange(n, dtype=np.float64)

    def sse_line(xv, yv):
        if len(xv) <= 1:
            return 0.0
        p = np.polyfit(xv, yv, 1)
        pred = p[0] * xv + p[1]
        res = yv - pred
        return float(np.dot(res, res))

    best_k = 0
    best_sse = float("inf")
    # Split into [0..k] and [k..n-1], both segments need >=2 points.
    for k in range(1, n - 1):
        x1 = x[: k + 1]
        y1 = arr[: k + 1]
        x2 = x[k:]
        y2 = arr[k:]
        if len(x1) < 2 or len(x2) < 2:
            continue
        sse = sse_line(x1, y1) + sse_line(x2, y2)
        if sse < best_sse:
            best_sse = sse
            best_k = k
    keep_n = best_k + 1
    return max(1, min(keep_n, n))


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


def _meta_scalar_to_text(v):
    if v is None:
        return None
    arr = np.asarray(v)
    if arr.size == 0:
        return None
    s = arr.ravel()[0]
    if isinstance(s, (bytes, bytearray, np.bytes_)):
        try:
            s = s.decode("utf-8")
        except Exception:
            s = s.decode(errors="ignore")
    text = str(s).strip()
    return text if len(text) > 0 else None


def resolve_reference_celestial_wcs(base: Path, ref_path: Path | None, ref_stars_all_path: Path | None):
    candidates = []
    if ref_path is not None:
        candidates.append(ref_path)
    if ref_stars_all_path is not None and ref_stars_all_path.exists():
        try:
            dat = np.load(ref_stars_all_path, allow_pickle=True)
            for k in ("source_fits", "reference_fits", "projected_fits"):
                if k not in dat:
                    continue
                text = _meta_scalar_to_text(dat[k])
                if text is None:
                    continue
                p = Path(text)
                if not p.is_absolute():
                    p = resolve_path(base, p)
                candidates.append(p)
        except Exception:
            pass
    seen = set()
    for p in candidates:
        if p is None:
            continue
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if not p.exists():
            continue
        try:
            wcs = WCS(fits.getheader(p)).celestial
            return wcs, p
        except Exception:
            continue
    return None, None


def resolve_reference_mjd(base: Path, ref_path: Path | None, ref_stars_all_path: Path | None, preferred_path: Path | None = None):
    candidates = []
    if preferred_path is not None:
        candidates.append(preferred_path)
    if ref_path is not None:
        candidates.append(ref_path)
    if ref_stars_all_path is not None and ref_stars_all_path.exists():
        try:
            dat = np.load(ref_stars_all_path, allow_pickle=True)
            for k in ("source_fits", "reference_fits", "projected_fits"):
                if k not in dat:
                    continue
                text = _meta_scalar_to_text(dat[k])
                if text is None:
                    continue
                p = Path(text)
                if not p.is_absolute():
                    p = resolve_path(base, p)
                candidates.append(p)
        except Exception:
            pass
    seen = set()
    mjd_keys = ("MJD-OBS", "MJD_OBS", "MJD", "MJD-AVG", "MJDAVG", "MJDSTART", "MJDEND")
    for p in candidates:
        if p is None:
            continue
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if not p.exists():
            continue
        try:
            hdr = fits.getheader(p)
        except Exception:
            continue
        for k in mjd_keys:
            if k in hdr:
                try:
                    return float(hdr[k]), p
                except Exception:
                    pass
        date_obs = hdr.get("DATE-OBS")
        if date_obs is not None:
            try:
                return float(Time(str(date_obs)).mjd), p
            except Exception:
                pass
    return float("nan"), None


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
    nonref_has_ref_nearby_mask=None,
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
        if nonref_has_ref_nearby_mask is None or len(nonref_has_ref_nearby_mask) != len(nonref_xy):
            near_mask = np.zeros(len(nonref_xy), dtype=bool)
        else:
            near_mask = np.asarray(nonref_has_ref_nearby_mask, dtype=bool)
        no_ref_nearby_mask = ~near_mask
        if np.any(near_mask):
            xy_near = nonref_xy[near_mask]
            ax.scatter(
                xy_near[:, 0],
                xy_near[:, 1],
                marker="x",
                s=32,
                c="#00E5FF",
                alpha=0.95,
                linewidths=1.0,
            )
        if np.any(no_ref_nearby_mask):
            xy_no_ref = nonref_xy[no_ref_nearby_mask]
            ax.scatter(
                xy_no_ref[:, 0],
                xy_no_ref[:, 1],
                marker="o",
                s=70,
                facecolors="none",
                edgecolors="#FFD400",
                linewidths=1.1,
                alpha=0.98,
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
    csv_detail = bool(args.csv_detail)
    out_csv_nonref_inner_border_for_timing = (
        args.out_csv_nonref_inner_border
        if args.out_csv_nonref_inner_border is not None
        else (base / "variable_candidates_nonref_only_inner_border.csv")
    )
    timing_path = (
        resolve_path(base, args.timing_jsonl)
        if args.timing_jsonl is not None
        else (out_csv_nonref_inner_border_for_timing.parent / "timing.jsonl")
    )
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    logger = TimingLogger(
        script=Path(__file__).name,
        timing_path=timing_path,
        run_id=args.timing_run_id,
    )

    def log_stage(step, t0, meta=None, status="ok"):
        t1 = time.perf_counter()
        logger.write_event(
            step,
            (t1 - t0) * 1000.0,
            status=status,
            meta=meta,
            start_perf=t0,
            end_perf=t1,
        )
        return t1

    phase_t0 = time.perf_counter()
    ref_stars_all_path = resolve_path(base, args.ref_stars_all) if args.ref_stars_all is not None else None
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
    log_stage("init_inputs_mode", phase_t0, meta={"npz_mode": int(npz_mode)})

    phase_t0 = time.perf_counter()
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
    out_csv_nonref_inner_border_pre_knee = (
        args.out_csv_nonref_inner_border_pre_knee
        if args.out_csv_nonref_inner_border_pre_knee is not None
        else (
            out_csv_nonref_inner_border.parent
            / "variable_candidates_nonref_only_inner_border_pre_knee.csv"
        )
    )
    out_overlap_expr = (
        args.out_overlap_expr if args.out_overlap_expr is not None else (base / "ref_target_overlap_polygon_expr.json")
    )
    out_overlap_expr_png = args.out_overlap_expr_png if args.out_overlap_expr_png is not None else None
    out_png = args.out_png if args.out_png is not None else (base / "variable_candidates_rank.png")
    out_png_aligned = out_png.with_name(f"{out_png.stem}_aligned_to_a{out_png.suffix}")
    expected_outputs = [
        out_csv_nonref_inner_border,
        out_overlap_expr,
        out_png,
        out_png_aligned,
    ]
    if csv_detail:
        expected_outputs.extend([out_csv, out_csv_nonref, out_csv_nonref_inner_border_pre_knee, out_csv_ref_missing])
    if out_overlap_expr_png is not None:
        expected_outputs.append(out_overlap_expr_png)
    if (not args.overwrite) and all(p.exists() for p in expected_outputs):
        print("SKIP rank_variable_candidates.py: outputs already exist (use --overwrite to regenerate)")
        for p in expected_outputs:
            print(f"EXISTS {p}")
        return

    out_csv_nonref_inner_border.parent.mkdir(parents=True, exist_ok=True)
    if csv_detail:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_csv_nonref.parent.mkdir(parents=True, exist_ok=True)
        out_csv_nonref_inner_border_pre_knee.parent.mkdir(parents=True, exist_ok=True)
        out_csv_ref_missing.parent.mkdir(parents=True, exist_ok=True)
    out_overlap_expr.parent.mkdir(parents=True, exist_ok=True)
    if out_overlap_expr_png is not None:
        out_overlap_expr_png.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    log_stage("prepare_output_paths", phase_t0)

    phase_t0 = time.perf_counter()
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
        xy_ref, flux_ref, ref_meta = load_stars_npz(ref_stars_all_path, return_meta=True)
        if len(xy_ref) == 0:
            raise RuntimeError(f"No stars in reference stars NPZ: {ref_stars_all_path}")
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
    log_stage("load_reference", phase_t0, meta={"ref_stars": int(len(xy_ref))})

    phase_t0 = time.perf_counter()
    ref_wcs, ref_wcs_source = resolve_reference_celestial_wcs(base, ref_path, ref_stars_all_path)
    if ref_wcs is None:
        print("WARNING: Reference WCS not available, RA/DEC columns will be empty in nonref inner-border CSV.")
    else:
        print(f"RA/DEC WCS source: {ref_wcs_source}")
    ref_mjd, ref_mjd_source = resolve_reference_mjd(base, ref_path, ref_stars_all_path, preferred_path=ref_wcs_source)
    if np.isfinite(ref_mjd):
        print(f"MJD source: {ref_mjd_source}, value={ref_mjd:.8f}")
    else:
        print("WARNING: Reference MJD not available, MJD column will be empty in nonref inner-border CSV.")
    arcsec_per_px_x = float("nan")
    arcsec_per_px_y = float("nan")
    arcsec_per_px_mean = float("nan")
    if ref_wcs is not None:
        try:
            # proj_plane_pixel_scales() returns degrees/pixel for each pixel axis.
            px_scales_deg = np.asarray(proj_plane_pixel_scales(ref_wcs), dtype=np.float64)
            if px_scales_deg.size >= 2:
                arcsec_per_px_x = float(px_scales_deg[0] * 3600.0)
                arcsec_per_px_y = float(px_scales_deg[1] * 3600.0)
                arcsec_per_px_mean = 0.5 * (arcsec_per_px_x + arcsec_per_px_y)
                print(
                    "Global plate scale arcsec/px: "
                    f"x={arcsec_per_px_x:.6f}, y={arcsec_per_px_y:.6f}, mean={arcsec_per_px_mean:.6f}"
                )
        except Exception:
            print("WARNING: Failed to estimate global plate scale from reference WCS.")
    log_stage(
        "resolve_reference_wcs_mjd_scale",
        phase_t0,
        meta={
            "has_wcs": int(ref_wcs is not None),
            "has_mjd": int(np.isfinite(ref_mjd)),
            "arcsec_per_px_mean": float(arcsec_per_px_mean) if np.isfinite(arcsec_per_px_mean) else None,
        },
    )

    n_ref = len(xy_ref)
    # First column uses reference flux as baseline.
    measurements = [np.asarray(flux_ref, dtype=np.float64)]
    used_files = [ref_path if ref_path is not None else ref_stars_all_path]
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

    t_targets_total = time.perf_counter()
    for stars_npz_path, align_npz_path, p in target_entries:
        frame_label = stars_npz_path.name if npz_mode else p.name
        t_target_frame = time.perf_counter()
        try:
            t_target_stage = time.perf_counter()
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
            else:
                data = fits.getdata(p).astype(float)
                h_b, w_b = data.shape
                xy_b, flux_b = detect_stars(data, max_stars=int(args.max_stars))
                if len(xy_b) == 0:
                    raise RuntimeError("No stars detected.")

                dx0, dy0 = estimate_translation_from_stars(xy_ref, xy_b, top_n=300, bin_size=2.0)
                ai_idx, bi_idx = build_matches(xy_ref, xy_b, dx0, dy0, match_radius=float(args.match_radius))
            log_stage(
                "target_load_and_match",
                t_target_stage,
                meta={
                    "frame": frame_label,
                    "mode": "npz" if npz_mode else "fits",
                    "detected_stars": int(len(xy_b)),
                    "matched_stars": int(len(ai_idx)),
                },
            )
            if len(ai_idx) < 10:
                raise RuntimeError(f"Too few matches ({len(ai_idx)}).")

            t_target_stage = time.perf_counter()
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
            log_stage(
                "target_scale_and_accumulate",
                t_target_stage,
                meta={
                    "frame": frame_label,
                    "matched_stars": int(len(ai_idx)),
                },
            )

            t_target_stage = time.perf_counter()
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
            log_stage(
                "target_collect_nonref",
                t_target_stage,
                meta={"frame": frame_label},
            )
            log_stage(
                "target_frame_total",
                t_target_frame,
                meta={
                    "frame": frame_label,
                    "status": "ok",
                    "matched_stars": int(len(ai_idx)),
                },
            )
        except Exception as exc:
            failed_files.append((frame_label, str(exc)))
            log_stage(
                "target_frame_total",
                t_target_frame,
                status="error",
                meta={
                    "frame": frame_label,
                    "status": "error",
                    "error": str(exc),
                },
            )
    log_stage(
        "process_targets_total",
        t_targets_total,
        meta={
            "target_count": int(len(target_entries)),
            "target_failed": int(len(failed_files)),
            "target_used": int(len(used_files) - 1),
        },
    )

    t_overlap_build = time.perf_counter()
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
    log_stage(
        "build_final_overlap_polygons",
        t_overlap_build,
        meta={"frame_count": int(len(final_overlap_polygons_by_frame))},
    )

    t_rank_core = time.perf_counter()

    t_rank_stage = time.perf_counter()
    flux_mat = np.vstack(measurements).T  # [n_ref, n_frames_used]
    log_stage(
        "rank_build_flux_matrix",
        t_rank_stage,
        meta={"n_ref_stars": int(n_ref), "n_used_frames": int(flux_mat.shape[1])},
    )

    t_rank_stage = time.perf_counter()
    n_obs = np.sum(np.isfinite(flux_mat), axis=1)
    log_stage(
        "rank_count_observations",
        t_rank_stage,
        meta={"n_ref_stars": int(n_ref)},
    )

    t_rank_stage = time.perf_counter()
    med_flux = np.nanmedian(flux_mat, axis=1)
    log_stage(
        "rank_compute_median_flux",
        t_rank_stage,
        meta={"n_ref_stars": int(n_ref)},
    )

    t_rank_stage = time.perf_counter()
    ref_region_ok = np.ones(n_ref, dtype=bool)
    if ref_valid_region_paths is not None:
        # Batch point-in-polygon checks to avoid Python per-star loops.
        ref_region_ok = np.zeros(n_ref, dtype=bool)
        for path_obj in ref_valid_region_paths:
            ref_region_ok |= path_obj.contains_points(xy_ref, radius=1e-9)
    log_stage(
        "rank_build_region_mask",
        t_rank_stage,
        meta={
            "n_ref_stars": int(n_ref),
            "has_ref_valid_region": int(ref_valid_region_paths is not None),
            "region_path_count": int(len(ref_valid_region_paths)) if ref_valid_region_paths is not None else 0,
        },
    )
    log_stage(
        "rank_build_flux_matrix_and_region_mask_total",
        t_rank_core,
        meta={
            "n_ref_stars": int(n_ref),
            "n_used_frames": int(flux_mat.shape[1]),
            "has_ref_valid_region": int(ref_valid_region_paths is not None),
        },
    )

    t_rank_stage = time.perf_counter()
    valid = (n_obs >= int(args.min_observations)) & np.isfinite(med_flux) & (med_flux > 0.0) & ref_region_ok
    n_valid = int(np.count_nonzero(valid))
    log_stage(
        "rank_filter_valid_candidates",
        t_rank_stage,
        meta={"valid_count": n_valid, "min_observations": int(args.min_observations)},
    )
    if n_valid == 0:
        raise RuntimeError("No stars meet min observation requirement.")

    t_rank_stage = time.perf_counter()
    rel = flux_mat / np.maximum(med_flux[:, None], 1e-12)
    # Vectorized robust MAD per star (ignore NaN values from missing observations).
    rel_med = np.nanmedian(rel, axis=1)
    mad_rel = 1.4826 * np.nanmedian(np.abs(rel - rel_med[:, None]), axis=1)
    p95 = np.nanpercentile(rel, 95, axis=1)
    p05 = np.nanpercentile(rel, 5, axis=1)
    amp_rel = p95 - p05

    # Composite variability score. Higher means more variable-like.
    score = mad_rel + 0.5 * amp_rel
    score[~valid] = np.nan
    log_stage(
        "rank_compute_variability_metrics",
        t_rank_stage,
        meta={"valid_count": n_valid},
    )

    t_rank_stage = time.perf_counter()
    idx = np.where(valid)[0]
    order = idx[np.argsort(score[idx])[::-1]]
    log_stage(
        "rank_sort_candidates",
        t_rank_stage,
        meta={"ranked_candidates": int(len(order))},
    )

    t_rank_stage = time.perf_counter()
    if csv_detail:
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
    log_stage(
        "rank_write_csv",
        t_rank_stage,
        meta={"rows_written": int(len(order)) if csv_detail else 0, "csv_detail": int(csv_detail)},
    )
    log_stage(
        "compute_scores_and_write_rank_csv",
        t_rank_core,
        meta={"ranked_candidates": int(len(order)), "csv_detail": int(csv_detail), "wrote_rank_csv": int(csv_detail)},
    )

    t_nonref_outputs = time.perf_counter()
    nonref_plot_xy = np.empty((0, 2), dtype=np.float64)
    nonref_plot_xy_rank = np.empty((0, 2), dtype=np.float64)
    nonref_plot_xy_inner_all = np.empty((0, 2), dtype=np.float64)
    nonref_has_ref_nearby_mask_rank = np.empty((0,), dtype=bool)
    nonref_plot_ranks_rank = np.empty((0,), dtype=np.int32)
    nonref_ref_check_radius = float(args.nonref_ref_check_radius)
    do_nonref_ref_check = (nonref_ref_check_radius > 0.0) and (len(xy_ref) > 0)
    ref_tree = cKDTree(np.asarray(xy_ref, dtype=np.float64)) if do_nonref_ref_check else None
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

        if csv_detail:
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

        nonref_inner_border_header = [
            "rank",
            "x",
            "y",
            "n_frames_detected",
            "n_detections",
            "median_flux_norm",
            "frames",
            "has_ref_nearby",
            "is_nonref_unique_vs_ref_all",
            "nearest_ref_dist_px",
            "ra_deg",
            "dec_deg",
            "arcsec_per_px_x",
            "arcsec_per_px_y",
            "arcsec_per_px_mean",
            "mjd",
        ]
        nonref_inner_border_pre_knee_header = nonref_inner_border_header + [
            "kept_in_inner_border_csv",
            "drop_reason_after_pre_knee",
        ]
        max_inner_csv = int(args.top_k_nonref_inner_border_csv)
        nonref_inner_min_median_flux_norm = 10.0
        nonref_order_inside = nonref_order[nonref_inside[nonref_order]]
        if do_nonref_ref_check:
            nearest_dist_all = np.asarray(ref_tree.query(nonref_xy_arr, k=1)[0], dtype=np.float64)
            has_ref_nearby_all = np.asarray(nearest_dist_all <= nonref_ref_check_radius, dtype=bool)
        else:
            nearest_dist_all = np.full(nonref_count, np.nan, dtype=np.float64)
            has_ref_nearby_all = np.zeros(nonref_count, dtype=bool)
        kept_in_inner_border_csv = np.zeros(nonref_count, dtype=bool)
        drop_reason_after_pre_knee = np.full(nonref_count, "", dtype=object)
        final_rank_by_idx = np.zeros(nonref_count, dtype=np.int32)
        rank_inner = 1
        for i in nonref_order:
            if not nonref_inside[i]:
                continue
            if has_ref_nearby_all[i]:
                continue
            if not (float(nonref_median_flux[i]) > float(nonref_inner_min_median_flux_norm)):
                drop_reason_after_pre_knee[i] = "median_flux_norm_le_10"
                continue
            if max_inner_csv > 0 and rank_inner > max_inner_csv:
                drop_reason_after_pre_knee[i] = "exceeds_top_k_nonref_inner_border_csv"
                continue
            kept_in_inner_border_csv[i] = True
            final_rank_by_idx[i] = rank_inner
            rank_inner += 1

        if csv_detail:
            with out_csv_nonref_inner_border_pre_knee.open("w", newline="", encoding="utf-8") as f_pre:
                writer_pre = csv.writer(f_pre)
                writer_pre.writerow(nonref_inner_border_pre_knee_header)
                rank_pre = 1
                for i in nonref_order:
                    if not nonref_inside[i]:
                        continue
                    frame_names = ";".join(sorted(nonref_frame_sets[i]))
                    nearest_ref_dist = float(nearest_dist_all[i])
                    has_ref_nearby = bool(has_ref_nearby_all[i])
                    # Pre-filter CSV still excludes points near reference stars (definition A).
                    if has_ref_nearby:
                        continue
                    if ref_wcs is not None:
                        try:
                            ra_deg, dec_deg = ref_wcs.pixel_to_world_values(
                                float(nonref_xy_arr[i, 0]),
                                float(nonref_xy_arr[i, 1]),
                            )
                        except Exception:
                            ra_deg, dec_deg = float("nan"), float("nan")
                    else:
                        ra_deg, dec_deg = float("nan"), float("nan")
                    writer_pre.writerow(
                        [
                            rank_pre,
                            f"{nonref_xy_arr[i, 0]:.4f}",
                            f"{nonref_xy_arr[i, 1]:.4f}",
                            int(nonref_n_frames[i]),
                            int(nonref_n_det[i]),
                            f"{nonref_median_flux[i]:.8f}",
                            frame_names,
                            int(has_ref_nearby),
                            int(not has_ref_nearby),
                            f"{nearest_ref_dist:.6f}" if np.isfinite(nearest_ref_dist) else "",
                            f"{float(ra_deg):.8f}" if np.isfinite(ra_deg) else "",
                            f"{float(dec_deg):.8f}" if np.isfinite(dec_deg) else "",
                            f"{arcsec_per_px_x:.8f}" if np.isfinite(arcsec_per_px_x) else "",
                            f"{arcsec_per_px_y:.8f}" if np.isfinite(arcsec_per_px_y) else "",
                            f"{arcsec_per_px_mean:.8f}" if np.isfinite(arcsec_per_px_mean) else "",
                            f"{ref_mjd:.8f}" if np.isfinite(ref_mjd) else "",
                            int(kept_in_inner_border_csv[i]),
                            str(drop_reason_after_pre_knee[i]),
                        ]
                    )
                    rank_pre += 1

        with out_csv_nonref_inner_border.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(nonref_inner_border_header)
            for i in nonref_order:
                if not kept_in_inner_border_csv[i]:
                    continue
                frame_names = ";".join(sorted(nonref_frame_sets[i]))
                nearest_ref_dist = float(nearest_dist_all[i])
                has_ref_nearby = bool(has_ref_nearby_all[i])
                if ref_wcs is not None:
                    try:
                        ra_deg, dec_deg = ref_wcs.pixel_to_world_values(
                            float(nonref_xy_arr[i, 0]),
                            float(nonref_xy_arr[i, 1]),
                        )
                    except Exception:
                        ra_deg, dec_deg = float("nan"), float("nan")
                else:
                    ra_deg, dec_deg = float("nan"), float("nan")
                writer.writerow(
                    [
                        int(final_rank_by_idx[i]),
                        f"{nonref_xy_arr[i, 0]:.4f}",
                        f"{nonref_xy_arr[i, 1]:.4f}",
                        int(nonref_n_frames[i]),
                        int(nonref_n_det[i]),
                        f"{nonref_median_flux[i]:.8f}",
                        frame_names,
                        int(has_ref_nearby),
                        int(not has_ref_nearby),
                        f"{nearest_ref_dist:.6f}" if np.isfinite(nearest_ref_dist) else "",
                        f"{float(ra_deg):.8f}" if np.isfinite(ra_deg) else "",
                        f"{float(dec_deg):.8f}" if np.isfinite(dec_deg) else "",
                        f"{arcsec_per_px_x:.8f}" if np.isfinite(arcsec_per_px_x) else "",
                        f"{arcsec_per_px_y:.8f}" if np.isfinite(arcsec_per_px_y) else "",
                        f"{arcsec_per_px_mean:.8f}" if np.isfinite(arcsec_per_px_mean) else "",
                        f"{ref_mjd:.8f}" if np.isfinite(ref_mjd) else "",
                    ]
                )

        top_k_nonref = int(args.top_k_nonref)
        nonref_plot_xy_inner_all = nonref_xy_arr[nonref_order_inside]
        keep_n = len(nonref_order_inside) if top_k_nonref <= 0 else min(top_k_nonref, len(nonref_order_inside))
        nonref_plot_xy = nonref_xy_arr[nonref_order_inside[:keep_n]]
        keep_n_rank = min(400, len(nonref_order_inside))
        rank_idx = nonref_order_inside[:keep_n_rank]
        nonref_plot_xy_rank = nonref_xy_arr[rank_idx]
        nonref_has_ref_nearby_mask_rank = np.asarray(has_ref_nearby_all[rank_idx], dtype=bool)
        nonref_plot_ranks_rank = np.arange(1, keep_n_rank + 1, dtype=np.int32)
    else:
        if csv_detail:
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
                        "has_ref_nearby",
                        "is_nonref_unique_vs_ref_all",
                        "nearest_ref_dist_px",
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
                    "has_ref_nearby",
                    "is_nonref_unique_vs_ref_all",
                    "nearest_ref_dist_px",
                    "ra_deg",
                    "dec_deg",
                    "arcsec_per_px_x",
                    "arcsec_per_px_y",
                    "arcsec_per_px_mean",
                    "mjd",
                ]
            )
        if csv_detail:
            with out_csv_nonref_inner_border_pre_knee.open("w", newline="", encoding="utf-8") as f:
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
                        "has_ref_nearby",
                        "is_nonref_unique_vs_ref_all",
                        "nearest_ref_dist_px",
                        "ra_deg",
                        "dec_deg",
                        "arcsec_per_px_x",
                        "arcsec_per_px_y",
                        "arcsec_per_px_mean",
                        "mjd",
                        "kept_in_inner_border_csv",
                        "drop_reason_after_pre_knee",
                    ]
                )
    log_stage(
        "write_nonref_outputs",
        t_nonref_outputs,
        meta={"nonref_count": int(nonref_count), "csv_detail": int(csv_detail)},
    )

    # Stars detected in reference but never matched in any successfully used target frame.
    t_ref_missing_outputs = time.perf_counter()
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

    if csv_detail:
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
    log_stage(
        "write_ref_missing_outputs",
        t_ref_missing_outputs,
        meta={"ref_missing_count": int(len(ref_missing_order)), "csv_detail": int(csv_detail)},
    )

    t_overlap_write = time.perf_counter()
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
    log_stage(
        "write_overlap_outputs",
        t_overlap_write,
        meta={"with_overlap_png": int(out_overlap_expr_png is not None)},
    )

    t_plot_outputs = time.perf_counter()
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
        nonref_has_ref_nearby_mask=nonref_has_ref_nearby_mask_rank,
    )
    log_stage(
        "write_candidate_png_outputs",
        t_plot_outputs,
        meta={"with_nonref": int(len(nonref_plot_xy_rank) > 0), "with_ref_missing": int(len(ref_missing_plot_xy_rank) > 0)},
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
    if csv_detail:
        print(f"WROTE {out_csv}")
    else:
        print(f"SKIP {out_csv} (csv_detail disabled)")
    if csv_detail:
        print(f"WROTE {out_csv_nonref}")
    else:
        print(f"SKIP {out_csv_nonref} (csv_detail disabled)")
    print(f"WROTE {out_csv_nonref_inner_border}")
    if csv_detail:
        print(f"WROTE {out_csv_nonref_inner_border_pre_knee}")
    else:
        print(f"SKIP {out_csv_nonref_inner_border_pre_knee} (csv_detail disabled)")
    if csv_detail:
        print(f"WROTE {out_csv_ref_missing}")
    else:
        print(f"SKIP {out_csv_ref_missing} (csv_detail disabled)")
    print(f"WROTE {out_overlap_expr}")
    if out_overlap_expr_png is not None:
        print(f"WROTE {out_overlap_expr_png}")
    print(f"WROTE {out_png}")
    print(f"WROTE {out_png_aligned}")
    print(f"timing_jsonl={timing_path}")
    print(f"timing_run_id={logger.run_id}")
    print(f"Non-reference-only stars: {nonref_count}")
    print("Inner-border nonref filter: median_flux_norm > 10 (knee disabled)")
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
    run_script_with_timing(main, script_name=Path(__file__).name)
