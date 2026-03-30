from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from alignment_common import detect_stars, select_stars_uniform_grid
from timing_logger import run_script_with_timing


def parse_args():
    parser = argparse.ArgumentParser(description="Export stars from one FITS file.")
    parser.add_argument("--fits", type=Path, required=True, help="Input FITS path.")
    parser.add_argument("--out", type=Path, default=None, help="Output stars file (.npz) for alignment (default cap: 5000).")
    parser.add_argument("--out-all", type=Path, default=None, help="Output all-stars file (.npz).")
    parser.add_argument("--max-stars", type=int, default=5000, help="Maximum stars kept in --out file.")
    parser.add_argument("--uniform-grid-x", type=int, default=7, help="Grid columns for uniform star selection.")
    parser.add_argument("--uniform-grid-y", type=int, default=7, help="Grid rows for uniform star selection.")
    parser.add_argument(
        "--uniform-per-cell",
        type=int,
        default=80,
        help="Maximum stars selected per grid cell in --out.",
    )
    parser.add_argument(
        "--no-uniform-selection",
        action="store_true",
        help="Disable uniform-grid selection for --out and use global brightness ranking.",
    )
    parser.add_argument(
        "--out-valid-region",
        type=Path,
        default=None,
        help="Optional JSON path to export valid image region polygons (valid: finite and non-zero).",
    )
    parser.add_argument(
        "--out-valid-region-png",
        type=Path,
        default=None,
        help="Optional PNG path to visualize valid image region polygons. Skip when not provided.",
    )
    parser.add_argument(
        "--out-all-png",
        type=Path,
        default=None,
        help="Optional PNG path to visualize full image with stars from .all.npz. Skip when not provided.",
    )
    parser.add_argument(
        "--all-png-stretch",
        choices=("none", "normal", "strong"),
        default="strong",
        help="Contrast stretch mode for --out-all-png (default: strong).",
    )
    parser.add_argument(
        "--all-png-gamma",
        type=float,
        default=0.45,
        help="Gamma used after percentile stretch for --out-all-png (default: 0.45).",
    )
    parser.add_argument(
        "--all-png-min-flux-percentile",
        type=float,
        default=95.0,
        help="Only stars above this flux percentile are marked in --out-all-png (default: 95).",
    )
    parser.add_argument(
        "--min-flux",
        type=float,
        default=1.0,
        help="Absolute lower bound for exported star flux (keep flux >= this value, default: 1).",
    )
    parser.add_argument(
        "--min-flux-percentile",
        type=float,
        default=20.0,
        help="Percentile-based lower bound for exported star flux (0-100, default: 20).",
    )
    return parser.parse_args()


def _polygon_area_xy(poly_xy):
    if poly_xy.shape[0] < 3:
        return 0.0
    x = poly_xy[:, 0]
    y = poly_xy[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _extract_valid_region_polygons(valid_mask):
    # Contour at 0.5 extracts boundaries between invalid(0) and valid(1).
    fig = plt.figure()
    try:
        cs = plt.contour(valid_mask.astype(np.uint8), levels=[0.5], origin="lower")
        raw_segs = cs.allsegs[0] if len(cs.allsegs) > 0 else []
    finally:
        plt.close(fig)

    # Matplotlib versions can return either:
    # - a list of (N, 2) arrays (multiple segments), or
    # - a single (N, 2) ndarray (one segment).
    segs = []
    if isinstance(raw_segs, np.ndarray) and raw_segs.ndim == 2 and raw_segs.shape[1] == 2:
        segs = [raw_segs]
    else:
        for seg in raw_segs:
            pts = np.asarray(seg)
            if pts.ndim == 2 and pts.shape[1] == 2:
                segs.append(pts)

    polygons = []
    for seg in segs:
        pts = np.asarray(seg, dtype=np.float64)
        if pts.shape[0] < 3:
            continue
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
        area = _polygon_area_xy(pts)
        if area <= 0.0:
            continue
        polygons.append((area, pts))
    polygons.sort(key=lambda t: t[0], reverse=True)
    return polygons


def export_valid_region(valid_mask, out_json: Path | None, out_png: Path | None, source_fits: Path):
    if out_json is None and out_png is None:
        return

    if not np.any(valid_mask):
        raise RuntimeError("No valid pixels found (only zeros/NaNs).")

    polygons = _extract_valid_region_polygons(valid_mask)
    if len(polygons) == 0:
        raise RuntimeError("Failed to extract valid region polygon from mask.")

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_fits": str(source_fits),
            "valid_rule": "valid = isfinite(pixel) and pixel != 0",
            "height": int(valid_mask.shape[0]),
            "width": int(valid_mask.shape[1]),
            "polygon_count": int(len(polygons)),
            "polygons_xy": [pts.tolist() for _, pts in polygons],
            "areas": [float(a) for a, _ in polygons],
        }
        import json

        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"WROTE {out_json}")

    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.imshow(valid_mask.astype(np.uint8), origin="lower", cmap="gray", interpolation="nearest")
        for idx, (_, pts) in enumerate(polygons, start=1):
            ax.plot(pts[:, 0], pts[:, 1], "-", linewidth=1.0, label=f"poly_{idx}" if idx <= 8 else None)
        ax.set_title("Valid Region (finite and non-zero)")
        ax.set_axis_off()
        if len(polygons) <= 8:
            ax.legend(loc="upper right", framealpha=0.7, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"WROTE {out_png}")


def _stretch_image_for_display(image_data, mode: str, gamma: float):
    finite = np.isfinite(image_data)
    fill = np.nanmedian(image_data[finite]) if np.any(finite) else 0.0
    view = np.where(finite, image_data, fill).astype(np.float64, copy=False)
    if mode == "none":
        return view

    finite_vals = view[np.isfinite(view)]
    if finite_vals.size == 0:
        return view

    if mode == "strong":
        p_low, p_high = 0.5, 99.8
    else:
        p_low, p_high = 1.0, 99.5
    vmin = float(np.percentile(finite_vals, p_low))
    vmax = float(np.percentile(finite_vals, p_high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return view

    clipped = np.clip(view, vmin, vmax)
    norm = (clipped - vmin) / (vmax - vmin)
    safe_gamma = float(gamma) if float(gamma) > 0 else 1.0
    return np.power(norm, safe_gamma)


def export_all_stars_png(
    image_data,
    xy_all,
    flux_all,
    out_png: Path | None,
    stretch_mode: str,
    stretch_gamma: float,
    min_flux_percentile: float,
):
    if out_png is None:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    view = _stretch_image_for_display(image_data, stretch_mode, stretch_gamma)
    ax.imshow(view, origin="lower", cmap="gray", interpolation="nearest")
    flux = np.asarray(flux_all, dtype=np.float64)
    p = float(np.clip(min_flux_percentile, 0.0, 100.0))
    thr = float(np.percentile(flux, p)) if flux.size > 0 else np.inf
    keep = flux > thr
    xy_draw = xy_all[keep] if len(xy_all) == len(flux) else xy_all
    if len(xy_draw) > 0:
        ax.scatter(
            xy_draw[:, 0],
            xy_draw[:, 1],
            s=10,
            marker="o",
            facecolors="none",
            edgecolors="#FFD400",
            linewidths=0.6,
            alpha=0.9,
        )
    ax.set_title(
        f"Stars with flux > P{p:g} ({thr:.3g}): {len(xy_draw)} / {len(xy_all)} "
        f"(stretch={stretch_mode}, gamma={float(stretch_gamma):.2f})"
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"WROTE {out_png}")


def main():
    args = parse_args()
    fits_path = args.fits
    out_path = args.out if args.out is not None else fits_path.with_suffix(".stars.npz")
    out_all_path = args.out_all if args.out_all is not None else out_path.with_name(f"{out_path.stem}.all.npz")

    data = fits.getdata(fits_path).astype(float)
    valid_mask = np.isfinite(data) & (data != 0.0)
    export_valid_region(
        valid_mask,
        args.out_valid_region,
        args.out_valid_region_png,
        fits_path,
    )

    xy_all, flux_all = detect_stars(data, max_stars=0)
    stars_detected_raw = int(len(xy_all))
    if stars_detected_raw == 0:
        raise RuntimeError(f"No stars detected: {fits_path}")

    flux_thr_abs = float(args.min_flux) if args.min_flux is not None else None
    flux_thr_pct = None
    flux_thr_pct_value = None
    flux_filter_threshold = None
    if args.min_flux_percentile is not None:
        flux_thr_pct = float(np.clip(float(args.min_flux_percentile), 0.0, 100.0))
        flux_thr_pct_value = float(np.percentile(flux_all, flux_thr_pct))

    thr_candidates = []
    if flux_thr_abs is not None:
        thr_candidates.append(float(flux_thr_abs))
    if flux_thr_pct_value is not None:
        thr_candidates.append(float(flux_thr_pct_value))
    if len(thr_candidates) > 0:
        flux_filter_threshold = max(thr_candidates)
        keep = np.asarray(flux_all, dtype=np.float64) >= float(flux_filter_threshold)
        xy_all = xy_all[keep]
        flux_all = flux_all[keep]
        if len(xy_all) == 0:
            raise RuntimeError(
                "No stars remain after flux filtering. "
                f"min_flux={flux_thr_abs}, min_flux_percentile={flux_thr_pct}, "
                f"applied_threshold={float(flux_filter_threshold):.6g}"
            )

    if args.no_uniform_selection:
        order = np.argsort(flux_all)[::-1]
        if int(args.max_stars) > 0:
            order = order[: int(args.max_stars)]
        xy_align = xy_all[order]
        flux_align = flux_all[order]
    else:
        xy_align, flux_align = select_stars_uniform_grid(
            xy_all,
            flux_all,
            height=int(data.shape[0]),
            width=int(data.shape[1]),
            grid_x=int(args.uniform_grid_x),
            grid_y=int(args.uniform_grid_y),
            per_cell=int(args.uniform_per_cell),
            max_total=int(args.max_stars),
        )
    if len(xy_align) == 0:
        raise RuntimeError(f"No stars selected for alignment: {fits_path}")

    np.savez_compressed(
        out_path,
        xy=xy_align.astype(np.float32),
        flux=flux_align.astype(np.float32),
        source_fits=str(fits_path),
        max_stars=int(args.max_stars),
        height=int(data.shape[0]),
        width=int(data.shape[1]),
    )
    np.savez_compressed(
        out_all_path,
        xy=xy_all.astype(np.float32),
        flux=flux_all.astype(np.float32),
        source_fits=str(fits_path),
        height=int(data.shape[0]),
        width=int(data.shape[1]),
    )
    export_all_stars_png(
        data,
        xy_all,
        flux_all,
        args.out_all_png,
        args.all_png_stretch,
        args.all_png_gamma,
        args.all_png_min_flux_percentile,
    )
    print(f"stars_align={len(xy_align)}")
    print(f"stars_all={len(xy_all)}")
    print(f"stars_detected_raw={stars_detected_raw}")
    if flux_filter_threshold is None:
        print("flux_filter=disabled")
    else:
        print(
            "flux_filter=enabled "
            f"min_flux={flux_thr_abs} "
            f"min_flux_percentile={flux_thr_pct} "
            f"threshold={float(flux_filter_threshold):.6g} "
            f"kept={len(xy_all)}/{stars_detected_raw}"
        )
    if args.no_uniform_selection:
        print("uniform_selection=disabled")
    else:
        print(
            f"uniform_selection=enabled grid={int(args.uniform_grid_x)}x{int(args.uniform_grid_y)} "
            f"per_cell={int(args.uniform_per_cell)}"
        )
    print(f"WROTE {out_path}")
    print(f"WROTE {out_all_path}")


if __name__ == "__main__":
    run_script_with_timing(main, script_name=Path(__file__).name)
