from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates, median_filter

from alignment_common import detect_stars, select_stars_uniform_grid
from timing_logger import run_script_with_timing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Median-denoise FITS B, reproject to FITS A by WCS, then export projected FITS and stars."
    )
    parser.add_argument("--a", type=Path, required=True, help="Reference FITS A path (target WCS/grid).")
    parser.add_argument("--b", type=Path, required=True, help="Input FITS B path (source WCS/grid).")
    parser.add_argument("--out-fits", type=Path, required=True, help="Output reprojected FITS path.")
    parser.add_argument("--out-stars", type=Path, required=True, help="Output projected stars file (.npz) for alignment.")
    parser.add_argument("--out-stars-all", type=Path, default=None, help="Output projected all-stars file (.npz).")
    parser.add_argument(
        "--out-stars-all-png",
        type=Path,
        default=None,
        help="Optional PNG path to visualize reprojected image with stars from .all.npz. Skip when not provided.",
    )
    parser.add_argument(
        "--all-png-stretch",
        choices=("none", "normal", "strong"),
        default="strong",
        help="Contrast stretch mode for --out-stars-all-png (default: strong).",
    )
    parser.add_argument(
        "--all-png-gamma",
        type=float,
        default=0.45,
        help="Gamma used after percentile stretch for --out-stars-all-png (default: 0.45).",
    )
    parser.add_argument(
        "--all-png-min-flux-percentile",
        type=float,
        default=95.0,
        help="Only stars above this flux percentile are marked in --out-stars-all-png (default: 95).",
    )
    parser.add_argument(
        "--min-flux",
        type=float,
        default=5.0,
        help="Absolute lower bound for exported star flux (keep flux >= this value, default: 5).",
    )
    parser.add_argument(
        "--min-flux-percentile",
        type=float,
        default=40.0,
        help="Percentile-based lower bound for exported star flux (0-100, default: 40).",
    )
    parser.add_argument(
        "--median-size",
        type=int,
        default=3,
        help="Median filter kernel size applied to B before reprojection (odd integer recommended).",
    )
    parser.add_argument("--max-stars", type=int, default=5000, help="Maximum stars to keep in output stars file.")
    parser.add_argument("--uniform-grid-x", type=int, default=7, help="Grid columns for uniform star selection.")
    parser.add_argument("--uniform-grid-y", type=int, default=7, help="Grid rows for uniform star selection.")
    parser.add_argument(
        "--uniform-per-cell",
        type=int,
        default=80,
        help="Maximum stars selected per grid cell in --out-stars.",
    )
    parser.add_argument(
        "--no-uniform-selection",
        action="store_true",
        help="Disable uniform-grid selection for --out-stars and use global brightness ranking.",
    )
    parser.add_argument("--chunk-rows", type=int, default=256, help="Rows per block during reprojection.")
    parser.add_argument(
        "--skip-median-filter",
        action="store_true",
        help="Skip median filtering and reproject raw B data directly.",
    )
    return parser.parse_args()


def _safe_for_detection(img):
    arr = np.asarray(img, dtype=float)
    finite = np.isfinite(arr)
    fill = float(np.nanmedian(arr[finite])) if np.any(finite) else 0.0
    return np.where(finite, arr, fill)


def reproject_b_to_a_wcs(a_data, b_data, wcs_a: WCS, wcs_b: WCS, chunk_rows=256):
    h, w = a_data.shape
    out = np.full((h, w), np.nan, dtype=np.float32)
    xx = np.arange(w, dtype=float)

    for y0 in range(0, h, int(chunk_rows)):
        y1 = min(h, y0 + int(chunk_rows))
        yy = np.arange(y0, y1, dtype=float)
        gx, gy = np.meshgrid(xx, yy)

        lon, lat = wcs_a.pixel_to_world_values(gx.ravel(), gy.ravel())
        src_x, src_y = wcs_b.world_to_pixel_values(lon, lat)
        block = map_coordinates(
            b_data,
            [src_y.reshape(gx.shape), src_x.reshape(gx.shape)],
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )
        out[y0:y1, :] = block.astype(np.float32)
    return out


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

    a_data = fits.getdata(args.a).astype(float)
    b_data = fits.getdata(args.b).astype(float)
    a_header = fits.getheader(args.a)
    h, w = a_data.shape

    wcs_a = WCS(a_header).celestial
    wcs_b = WCS(fits.getheader(args.b)).celestial

    if args.skip_median_filter:
        b_input = b_data
    else:
        b_input = median_filter(b_data, size=int(args.median_size))
    out = reproject_b_to_a_wcs(a_data, b_input, wcs_a, wcs_b, chunk_rows=args.chunk_rows)
    out_stars_all = (
        args.out_stars_all
        if args.out_stars_all is not None
        else args.out_stars.with_name(f"{args.out_stars.stem}.all.npz")
    )

    args.out_fits.parent.mkdir(parents=True, exist_ok=True)
    args.out_stars.parent.mkdir(parents=True, exist_ok=True)
    out_stars_all.parent.mkdir(parents=True, exist_ok=True)

    fits.writeto(args.out_fits, out, a_header, overwrite=True)

    detect_img = _safe_for_detection(out)
    xy_all, flux_all = detect_stars(detect_img, max_stars=0)
    stars_detected_raw = int(len(xy_all))
    if stars_detected_raw == 0:
        raise RuntimeError("No stars detected from reprojected image.")

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
            height=int(h),
            width=int(w),
            grid_x=int(args.uniform_grid_x),
            grid_y=int(args.uniform_grid_y),
            per_cell=int(args.uniform_per_cell),
            max_total=int(args.max_stars),
        )
    if len(xy_align) == 0:
        raise RuntimeError("No stars selected for alignment from reprojected image.")

    np.savez_compressed(
        args.out_stars,
        xy=xy_align.astype(np.float32),
        flux=flux_align.astype(np.float32),
        max_stars=int(args.max_stars),
        source_fits=str(args.b),
        reference_fits=str(args.a),
        projected_fits=str(args.out_fits),
        height=int(h),
        width=int(w),
    )
    np.savez_compressed(
        out_stars_all,
        xy=xy_all.astype(np.float32),
        flux=flux_all.astype(np.float32),
        source_fits=str(args.b),
        reference_fits=str(args.a),
        projected_fits=str(args.out_fits),
        height=int(h),
        width=int(w),
    )
    export_all_stars_png(
        out,
        xy_all,
        flux_all,
        args.out_stars_all_png,
        args.all_png_stretch,
        args.all_png_gamma,
        args.all_png_min_flux_percentile,
    )

    if args.skip_median_filter:
        print("median_filter=skipped")
    else:
        print(f"median_size={int(args.median_size)}")
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
    print(f"WROTE {args.out_fits}")
    print(f"WROTE {args.out_stars}")
    print(f"WROTE {out_stars_all}")


if __name__ == "__main__":
    run_script_with_timing(main, script_name=Path(__file__).name)
