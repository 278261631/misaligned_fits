from pathlib import Path
import argparse

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates, median_filter

from alignment_common import detect_stars, select_stars_uniform_grid


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
    if len(xy_all) == 0:
        raise RuntimeError("No stars detected from reprojected image.")
    if args.no_uniform_selection:
        xy_align, flux_align = detect_stars(detect_img, max_stars=int(args.max_stars))
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

    if args.skip_median_filter:
        print("median_filter=skipped")
    else:
        print(f"median_size={int(args.median_size)}")
    print(f"stars_align={len(xy_align)}")
    print(f"stars_all={len(xy_all)}")
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
    main()
