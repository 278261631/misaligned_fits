from pathlib import Path
import argparse

import numpy as np
from astropy.io import fits

from alignment_common import detect_stars, select_stars_uniform_grid


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
    return parser.parse_args()


def main():
    args = parse_args()
    fits_path = args.fits
    out_path = args.out if args.out is not None else fits_path.with_suffix(".stars.npz")
    out_all_path = args.out_all if args.out_all is not None else out_path.with_name(f"{out_path.stem}.all.npz")

    data = fits.getdata(fits_path).astype(float)
    xy_all, flux_all = detect_stars(data, max_stars=0)
    if len(xy_all) == 0:
        raise RuntimeError(f"No stars detected: {fits_path}")
    if args.no_uniform_selection:
        xy_align, flux_align = detect_stars(data, max_stars=args.max_stars)
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
    print(f"stars_align={len(xy_align)}")
    print(f"stars_all={len(xy_all)}")
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
    main()
