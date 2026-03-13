from pathlib import Path
import argparse

import numpy as np
from astropy.io import fits

from alignment_common import detect_stars


def parse_args():
    parser = argparse.ArgumentParser(description="Export stars from one FITS file.")
    parser.add_argument("--fits", type=Path, required=True, help="Input FITS path.")
    parser.add_argument("--out", type=Path, default=None, help="Output stars file (.npz) for alignment (default cap: 5000).")
    parser.add_argument("--out-all", type=Path, default=None, help="Output all-stars file (.npz).")
    parser.add_argument("--max-stars", type=int, default=5000, help="Maximum stars kept in --out file.")
    return parser.parse_args()


def main():
    args = parse_args()
    fits_path = args.fits
    out_path = args.out if args.out is not None else fits_path.with_suffix(".stars.npz")
    out_all_path = args.out_all if args.out_all is not None else out_path.with_name(f"{out_path.stem}.all.npz")

    data = fits.getdata(fits_path).astype(float)
    xy_align, flux_align = detect_stars(data, max_stars=args.max_stars)
    if len(xy_align) == 0:
        raise RuntimeError(f"No stars detected: {fits_path}")
    xy_all, flux_all = detect_stars(data, max_stars=0)

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
    print(f"WROTE {out_path}")
    print(f"WROTE {out_all_path}")


if __name__ == "__main__":
    main()
