from pathlib import Path
import argparse

import numpy as np
from astropy.io import fits

from alignment_common import detect_stars


def parse_args():
    parser = argparse.ArgumentParser(description="Export stars from one FITS file.")
    parser.add_argument("--fits", type=Path, required=True, help="Input FITS path.")
    parser.add_argument("--out", type=Path, default=None, help="Output stars file (.npz).")
    parser.add_argument("--max-stars", type=int, default=5000, help="Maximum stars to keep.")
    return parser.parse_args()


def main():
    args = parse_args()
    fits_path = args.fits
    out_path = args.out if args.out is not None else fits_path.with_suffix(".stars.npz")

    data = fits.getdata(fits_path).astype(float)
    xy, flux = detect_stars(data, max_stars=args.max_stars)
    if len(xy) == 0:
        raise RuntimeError(f"No stars detected: {fits_path}")

    np.savez_compressed(
        out_path,
        xy=xy.astype(np.float32),
        flux=flux.astype(np.float32),
        source_fits=str(fits_path),
        height=int(data.shape[0]),
        width=int(data.shape[1]),
    )
    print(f"stars={len(xy)}")
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
