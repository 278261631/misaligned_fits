from pathlib import Path
import argparse

import numpy as np
from astropy.io import fits

from alignment_common import resample_b_to_a, save_png


def parse_args():
    parser = argparse.ArgumentParser(description="Apply alignment solution and write FITS/PNG outputs.")
    parser.add_argument("--a", type=Path, required=True, help="Reference FITS path.")
    parser.add_argument("--b", type=Path, required=True, help="Target FITS path.")
    parser.add_argument("--align", type=Path, required=True, help="Alignment solution file (.npz).")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory (default: <a_dir>/direct_no_wcs).")
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = args.outdir if args.outdir is not None else (args.a.parent / "direct_no_wcs")
    outdir.mkdir(parents=True, exist_ok=True)

    a_data = fits.getdata(args.a).astype(float)
    b_data = fits.getdata(args.b).astype(float)
    a_header = fits.getheader(args.a)

    sol = np.load(args.align, allow_pickle=True)
    cx = np.asarray(sol["cx"], dtype=float)
    cy = np.asarray(sol["cy"], dtype=float)
    fit_degree = int(np.asarray(sol["fit_degree"]).ravel()[0])

    out = resample_b_to_a(a_data, b_data, cx, cy, fit_degree)

    out_fits = outdir / f"{args.b.stem}_on_{args.a.stem}_from_stars_poly{fit_degree}.fits"
    fits.writeto(out_fits, out, a_header, overwrite=True)

    preview_png = outdir / f"{args.b.stem}_on_{args.a.stem}_from_stars_poly{fit_degree}_preview.png"
    save_png(out, preview_png, "Direct fit from stars: B -> A")

    absdiff = np.abs(np.nan_to_num(a_data, nan=0.0) - np.nan_to_num(out, nan=0.0))
    absdiff_png = outdir / f"{args.b.stem}_on_{args.a.stem}_absdiff_from_stars_poly{fit_degree}.png"
    save_png(absdiff, absdiff_png, "Abs diff |A - B_from_stars|")

    print(f"WROTE {out_fits}")
    print(f"WROTE {preview_png}")
    print(f"WROTE {absdiff_png}")


if __name__ == "__main__":
    main()
