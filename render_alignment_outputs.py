from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from alignment_common import (
    build_matches,
    detect_stars,
    estimate_translation_from_stars,
    resample_b_to_a,
    save_png,
)
from timing_logger import run_script_with_timing


def _ordered_hull(x, y):
    if len(x) < 3:
        return None
    pts = np.column_stack([x, y])
    c = np.mean(pts, axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)
    ring = pts[order]
    return np.vstack([ring, ring[0]])


def _save_match_selection_png(a_data, b_data, xa, ya, xb, yb, out_path: Path):
    n = len(xa)
    colors = plt.cm.tab20(np.linspace(0, 1, min(max(n, 1), 20)))
    ci = np.arange(n) % len(colors)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    ax[0].imshow(a_data, origin="lower", cmap="gray")
    ax[0].scatter(xa, ya, s=12, c=colors[ci], edgecolors="none")
    ax[0].set_title(f"Matched stars on A (n={n})")
    ax[0].set_axis_off()

    ax[1].imshow(b_data, origin="lower", cmap="gray")
    ax[1].scatter(xb, yb, s=12, c=colors[ci], edgecolors="none")
    ax[1].set_title(f"Matched stars on B (n={n})")
    ax[1].set_axis_off()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_match_shape_png(xa, ya, xb, yb, dx0, dy0, out_path: Path):
    xb_s = xb + dx0
    yb_s = yb + dy0

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    ax.scatter(xa, ya, s=10, c="tab:cyan", alpha=0.8, label="A matched stars")
    ax.scatter(xb_s, yb_s, s=10, c="tab:orange", alpha=0.6, label="B matched stars (shifted)")

    ring_a = _ordered_hull(xa, ya)
    ring_b = _ordered_hull(xb_s, yb_s)
    if ring_a is not None:
        ax.plot(ring_a[:, 0], ring_a[:, 1], c="tab:cyan", lw=1.5)
    if ring_b is not None:
        ax.plot(ring_b[:, 0], ring_b[:, 1], c="tab:orange", lw=1.5)

    ax.set_title("Matched-star shape comparison")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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

    # Visualize selected matched stars and point-cloud shape.
    xy_a, _ = detect_stars(a_data, max_stars=5000)
    xy_b, _ = detect_stars(b_data, max_stars=5000)
    if len(xy_a) > 0 and len(xy_b) > 0:
        if "dx0" in sol.files and "dy0" in sol.files:
            dx0 = float(np.asarray(sol["dx0"]).ravel()[0])
            dy0 = float(np.asarray(sol["dy0"]).ravel()[0])
        else:
            dx0, dy0 = estimate_translation_from_stars(xy_a, xy_b, top_n=300, bin_size=2.0)

        radii = [8.0, 12.0, 16.0, 24.0]
        min_matches = int(np.asarray(sol["matches_initial"]).ravel()[0]) if "matches_initial" in sol.files else 100
        ai_idx = bi_idx = np.array([], dtype=int)
        for radius in radii:
            ai_idx, bi_idx = build_matches(xy_a, xy_b, dx0, dy0, match_radius=radius)
            if len(ai_idx) >= min_matches:
                break
        if len(ai_idx) == 0:
            ai_idx, bi_idx = build_matches(xy_a, xy_b, dx0, dy0, match_radius=radii[-1])

        if len(ai_idx) > 0:
            xa, ya = xy_a[ai_idx, 0], xy_a[ai_idx, 1]
            xb, yb = xy_b[bi_idx, 0], xy_b[bi_idx, 1]
            match_png = outdir / f"{args.b.stem}_on_{args.a.stem}_matched_stars.png"
            shape_png = outdir / f"{args.b.stem}_on_{args.a.stem}_matched_shape.png"
            _save_match_selection_png(a_data, b_data, xa, ya, xb, yb, match_png)
            _save_match_shape_png(xa, ya, xb, yb, dx0, dy0, shape_png)
        else:
            match_png = None
            shape_png = None
    else:
        match_png = None
        shape_png = None

    print(f"WROTE {out_fits}")
    print(f"WROTE {preview_png}")
    print(f"WROTE {absdiff_png}")
    if match_png is not None:
        print(f"WROTE {match_png}")
    if shape_png is not None:
        print(f"WROTE {shape_png}")


if __name__ == "__main__":
    run_script_with_timing(main, script_name=Path(__file__).name)
