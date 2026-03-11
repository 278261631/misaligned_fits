from pathlib import Path
import argparse

import numpy as np

from alignment_common import (
    build_matches,
    estimate_translation_from_stars,
    eval_poly,
    fit_with_fallback,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Solve alignment from two star files.")
    parser.add_argument("--a-stars", type=Path, required=True, help="Reference stars file (.npz).")
    parser.add_argument("--b-stars", type=Path, required=True, help="Target stars file (.npz).")
    parser.add_argument("--out", type=Path, required=True, help="Output alignment solution file (.npz).")
    parser.add_argument(
        "--radii",
        nargs="+",
        type=float,
        default=[8.0, 12.0, 16.0, 24.0],
        help="Match radii attempts.",
    )
    parser.add_argument("--min-matches", type=int, default=100, help="Minimum initial matches required.")
    return parser.parse_args()


def main():
    args = parse_args()
    a = np.load(args.a_stars, allow_pickle=True)
    b = np.load(args.b_stars, allow_pickle=True)
    xy_a = np.asarray(a["xy"], dtype=float)
    xy_b = np.asarray(b["xy"], dtype=float)
    if len(xy_a) < 80 or len(xy_b) < 80:
        raise RuntimeError(f"Not enough stars for solve: A={len(xy_a)}, B={len(xy_b)}")

    dx0, dy0 = estimate_translation_from_stars(xy_a, xy_b, top_n=300, bin_size=2.0)

    ai_idx = bi_idx = np.array([], dtype=int)
    for radius in args.radii:
        ai_idx, bi_idx = build_matches(xy_a, xy_b, dx0, dy0, match_radius=radius)
        if len(ai_idx) >= args.min_matches:
            break
    if len(ai_idx) < args.min_matches:
        raise RuntimeError(f"Too few initial matches: {len(ai_idx)}")

    xa, ya = xy_a[ai_idx, 0], xy_a[ai_idx, 1]
    xb, yb = xy_b[bi_idx, 0], xy_b[bi_idx, 1]
    cx, cy, keep, fit_degree = fit_with_fallback(xa, ya, xb, yb)

    xa_k, ya_k = xa[keep], ya[keep]
    xb_k, yb_k = xb[keep], yb[keep]
    px, py = eval_poly(xa_k, ya_k, cx, cy, degree=fit_degree)
    r = np.hypot(px - xb_k, py - yb_k)

    np.savez_compressed(
        args.out,
        cx=np.asarray(cx, dtype=np.float64),
        cy=np.asarray(cy, dtype=np.float64),
        fit_degree=np.asarray([fit_degree], dtype=np.int32),
        dx0=np.asarray([dx0], dtype=np.float64),
        dy0=np.asarray([dy0], dtype=np.float64),
        matches_initial=np.asarray([len(ai_idx)], dtype=np.int32),
        matches_used=np.asarray([np.count_nonzero(keep)], dtype=np.int32),
        residual_mean=np.asarray([np.mean(r)], dtype=np.float64),
        residual_rms=np.asarray([np.sqrt(np.mean(r**2))], dtype=np.float64),
        residual_max=np.asarray([np.max(r)], dtype=np.float64),
        a_stars=str(args.a_stars),
        b_stars=str(args.b_stars),
    )
    print(f"initial_shift_dxdy=({dx0:.3f},{dy0:.3f})")
    print(f"fit_degree={fit_degree}")
    print(f"matches_initial={len(ai_idx)}, matches_used={np.count_nonzero(keep)}")
    print(f"fit_residual_mean={np.mean(r):.4f}px, rms={np.sqrt(np.mean(r**2)):.4f}px, max={np.max(r):.4f}px")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
