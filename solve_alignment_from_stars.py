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
        default=[8.0, 12.0, 16.0, 24.0, 32.0, 40.0],
        help="Match radii attempts.",
    )
    parser.add_argument("--min-matches", type=int, default=80, help="Minimum initial matches required.")
    parser.add_argument(
        "--balance-grid-x",
        type=int,
        default=7,
        help="Grid columns used to balance matched-star spatial distribution.",
    )
    parser.add_argument(
        "--balance-grid-y",
        type=int,
        default=7,
        help="Grid rows used to balance matched-star spatial distribution.",
    )
    parser.add_argument(
        "--balance-per-cell",
        type=int,
        default=160,
        help="Maximum matched pairs kept per grid cell.",
    )
    parser.add_argument(
        "--no-balance-matches",
        action="store_true",
        help="Disable grid-balanced down-selection of matched pairs.",
    )
    return parser.parse_args()


def _frame_size_from_npz(stars_npz, xy):
    if "height" in stars_npz.files and "width" in stars_npz.files:
        h = int(np.asarray(stars_npz["height"]).ravel()[0])
        w = int(np.asarray(stars_npz["width"]).ravel()[0])
        return max(h, 1), max(w, 1)
    if len(xy) == 0:
        return 1, 1
    w = int(np.ceil(np.nanmax(xy[:, 0]) + 1.0))
    h = int(np.ceil(np.nanmax(xy[:, 1]) + 1.0))
    return max(h, 1), max(w, 1)


def _select_balanced_matches(xy_a, ai_idx, bi_idx, dist, h, w, grid_x, grid_y, per_cell):
    gx = max(int(grid_x), 1)
    gy = max(int(grid_y), 1)
    pc = max(int(per_cell), 1)
    if len(ai_idx) == 0:
        return ai_idx, bi_idx, np.zeros((gy, gx), dtype=np.int32)

    xa = np.clip(xy_a[ai_idx, 0], 0.0, w - 1e-6)
    ya = np.clip(xy_a[ai_idx, 1], 0.0, h - 1e-6)
    ix = np.minimum((xa / w * gx).astype(int), gx - 1)
    iy = np.minimum((ya / h * gy).astype(int), gy - 1)
    cell_id = iy * gx + ix

    keep_local = []
    cell_counts = np.zeros((gy, gx), dtype=np.int32)
    for cid in range(gx * gy):
        idx = np.where(cell_id == cid)[0]
        if len(idx) == 0:
            continue
        # Prefer tighter pre-fit matches in each cell.
        loc = idx[np.argsort(dist[idx])]
        loc = loc[: min(pc, len(loc))]
        keep_local.append(loc)
        cy, cx = divmod(cid, gx)
        cell_counts[cy, cx] = int(len(loc))

    if len(keep_local) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), cell_counts
    k = np.concatenate(keep_local)
    return ai_idx[k], bi_idx[k], cell_counts


def main():
    args = parse_args()
    a = np.load(args.a_stars, allow_pickle=True)
    b = np.load(args.b_stars, allow_pickle=True)
    xy_a = np.asarray(a["xy"], dtype=float)
    xy_b = np.asarray(b["xy"], dtype=float)
    if len(xy_a) < 60 or len(xy_b) < 60:
        raise RuntimeError(f"Not enough stars for solve: A={len(xy_a)}, B={len(xy_b)}")

    dx0, dy0 = estimate_translation_from_stars(xy_a, xy_b, top_n=300, bin_size=2.0)

    h_a, w_a = _frame_size_from_npz(a, xy_a)
    ai_idx = bi_idx = np.array([], dtype=int)
    raw_ai_best = raw_bi_best = np.array([], dtype=int)
    selected_cell_counts = None
    for radius in args.radii:
        raw_ai, raw_bi = build_matches(xy_a, xy_b, dx0, dy0, match_radius=radius)
        if len(raw_ai) > len(raw_ai_best):
            raw_ai_best, raw_bi_best = raw_ai, raw_bi
        if len(raw_ai) < args.min_matches:
            continue
        if args.no_balance_matches:
            ai_idx, bi_idx = raw_ai, raw_bi
            break
        dist0 = np.hypot(
            xy_a[raw_ai, 0] - (xy_b[raw_bi, 0] + dx0),
            xy_a[raw_ai, 1] - (xy_b[raw_bi, 1] + dy0),
        )
        bal_ai, bal_bi, cell_counts = _select_balanced_matches(
            xy_a,
            raw_ai,
            raw_bi,
            dist0,
            h=h_a,
            w=w_a,
            grid_x=int(args.balance_grid_x),
            grid_y=int(args.balance_grid_y),
            per_cell=int(args.balance_per_cell),
        )
        if len(bal_ai) >= args.min_matches:
            ai_idx, bi_idx = bal_ai, bal_bi
            selected_cell_counts = cell_counts
            break
        # Fallback if balancing keeps too few pairs.
        ai_idx, bi_idx = raw_ai, raw_bi
        break
    if len(ai_idx) < args.min_matches:
        raise RuntimeError(f"Too few initial matches: {len(ai_idx)} (best raw={len(raw_ai_best)})")

    if selected_cell_counts is None:
        dist0 = np.hypot(
            xy_a[ai_idx, 0] - (xy_b[bi_idx, 0] + dx0),
            xy_a[ai_idx, 1] - (xy_b[bi_idx, 1] + dy0),
        )
        _, _, selected_cell_counts = _select_balanced_matches(
            xy_a,
            ai_idx,
            bi_idx,
            dist0,
            h=h_a,
            w=w_a,
            grid_x=int(args.balance_grid_x),
            grid_y=int(args.balance_grid_y),
            per_cell=10**9,
        )

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
        balance_grid_x=np.asarray([int(args.balance_grid_x)], dtype=np.int32),
        balance_grid_y=np.asarray([int(args.balance_grid_y)], dtype=np.int32),
        balance_per_cell=np.asarray([int(args.balance_per_cell)], dtype=np.int32),
        balance_enabled=np.asarray([0 if args.no_balance_matches else 1], dtype=np.int32),
        match_cell_counts=selected_cell_counts.astype(np.int32),
        residual_mean=np.asarray([np.mean(r)], dtype=np.float64),
        residual_rms=np.asarray([np.sqrt(np.mean(r**2))], dtype=np.float64),
        residual_max=np.asarray([np.max(r)], dtype=np.float64),
        a_stars=str(args.a_stars),
        b_stars=str(args.b_stars),
    )
    print(f"initial_shift_dxdy=({dx0:.3f},{dy0:.3f})")
    print(f"fit_degree={fit_degree}")
    print(f"matches_initial={len(ai_idx)}, matches_used={np.count_nonzero(keep)}")
    nz = selected_cell_counts[selected_cell_counts > 0]
    if len(nz) > 0:
        print(
            f"match_cell_counts_nonzero min/median/max="
            f"{int(np.min(nz))}/{int(np.median(nz))}/{int(np.max(nz))}"
        )
    else:
        print("match_cell_counts_nonzero min/median/max=0/0/0")
    print(f"fit_residual_mean={np.mean(r):.4f}px, rms={np.sqrt(np.mean(r**2)):.4f}px, max={np.max(r):.4f}px")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
