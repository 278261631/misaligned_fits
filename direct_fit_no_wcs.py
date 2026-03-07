from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch
from photutils.detection import DAOStarFinder
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree


def detect_stars(image, max_stars=5000, weak_signal=False):
    if weak_signal:
        detect_cfg = [
            (8.0, 3.0, 5.0, 3.0),
            (8.0, 3.0, 4.0, 2.8),
            (6.0, 3.0, 3.5, 2.6),
            (4.5, 3.0, 3.0, 2.4),
        ]
    else:
        detect_cfg = [(8.0, 3.0, 5.0, 3.0)]

    best_xy = np.empty((0, 2))
    best_flux = np.empty((0,))
    for hp_sigma, stats_sigma, thr_scale, fwhm in detect_cfg:
        hp = image - gaussian_filter(image, sigma=hp_sigma)
        _, med, std = sigma_clipped_stats(hp, sigma=stats_sigma, maxiters=10)
        finder = DAOStarFinder(fwhm=fwhm, threshold=max(thr_scale * std, 1e-6))
        src = finder(hp - med)
        if src is None or len(src) == 0:
            continue
        xy = np.column_stack([np.asarray(src["xcentroid"]), np.asarray(src["ycentroid"])])
        flux = np.asarray(src["flux"])
        if len(xy) > len(best_xy):
            best_xy = xy
            best_flux = flux

    if len(best_xy) == 0:
        return np.empty((0, 2))
    order = np.argsort(best_flux)[::-1]
    return best_xy[order][:max_stars]


def estimate_shift_fft(a, b):
    a_hp = a - gaussian_filter(a, sigma=12.0)
    b_hp = b - gaussian_filter(b, sigma=12.0)
    a_small = a_hp[::4, ::4]
    b_small = b_hp[::4, ::4]
    corr = fftconvolve(a_small, b_small[::-1, ::-1], mode="same")
    py, px = np.unravel_index(np.argmax(corr), corr.shape)
    cy, cx = np.array(corr.shape) // 2
    dy = (py - cy) * 4.0
    dx = (px - cx) * 4.0
    return dx, dy


def poly_terms(x, y, degree=3):
    terms = [np.ones_like(x)]
    for d in range(1, degree + 1):
        for i in range(d + 1):
            j = d - i
            terms.append((x**i) * (y**j))
    return np.vstack(terms).T


def robust_poly_fit(x_src, y_src, x_dst, y_dst, degree=3, n_iter=4, clip_sigma=2.8):
    X = poly_terms(x_src, y_src, degree=degree)
    keep = np.ones(len(x_src), dtype=bool)
    for _ in range(n_iter):
        A = X[keep]
        bx = x_dst[keep]
        by = y_dst[keep]
        cx, *_ = np.linalg.lstsq(A, bx, rcond=None)
        cy, *_ = np.linalg.lstsq(A, by, rcond=None)
        x_pred = X @ cx
        y_pred = X @ cy
        r = np.hypot(x_pred - x_dst, y_pred - y_dst)
        med = np.median(r)
        mad = np.median(np.abs(r - med)) + 1e-12
        sigma = 1.4826 * mad
        keep = r < (med + clip_sigma * sigma)
    return cx, cy, keep


def eval_poly(x, y, cx, cy, degree=3):
    X = poly_terms(x, y, degree=degree)
    return X @ cx, X @ cy


def n_poly_terms(degree):
    return (degree + 1) * (degree + 2) // 2


def fit_with_fallback(xa, ya, xb, yb, weak_signal=False):
    degrees = [3, 2, 1] if weak_signal else [3]
    n_iter = 6 if weak_signal else 5
    clip_sigma = 3.2 if weak_signal else 2.8

    for degree in degrees:
        min_points = max(20, n_poly_terms(degree) * 3)
        if len(xa) < min_points:
            continue
        cx, cy, keep = robust_poly_fit(xa, ya, xb, yb, degree=degree, n_iter=n_iter, clip_sigma=clip_sigma)
        if np.count_nonzero(keep) >= min_points:
            return cx, cy, keep, degree
    raise RuntimeError("Polynomial fit failed for all fallback degrees.")


def build_matches(xy_a, xy_b, dx0, dy0, match_radius=8.0):
    # initial translation: B -> A
    b_shift = xy_b + np.array([dx0, dy0])
    tree = cKDTree(xy_a)
    d, idx = tree.query(b_shift, distance_upper_bound=match_radius)
    good = np.isfinite(d) & (idx < len(xy_a))

    # unique by A index
    best = {}
    for bi, ai, dist in zip(np.where(good)[0], idx[good], d[good]):
        if (ai not in best) or (dist < best[ai][1]):
            best[ai] = (bi, dist)
    ai_idx = np.array(sorted(best.keys()), dtype=int)
    bi_idx = np.array([best[i][0] for i in ai_idx], dtype=int)
    return ai_idx, bi_idx


def save_png(img, path: Path, title: str):
    finite = np.isfinite(img)
    fill = np.nanmedian(img[finite]) if np.any(finite) else 0.0
    view = np.where(finite, img, fill)
    norm = ImageNormalize(view, interval=PercentileInterval(99.5), stretch=SqrtStretch())
    plt.figure(figsize=(10, 7))
    plt.imshow(view, origin="lower", cmap="gray", norm=norm)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def align_pair(a_path: Path, b_path: Path, outdir: Path, weak_signal=False):
    a_data = fits.getdata(a_path).astype(float)
    b_data = fits.getdata(b_path).astype(float)
    a_header = fits.getheader(a_path)

    max_stars = 9000 if weak_signal else 5000
    xy_a = detect_stars(a_data, max_stars=max_stars, weak_signal=weak_signal)
    xy_b = detect_stars(b_data, max_stars=max_stars, weak_signal=weak_signal)
    min_stars = 40 if weak_signal else 80
    if len(xy_a) < min_stars or len(xy_b) < min_stars:
        raise RuntimeError("Not enough stars detected for direct fit.")

    dx0, dy0 = estimate_shift_fft(a_data, b_data)
    radii = (8.0, 12.0, 16.0, 24.0, 32.0) if weak_signal else (8.0, 12.0, 16.0, 24.0)
    min_matches = 60 if weak_signal else 100
    ai_idx = bi_idx = np.array([], dtype=int)
    for radius in radii:
        ai_idx, bi_idx = build_matches(xy_a, xy_b, dx0, dy0, match_radius=radius)
        if len(ai_idx) >= min_matches:
            break
    if len(ai_idx) < min_matches:
        raise RuntimeError(f"Too few initial matches after retries: {len(ai_idx)}")

    xa, ya = xy_a[ai_idx, 0], xy_a[ai_idx, 1]
    xb, yb = xy_b[bi_idx, 0], xy_b[bi_idx, 1]

    # Fit inverse model A(x,y) -> B(x,y) for resampling.
    cx, cy, keep, fit_degree = fit_with_fallback(xa, ya, xb, yb, weak_signal=weak_signal)
    xa_k, ya_k = xa[keep], ya[keep]
    xb_k, yb_k = xb[keep], yb[keep]

    h, w = a_data.shape
    out = np.full((h, w), np.nan, dtype=np.float32)
    xx = np.arange(w, dtype=float)
    for y0 in range(0, h, 256):
        y1 = min(h, y0 + 256)
        yy = np.arange(y0, y1, dtype=float)
        gx, gy = np.meshgrid(xx, yy)
        src_x, src_y = eval_poly(gx.ravel(), gy.ravel(), cx, cy, degree=fit_degree)
        block = map_coordinates(
            b_data,
            [src_y.reshape(gx.shape), src_x.reshape(gx.shape)],
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )
        out[y0:y1, :] = block.astype(np.float32)

    out_fits = outdir / f"{b_path.stem}_on_{a_path.stem}_direct_poly{fit_degree}.fits"
    fits.writeto(out_fits, out, a_header, overwrite=True)

    preview_png = outdir / f"{b_path.stem}_on_{a_path.stem}_direct_poly{fit_degree}_preview.png"
    save_png(out, preview_png, "Direct fit (no WCS): B -> A")
    absdiff = np.abs(np.nan_to_num(a_data, nan=0.0) - np.nan_to_num(out, nan=0.0))
    absdiff_png = outdir / f"{b_path.stem}_on_{a_path.stem}_absdiff_direct_poly{fit_degree}.png"
    save_png(absdiff, absdiff_png, "Abs diff |A - B_direct|")

    px, py = eval_poly(xa_k, ya_k, cx, cy, degree=fit_degree)
    r = np.hypot(px - xb_k, py - yb_k)
    print(f"stars_A={len(xy_a)}, stars_B={len(xy_b)}")
    print(f"initial_shift_dxdy=({dx0:.3f},{dy0:.3f})")
    print(f"fit_degree={fit_degree}, weak_signal={weak_signal}")
    print(f"matches_initial={len(ai_idx)}, matches_used={np.count_nonzero(keep)}")
    print(f"fit_residual_mean={np.mean(r):.4f}px, rms={np.sqrt(np.mean(r**2)):.4f}px, max={np.max(r):.4f}px")
    print(f"WROTE {out_fits}")
    print(f"WROTE {preview_png}")
    print(f"WROTE {absdiff_png}")


def parse_args():
    parser = argparse.ArgumentParser(description="Direct no-WCS polynomial alignment for FITS/FT files.")
    parser.add_argument("--base", type=Path, default=Path(r"D:/missalign_fits/astap_sip"), help="Input directory.")
    parser.add_argument("--a", type=Path, default=None, help="Reference frame filename or path.")
    parser.add_argument("--b", type=Path, default=None, help="Target frame filename or path.")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory (default: <base>/direct_no_wcs).")
    parser.add_argument(
        "--pattern",
        nargs="+",
        default=["*.fits", "*.fit", "*.FITS", "*.FIT"],
        help="Glob patterns used in batch mode.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Align all matching files to the first matching frame (or --a if provided).",
    )
    parser.add_argument(
        "--weak-signal",
        action="store_true",
        help="Use weaker detection thresholds and polynomial fallback for low SNR data.",
    )
    return parser.parse_args()


def resolve_path(base: Path, maybe_path: Path | None):
    if maybe_path is None:
        return None
    if maybe_path.is_absolute():
        return maybe_path
    return base / maybe_path


def list_inputs(base: Path, patterns):
    files = []
    seen = set()
    for pattern in patterns:
        for p in sorted(base.glob(pattern)):
            if p.is_file() and p not in seen:
                seen.add(p)
                files.append(p)
    return files


def main():
    args = parse_args()
    base = args.base
    outdir = args.outdir if args.outdir is not None else (base / "direct_no_wcs")
    outdir.mkdir(parents=True, exist_ok=True)

    a_path = resolve_path(base, args.a)
    b_path = resolve_path(base, args.b)

    if args.batch:
        inputs = list_inputs(base, args.pattern)
        if len(inputs) < 2:
            raise RuntimeError(f"Need at least 2 input files in {base} for batch mode.")
        ref = a_path if a_path is not None else inputs[0]
        targets = [p for p in inputs if p != ref]
        if not targets:
            raise RuntimeError("Batch mode found no target files after selecting reference frame.")
        print(f"Reference: {ref}")
        print(f"Targets: {len(targets)}")
        ok = 0
        failed = 0
        for tgt in targets:
            print("-" * 72)
            print(f"Aligning target: {tgt.name}")
            try:
                align_pair(ref, tgt, outdir, weak_signal=args.weak_signal)
                ok += 1
            except Exception as exc:
                failed += 1
                print(f"FAILED {tgt.name}: {exc}")
        print("-" * 72)
        print(f"Batch done. success={ok}, failed={failed}")
        return

    if a_path is None or b_path is None:
        a_path = base / "K024-6_a.fits"
        b_path = base / "K024-6_b.fits"

    align_pair(a_path, b_path, outdir, weak_signal=args.weak_signal)


if __name__ == "__main__":
    main()
