from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch
from photutils.detection import DAOStarFinder
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree


def detect_stars(image, max_stars=5000):
    hp = image - gaussian_filter(image, sigma=8.0)
    _, med, std = sigma_clipped_stats(hp, sigma=3.0, maxiters=10)
    finder = DAOStarFinder(fwhm=3.0, threshold=max(5.0 * std, 1e-6))
    src = finder(hp - med)
    if src is None or len(src) == 0:
        return np.empty((0, 2))
    xy = np.column_stack([np.asarray(src["xcentroid"]), np.asarray(src["ycentroid"])])
    flux = np.asarray(src["flux"])
    order = np.argsort(flux)[::-1]
    return xy[order][:max_stars]


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


def main():
    base = Path(r"D:/missalign_fits/astap_sip")
    a_path = base / "K024-6_a.fits"
    b_path = base / "K024-6_b.fits"
    outdir = base / "direct_no_wcs"
    outdir.mkdir(parents=True, exist_ok=True)

    a_data = fits.getdata(a_path).astype(float)
    b_data = fits.getdata(b_path).astype(float)
    a_header = fits.getheader(a_path)

    xy_a = detect_stars(a_data, max_stars=5000)
    xy_b = detect_stars(b_data, max_stars=5000)
    if len(xy_a) < 200 or len(xy_b) < 200:
        raise RuntimeError("Not enough stars detected for direct fit.")

    dx0, dy0 = estimate_shift_fft(a_data, b_data)
    ai_idx, bi_idx = build_matches(xy_a, xy_b, dx0, dy0, match_radius=8.0)
    if len(ai_idx) < 300:
        raise RuntimeError(f"Too few initial matches: {len(ai_idx)}")

    xa, ya = xy_a[ai_idx, 0], xy_a[ai_idx, 1]
    xb, yb = xy_b[bi_idx, 0], xy_b[bi_idx, 1]

    # Fit inverse model A(x,y) -> B(x,y) for resampling.
    cx, cy, keep = robust_poly_fit(xa, ya, xb, yb, degree=3, n_iter=5, clip_sigma=2.8)
    xa_k, ya_k = xa[keep], ya[keep]
    xb_k, yb_k = xb[keep], yb[keep]

    h, w = a_data.shape
    out = np.full((h, w), np.nan, dtype=np.float32)
    xx = np.arange(w, dtype=float)
    for y0 in range(0, h, 256):
        y1 = min(h, y0 + 256)
        yy = np.arange(y0, y1, dtype=float)
        gx, gy = np.meshgrid(xx, yy)
        src_x, src_y = eval_poly(gx.ravel(), gy.ravel(), cx, cy, degree=3)
        block = map_coordinates(
            b_data,
            [src_y.reshape(gx.shape), src_x.reshape(gx.shape)],
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )
        out[y0:y1, :] = block.astype(np.float32)

    out_fits = outdir / "K024-6_b_on_a_direct_poly3.fits"
    fits.writeto(out_fits, out, a_header, overwrite=True)

    save_png(out, outdir / "K024-6_b_on_a_direct_poly3_preview.png", "Direct fit (no WCS): B -> A")
    absdiff = np.abs(np.nan_to_num(a_data, nan=0.0) - np.nan_to_num(out, nan=0.0))
    save_png(absdiff, outdir / "K024-6_absdiff_direct_poly3.png", "Abs diff |A - B_direct|")

    px, py = eval_poly(xa_k, ya_k, cx, cy, degree=3)
    r = np.hypot(px - xb_k, py - yb_k)
    print(f"stars_A={len(xy_a)}, stars_B={len(xy_b)}")
    print(f"initial_shift_dxdy=({dx0:.3f},{dy0:.3f})")
    print(f"matches_initial={len(ai_idx)}, matches_used={np.count_nonzero(keep)}")
    print(f"fit_residual_mean={np.mean(r):.4f}px, rms={np.sqrt(np.mean(r**2)):.4f}px, max={np.max(r):.4f}px")
    print(f"WROTE {out_fits}")
    print(f"WROTE {outdir / 'K024-6_b_on_a_direct_poly3_preview.png'}")
    print(f"WROTE {outdir / 'K024-6_absdiff_direct_poly3.png'}")


if __name__ == "__main__":
    main()
