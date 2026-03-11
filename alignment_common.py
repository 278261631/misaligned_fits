from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch
from photutils.detection import DAOStarFinder
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree


def detect_stars(image, max_stars=5000):
    hp = image - gaussian_filter(image, sigma=8.0)
    _, med, std = sigma_clipped_stats(hp, sigma=3.0, maxiters=10)
    finder = DAOStarFinder(fwhm=3.0, threshold=max(5.0 * std, 1e-6))
    src = finder(hp - med)
    if src is None or len(src) == 0:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)

    xy = np.column_stack([np.asarray(src["xcentroid"]), np.asarray(src["ycentroid"])])
    flux = np.asarray(src["flux"])
    order = np.argsort(flux)[::-1]
    order = order[:max_stars]
    return xy[order], flux[order]


def estimate_translation_from_stars(xy_a, xy_b, top_n=300, bin_size=2.0):
    if len(xy_a) == 0 or len(xy_b) == 0:
        raise RuntimeError("Empty star list.")

    a = xy_a[: min(top_n, len(xy_a))]
    b = xy_b[: min(top_n, len(xy_b))]

    dx = a[:, None, 0] - b[None, :, 0]
    dy = a[:, None, 1] - b[None, :, 1]
    dx1 = dx.ravel()
    dy1 = dy.ravel()
    if dx1.size == 0:
        raise RuntimeError("No pairs available to estimate translation.")

    dx_min, dx_max = float(np.min(dx1)), float(np.max(dx1))
    dy_min, dy_max = float(np.min(dy1)), float(np.max(dy1))
    xbins = np.arange(dx_min, dx_max + bin_size, bin_size)
    ybins = np.arange(dy_min, dy_max + bin_size, bin_size)
    if xbins.size < 2 or ybins.size < 2:
        return float(np.median(dx1)), float(np.median(dy1))

    hist, xedges, yedges = np.histogram2d(dx1, dy1, bins=(xbins, ybins))
    iy, ix = np.unravel_index(np.argmax(hist), hist.shape)
    dx0 = 0.5 * (xedges[iy] + xedges[iy + 1])
    dy0 = 0.5 * (yedges[ix] + yedges[ix + 1])
    return float(dx0), float(dy0)


def poly_terms(x, y, degree=3):
    terms = [np.ones_like(x)]
    for d in range(1, degree + 1):
        for i in range(d + 1):
            j = d - i
            terms.append((x**i) * (y**j))
    return np.vstack(terms).T


def robust_poly_fit(x_src, y_src, x_dst, y_dst, degree=3, n_iter=5, clip_sigma=2.8):
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


def n_poly_terms(degree):
    return (degree + 1) * (degree + 2) // 2


def fit_with_fallback(xa, ya, xb, yb):
    degrees = [3, 2, 1]
    for degree in degrees:
        min_points = max(20, n_poly_terms(degree) * 3)
        if len(xa) < min_points:
            continue
        cx, cy, keep = robust_poly_fit(xa, ya, xb, yb, degree=degree, n_iter=5, clip_sigma=2.8)
        if np.count_nonzero(keep) >= min_points:
            return cx, cy, keep, degree
    raise RuntimeError("Polynomial fit failed for all fallback degrees.")


def build_matches(xy_a, xy_b, dx0, dy0, match_radius=8.0):
    b_shift = xy_b + np.array([dx0, dy0])
    tree = cKDTree(xy_a)
    d, idx = tree.query(b_shift, distance_upper_bound=match_radius)
    good = np.isfinite(d) & (idx < len(xy_a))

    best = {}
    for bi, ai, dist in zip(np.where(good)[0], idx[good], d[good]):
        if (ai not in best) or (dist < best[ai][1]):
            best[ai] = (bi, dist)
    ai_idx = np.array(sorted(best.keys()), dtype=int)
    bi_idx = np.array([best[i][0] for i in ai_idx], dtype=int)
    return ai_idx, bi_idx


def eval_poly(x, y, cx, cy, degree=3):
    X = poly_terms(x, y, degree=degree)
    return X @ cx, X @ cy


def resample_b_to_a(a_data, b_data, cx, cy, fit_degree):
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
    return out


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
