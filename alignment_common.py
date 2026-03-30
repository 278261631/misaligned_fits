from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch
from photutils.detection import DAOStarFinder
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree


def _dao_detect_single(image_for_detection, fwhm=3.0, threshold_sigma=5.0):
    _, med, std = sigma_clipped_stats(image_for_detection, sigma=3.0, maxiters=10)
    if not np.isfinite(std) or float(std) <= 0.0:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)

    finder = DAOStarFinder(
        fwhm=float(fwhm),
        threshold=max(float(threshold_sigma) * float(std), 1e-6),
        brightest=None,
        peakmax=None,
    )
    src = finder(image_for_detection - med)
    if src is None or len(src) == 0:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)

    xy = np.column_stack([np.asarray(src["xcentroid"]), np.asarray(src["ycentroid"])])
    flux = np.asarray(src["flux"], dtype=float)
    ok = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1]) & np.isfinite(flux) & (flux > 0.0)
    if not np.any(ok):
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)
    return xy[ok], flux[ok]


def _dedupe_by_radius_keep_brightest(xy, flux, radius_px=2.5):
    if len(xy) == 0:
        return xy, flux
    order = np.argsort(flux)[::-1]
    xy_sorted = np.asarray(xy[order], dtype=float)
    flux_sorted = np.asarray(flux[order], dtype=float)

    cell_size = max(float(radius_px), 1e-6)
    r2 = float(radius_px) * float(radius_px)
    grid = {}
    keep_xy = []
    keep_flux = []

    for pt, f in zip(xy_sorted, flux_sorted):
        cx = int(np.floor(pt[0] / cell_size))
        cy = int(np.floor(pt[1] / cell_size))
        duplicate = False
        for ny in range(cy - 1, cy + 2):
            for nx in range(cx - 1, cx + 2):
                for qx, qy in grid.get((nx, ny), []):
                    dx = float(pt[0]) - float(qx)
                    dy = float(pt[1]) - float(qy)
                    if (dx * dx + dy * dy) <= r2:
                        duplicate = True
                        break
                if duplicate:
                    break
            if duplicate:
                break
        if duplicate:
            continue
        keep_xy.append((float(pt[0]), float(pt[1])))
        keep_flux.append(float(f))
        grid.setdefault((cx, cy), []).append((float(pt[0]), float(pt[1])))

    return np.asarray(keep_xy, dtype=float), np.asarray(keep_flux, dtype=float)


def detect_stars(image, max_stars=5000):
    arr = np.asarray(image, dtype=float)
    finite = np.isfinite(arr)
    fill = float(np.nanmedian(arr[finite])) if np.any(finite) else 0.0
    base = np.where(finite, arr, fill)
    hp = base - gaussian_filter(base, sigma=6.0)
    smooth = gaussian_filter(base, sigma=1.2)

    # Multi-scale detection: keep legacy high-pass pass and add bright-star-friendly passes.
    passes = [
        (hp, 3.0, 4.0),
        (smooth, 4.5, 3.8),
        (base, 6.0, 3.5),
    ]
    xy_chunks = []
    flux_chunks = []
    for detect_img, fwhm, thr_sigma in passes:
        xy_i, flux_i = _dao_detect_single(detect_img, fwhm=fwhm, threshold_sigma=thr_sigma)
        if len(xy_i) == 0:
            continue
        xy_chunks.append(xy_i)
        flux_chunks.append(flux_i)

    if len(xy_chunks) == 0:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)

    xy = np.vstack(xy_chunks)
    flux = np.concatenate(flux_chunks)
    xy, flux = _dedupe_by_radius_keep_brightest(xy, flux, radius_px=2.5)
    if len(xy) == 0:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)

    order = np.argsort(flux)[::-1]
    if max_stars is not None and int(max_stars) > 0:
        order = order[: int(max_stars)]
    return xy[order], flux[order]


def select_stars_uniform_grid(xy, flux, height, width, grid_x=7, grid_y=7, per_cell=80, max_total=5000):
    xy = np.asarray(xy, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if len(xy) == 0 or len(flux) == 0:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)
    if xy.shape[0] != flux.shape[0]:
        raise ValueError(f"xy/flux length mismatch: {xy.shape[0]} vs {flux.shape[0]}")

    gx = max(int(grid_x), 1)
    gy = max(int(grid_y), 1)
    pc = max(int(per_cell), 1)
    h = max(int(height), 1)
    w = max(int(width), 1)

    x = np.clip(xy[:, 0], 0.0, w - 1e-6)
    y = np.clip(xy[:, 1], 0.0, h - 1e-6)
    ix = np.minimum((x / w * gx).astype(int), gx - 1)
    iy = np.minimum((y / h * gy).astype(int), gy - 1)
    cell_id = iy * gx + ix

    keep_idx = []
    for cid in range(gx * gy):
        idx = np.where(cell_id == cid)[0]
        if len(idx) == 0:
            continue
        loc = idx[np.argsort(flux[idx])[::-1]]
        keep_idx.append(loc[: min(pc, len(loc))])

    if len(keep_idx) == 0:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)

    sel = np.concatenate(keep_idx)
    if max_total is not None and int(max_total) > 0 and len(sel) > int(max_total):
        sel = sel[np.argsort(flux[sel])[::-1][: int(max_total)]]

    # Keep output stably ordered by brightness.
    sel = sel[np.argsort(flux[sel])[::-1]]
    return xy[sel], flux[sel]


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
