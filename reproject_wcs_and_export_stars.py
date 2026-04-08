from pathlib import Path
import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates, median_filter

from alignment_common import detect_stars, select_stars_uniform_grid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Median-denoise FITS B, reproject to FITS A by WCS, then export projected FITS and stars."
    )
    parser.add_argument("--a", type=Path, required=True, help="Reference FITS A path (target WCS/grid).")
    parser.add_argument("--b", type=Path, required=True, help="Input FITS B path (source WCS/grid).")
    parser.add_argument("--out-fits", type=Path, required=True, help="Output reprojected FITS path.")
    parser.add_argument("--out-stars", type=Path, required=True, help="Output projected stars file (.npz) for alignment.")
    parser.add_argument("--out-stars-all", type=Path, default=None, help="Output projected all-stars file (.npz).")
    parser.add_argument(
        "--out-stars-all-png",
        type=Path,
        default=None,
        help="Optional PNG path to visualize reprojected image with stars from .all.npz. Skip when not provided.",
    )
    parser.add_argument(
        "--out-timing-png",
        type=Path,
        default=None,
        help="Output timing timeline PNG path (default: next to --out-fits with .timing.png suffix).",
    )
    parser.add_argument(
        "--all-png-stretch",
        choices=("none", "normal", "strong"),
        default="strong",
        help="Contrast stretch mode for --out-stars-all-png (default: strong).",
    )
    parser.add_argument(
        "--all-png-gamma",
        type=float,
        default=0.45,
        help="Gamma used after percentile stretch for --out-stars-all-png (default: 0.45).",
    )
    parser.add_argument(
        "--all-png-min-flux-percentile",
        type=float,
        default=95.0,
        help="Only stars above this flux percentile are marked in --out-stars-all-png (default: 95).",
    )
    parser.add_argument(
        "--min-flux",
        type=float,
        default=5.0,
        help="Absolute lower bound for exported star flux (keep flux >= this value, default: 5).",
    )
    parser.add_argument(
        "--min-flux-percentile",
        type=float,
        default=40.0,
        help="Percentile-based lower bound for exported star flux (0-100, default: 40).",
    )
    parser.add_argument(
        "--median-size",
        type=int,
        default=3,
        help="Median filter kernel size applied to B before reprojection (odd integer recommended).",
    )
    parser.add_argument("--max-stars", type=int, default=5000, help="Maximum stars to keep in output stars file.")
    parser.add_argument("--uniform-grid-x", type=int, default=7, help="Grid columns for uniform star selection.")
    parser.add_argument("--uniform-grid-y", type=int, default=7, help="Grid rows for uniform star selection.")
    parser.add_argument(
        "--uniform-per-cell",
        type=int,
        default=80,
        help="Maximum stars selected per grid cell in --out-stars.",
    )
    parser.add_argument(
        "--no-uniform-selection",
        action="store_true",
        help="Disable uniform-grid selection for --out-stars and use global brightness ranking.",
    )
    parser.add_argument("--chunk-rows", type=int, default=512, help="Rows per block during reprojection.")
    parser.add_argument(
        "--skip-median-filter",
        action="store_true",
        help="Skip median filtering and reproject raw B data directly.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs; default skips when all expected outputs already exist.",
    )
    return parser.parse_args()


def _safe_for_detection(img):
    arr = np.asarray(img, dtype=float)
    finite = np.isfinite(arr)
    fill = float(np.nanmedian(arr[finite])) if np.any(finite) else 0.0
    return np.where(finite, arr, fill)


def _resolve_mjd_from_b_header(b_header):
    for k in ("JD", "JD-OBS", "JD_OBS", "JDAVG", "JD-AVG"):
        if k not in b_header:
            continue
        try:
            jd = float(b_header[k])
            return float(jd - 2400000.5), k
        except Exception:
            continue
    return None, None


def reproject_b_to_a_wcs(a_data, b_data, wcs_a: WCS, wcs_b: WCS, chunk_rows=256):
    h, w = a_data.shape
    out = np.full((h, w), np.nan, dtype=np.float32)
    xx = np.arange(w, dtype=float)

    for y0 in range(0, h, int(chunk_rows)):
        y1 = min(h, y0 + int(chunk_rows))
        yy = np.arange(y0, y1, dtype=float)
        gx, gy = np.meshgrid(xx, yy)

        lon, lat = wcs_a.pixel_to_world_values(gx.ravel(), gy.ravel())
        src_x, src_y = wcs_b.world_to_pixel_values(lon, lat)
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


def _stretch_image_for_display(image_data, mode: str, gamma: float):
    finite = np.isfinite(image_data)
    fill = np.nanmedian(image_data[finite]) if np.any(finite) else 0.0
    view = np.where(finite, image_data, fill).astype(np.float64, copy=False)
    if mode == "none":
        return view

    finite_vals = view[np.isfinite(view)]
    if finite_vals.size == 0:
        return view

    if mode == "strong":
        p_low, p_high = 0.5, 99.8
    else:
        p_low, p_high = 1.0, 99.5
    vmin = float(np.percentile(finite_vals, p_low))
    vmax = float(np.percentile(finite_vals, p_high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return view

    clipped = np.clip(view, vmin, vmax)
    norm = (clipped - vmin) / (vmax - vmin)
    safe_gamma = float(gamma) if float(gamma) > 0 else 1.0
    return np.power(norm, safe_gamma)


def export_all_stars_png(
    image_data,
    xy_all,
    flux_all,
    out_png: Path | None,
    stretch_mode: str,
    stretch_gamma: float,
    min_flux_percentile: float,
):
    if out_png is None:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    view = _stretch_image_for_display(image_data, stretch_mode, stretch_gamma)
    ax.imshow(view, origin="lower", cmap="gray", interpolation="nearest")

    flux = np.asarray(flux_all, dtype=np.float64)
    p = float(np.clip(min_flux_percentile, 0.0, 100.0))
    thr = float(np.percentile(flux, p)) if flux.size > 0 else np.inf
    keep = flux > thr
    xy_draw = xy_all[keep] if len(xy_all) == len(flux) else xy_all
    if len(xy_draw) > 0:
        ax.scatter(
            xy_draw[:, 0],
            xy_draw[:, 1],
            s=10,
            marker="o",
            facecolors="none",
            edgecolors="#FFD400",
            linewidths=0.6,
            alpha=0.9,
        )
    ax.set_title(
        f"Stars with flux > P{p:g} ({thr:.3g}): {len(xy_draw)} / {len(xy_all)} "
        f"(stretch={stretch_mode}, gamma={float(stretch_gamma):.2f})"
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"WROTE {out_png}")


def export_timing_timeline_png(stages, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if len(stages) == 0:
        stages = [{"name": "script_total", "start_s": 0.0, "dur_s": 0.0}]

    labels = [str(s["name"]) for s in stages]
    starts = np.asarray([float(s["start_s"]) for s in stages], dtype=np.float64)
    durs = np.asarray([max(float(s["dur_s"]), 0.0) for s in stages], dtype=np.float64)
    y = np.arange(len(stages), dtype=float)

    total_s = float(np.max(starts + durs)) if len(stages) > 0 else 0.0
    x_max = max(total_s * 1.05, 0.1)

    fig_h = max(3.0, 0.5 * len(stages) + 1.5)
    fig = plt.figure(figsize=(12, fig_h))
    ax = fig.add_subplot(111)
    ax.barh(y, durs, left=starts, height=0.65, color="#4C78A8", edgecolor="#2F4B7C", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, x_max)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_title(f"reproject_wcs_and_export_stars timing timeline (total={total_s:.3f}s)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    for yi, st, du in zip(y, starts, durs):
        ax.text(st + du + max(x_max * 0.005, 0.005), yi, f"{du:.3f}s", va="center", ha="left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"WROTE {out_png}")


def export_timing_jsonl(stages, out_jsonl: Path):
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for i, s in enumerate(stages):
            start_s = float(s.get("start_s", 0.0))
            dur_s = max(float(s.get("dur_s", 0.0)), 0.0)
            payload = {
                "seq": int(i),
                "step": str(s.get("name", "unknown")),
                "start_s": start_s,
                "duration_s": dur_s,
                "end_s": start_s + dur_s,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"WROTE {out_jsonl}")


def main():
    args = parse_args()
    timing_png = (
        args.out_timing_png
        if args.out_timing_png is not None
        else args.out_fits.with_name(f"{args.out_fits.stem}.timing.png")
    )
    timing_jsonl = timing_png.with_suffix(".jsonl")
    t_script0 = time.perf_counter()
    stages = []

    def _run_stage(name, fn):
        t0 = time.perf_counter()
        ret = fn()
        t1 = time.perf_counter()
        stages.append({"name": str(name), "start_s": float(t0 - t_script0), "dur_s": float(t1 - t0)})
        return ret

    out_stars_all = (
        args.out_stars_all
        if args.out_stars_all is not None
        else args.out_stars.with_name(f"{args.out_stars.stem}.all.npz")
    )
    expected_outputs = [args.out_fits, args.out_stars, out_stars_all]
    if args.out_stars_all_png is not None:
        expected_outputs.append(args.out_stars_all_png)
    expected_outputs.append(timing_png)
    expected_outputs.append(timing_jsonl)
    if (not args.overwrite) and all(p.exists() for p in expected_outputs):
        print("SKIP reproject_wcs_and_export_stars.py: outputs already exist (use --overwrite to regenerate)")
        for p in expected_outputs:
            print(f"EXISTS {p}")
        return

    def _load_inputs():
        a_data_local = fits.getdata(args.a).astype(np.float32)
        b_data_local = fits.getdata(args.b).astype(np.float32)
        a_header_local = fits.getheader(args.a)
        b_header_local = fits.getheader(args.b)
        return a_data_local, b_data_local, a_header_local, b_header_local

    a_data, b_data, a_header, b_header = _run_stage("load_fits_and_headers", _load_inputs)
    h, w = a_data.shape

    def _build_wcs():
        return WCS(a_header).celestial, WCS(b_header).celestial

    wcs_a, wcs_b = _run_stage("build_wcs", _build_wcs)

    if args.skip_median_filter:
        b_input = b_data
        stages.append({"name": "median_filter(skipped)", "start_s": float(time.perf_counter() - t_script0), "dur_s": 0.0})
    else:
        b_input = _run_stage("median_filter", lambda: median_filter(b_data, size=int(args.median_size)))
    out = _run_stage(
        "reproject_to_a_wcs",
        lambda: reproject_b_to_a_wcs(a_data, b_input, wcs_a, wcs_b, chunk_rows=args.chunk_rows),
    )
    _run_stage("prepare_output_dirs", lambda: (args.out_fits.parent.mkdir(parents=True, exist_ok=True), args.out_stars.parent.mkdir(parents=True, exist_ok=True), out_stars_all.parent.mkdir(parents=True, exist_ok=True), timing_png.parent.mkdir(parents=True, exist_ok=True)))

    out_header = a_header.copy()
    mjd_from_b, jd_key = _resolve_mjd_from_b_header(b_header)
    if mjd_from_b is not None:
        out_header["MJD"] = float(mjd_from_b)
        out_header["MJD-OBS"] = float(mjd_from_b)
        out_header["HIERARCH SRC_B_JD_KEY"] = str(jd_key)
    else:
        print("WARNING: No JD key found in --b header; output MJD keeps reference header value.")
    _run_stage("write_projected_fits", lambda: fits.writeto(args.out_fits, out, out_header, overwrite=True))

    detect_img = _run_stage("prepare_detection_image", lambda: _safe_for_detection(out))
    t_detect0 = time.perf_counter()
    xy_all, flux_all, detect_debug = detect_stars(detect_img, max_stars=0, return_debug=True)
    t_detect1 = time.perf_counter()
    detect_start = float(t_detect0 - t_script0)
    stages.append({"name": "detect_prefilter", "start_s": detect_start, "dur_s": float(detect_debug.get("prefilter_duration_s", 0.0))})
    cursor_s = detect_start + float(detect_debug.get("prefilter_duration_s", 0.0))
    for p in detect_debug.get("passes", []):
        d = float(p.get("duration_s", 0.0))
        n = int(p.get("raw_count", 0))
        stages.append({"name": f"{p.get('name', 'detect_pass')}[{n}]", "start_s": cursor_s, "dur_s": d})
        cursor_s += d
    merge_dur = float(detect_debug.get("merge_duration_s", 0.0))
    stages.append({"name": "detect_merge_candidates", "start_s": cursor_s, "dur_s": merge_dur})
    cursor_s += merge_dur
    dedupe_dur = float(detect_debug.get("dedupe_duration_s", 0.0))
    stages.append({"name": "detect_dedupe", "start_s": cursor_s, "dur_s": dedupe_dur})
    cursor_s += dedupe_dur
    sort_dur = float(detect_debug.get("sort_duration_s", 0.0))
    stages.append({"name": "detect_sort_limit", "start_s": cursor_s, "dur_s": sort_dur})
    accounted = (
        float(detect_debug.get("prefilter_duration_s", 0.0))
        + float(sum(float(p.get("duration_s", 0.0)) for p in detect_debug.get("passes", [])))
        + merge_dur
        + dedupe_dur
        + sort_dur
    )
    remain = max(float(t_detect1 - t_detect0) - accounted, 0.0)
    if remain > 1e-6:
        stages.append({"name": "detect_overhead", "start_s": cursor_s + sort_dur, "dur_s": remain})
    stars_detected_raw = int(len(xy_all))
    if stars_detected_raw == 0:
        raise RuntimeError("No stars detected from reprojected image.")

    flux_thr_abs = float(args.min_flux) if args.min_flux is not None else None
    flux_thr_pct = None
    flux_thr_pct_value = None
    flux_filter_threshold = None
    if args.min_flux_percentile is not None:
        flux_thr_pct = float(np.clip(float(args.min_flux_percentile), 0.0, 100.0))
        flux_thr_pct_value = float(np.percentile(flux_all, flux_thr_pct))

    thr_candidates = []
    if flux_thr_abs is not None:
        thr_candidates.append(float(flux_thr_abs))
    if flux_thr_pct_value is not None:
        thr_candidates.append(float(flux_thr_pct_value))
    if len(thr_candidates) > 0:
        flux_filter_threshold = max(thr_candidates)
        def _apply_flux_filter():
            keep_local = np.asarray(flux_all, dtype=np.float64) >= float(flux_filter_threshold)
            return xy_all[keep_local], flux_all[keep_local]

        xy_all, flux_all = _run_stage("flux_filter", _apply_flux_filter)
        if len(xy_all) == 0:
            raise RuntimeError(
                "No stars remain after flux filtering. "
                f"min_flux={flux_thr_abs}, min_flux_percentile={flux_thr_pct}, "
                f"applied_threshold={float(flux_filter_threshold):.6g}"
            )

    if args.no_uniform_selection:
        def _select_global_brightest():
            order_local = np.argsort(flux_all)[::-1]
            if int(args.max_stars) > 0:
                order_local = order_local[: int(args.max_stars)]
            return xy_all[order_local], flux_all[order_local]

        xy_align, flux_align = _run_stage("select_stars_global_brightness", _select_global_brightest)
    else:
        xy_align, flux_align = _run_stage(
            "select_stars_uniform_grid",
            lambda: select_stars_uniform_grid(
                xy_all,
                flux_all,
                height=int(h),
                width=int(w),
                grid_x=int(args.uniform_grid_x),
                grid_y=int(args.uniform_grid_y),
                per_cell=int(args.uniform_per_cell),
                max_total=int(args.max_stars),
            ),
        )
    if len(xy_align) == 0:
        raise RuntimeError("No stars selected for alignment from reprojected image.")

    _run_stage(
        "write_alignment_stars_npz",
        lambda: np.savez_compressed(
            args.out_stars,
            xy=xy_align.astype(np.float32),
            flux=flux_align.astype(np.float32),
            max_stars=int(args.max_stars),
            source_fits=str(args.b),
            reference_fits=str(args.a),
            projected_fits=str(args.out_fits),
            height=int(h),
            width=int(w),
        ),
    )
    _run_stage(
        "write_all_stars_npz",
        lambda: np.savez_compressed(
            out_stars_all,
            xy=xy_all.astype(np.float32),
            flux=flux_all.astype(np.float32),
            source_fits=str(args.b),
            reference_fits=str(args.a),
            projected_fits=str(args.out_fits),
            height=int(h),
            width=int(w),
        ),
    )
    _run_stage(
        "export_all_stars_png",
        lambda: export_all_stars_png(
            out,
            xy_all,
            flux_all,
            args.out_stars_all_png,
            args.all_png_stretch,
            args.all_png_gamma,
            args.all_png_min_flux_percentile,
        ),
    )
    _run_stage("write_timing_timeline_png", lambda: export_timing_timeline_png(stages, timing_png))
    export_timing_jsonl(stages, timing_jsonl)

    if args.skip_median_filter:
        print("median_filter=skipped")
    else:
        print(f"median_size={int(args.median_size)}")
    print(f"stars_align={len(xy_align)}")
    print(f"stars_all={len(xy_all)}")
    print(f"stars_detected_raw={stars_detected_raw}")
    if flux_filter_threshold is None:
        print("flux_filter=disabled")
    else:
        print(
            "flux_filter=enabled "
            f"min_flux={flux_thr_abs} "
            f"min_flux_percentile={flux_thr_pct} "
            f"threshold={float(flux_filter_threshold):.6g} "
            f"kept={len(xy_all)}/{stars_detected_raw}"
        )
    if args.no_uniform_selection:
        print("uniform_selection=disabled")
    else:
        print(
            f"uniform_selection=enabled grid={int(args.uniform_grid_x)}x{int(args.uniform_grid_y)} "
            f"per_cell={int(args.uniform_per_cell)}"
        )
    print(f"WROTE {args.out_fits}")
    print(f"WROTE {args.out_stars}")
    print(f"WROTE {out_stars_all}")
    print(f"WROTE {timing_png}")
    print(f"WROTE {timing_jsonl}")


if __name__ == "__main__":
    main()
