from pathlib import Path
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch

from alignment_common import build_matches, detect_stars, estimate_translation_from_stars


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank variable-star candidates from a FITS sequence using star-catalog matching."
    )
    parser.add_argument("--base", type=Path, required=True, help="Input directory containing FITS files.")
    parser.add_argument(
        "--pattern",
        nargs="+",
        default=["*.fits", "*.fit", "*.FITS", "*.FIT"],
        help="Glob patterns used to find FITS files.",
    )
    parser.add_argument("--ref", type=Path, default=None, help="Reference FITS path or filename (default: first match).")
    parser.add_argument("--out-csv", type=Path, default=None, help="Output ranking CSV path.")
    parser.add_argument("--out-png", type=Path, default=None, help="Output candidate scatter PNG path.")
    parser.add_argument("--max-stars", type=int, default=5000, help="Maximum stars detected per frame.")
    parser.add_argument("--match-radius", type=float, default=24.0, help="Star matching radius in pixels.")
    parser.add_argument("--min-observations", type=int, default=5, help="Minimum matched frames required per star.")
    parser.add_argument("--top-k", type=int, default=200, help="Top K candidates to highlight in the scatter plot.")
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


def robust_mad(x):
    med = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - med))


def save_candidate_scatter(ref_img, x, y, score, top_k, out_png: Path):
    finite = np.isfinite(ref_img)
    fill = np.nanmedian(ref_img[finite]) if np.any(finite) else 0.0
    view = np.where(finite, ref_img, fill)

    norm = ImageNormalize(view, interval=PercentileInterval(99.5), stretch=SqrtStretch())
    order = np.argsort(score)[::-1]
    keep = order[: min(top_k, len(order))]

    plt.figure(figsize=(12, 8))
    plt.imshow(view, origin="lower", cmap="gray", norm=norm)
    sc = plt.scatter(
        x[keep],
        y[keep],
        c=score[keep],
        s=24,
        cmap="turbo",
        alpha=0.9,
        edgecolors="white",
        linewidths=0.3,
    )
    plt.colorbar(sc, label="Variability score")
    plt.title(f"Top {len(keep)} variable candidates")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    args = parse_args()
    base = args.base

    inputs = list_inputs(base, args.pattern)
    if len(inputs) < 2:
        raise RuntimeError(f"Need at least 2 FITS files in {base}.")

    ref_path = resolve_path(base, args.ref) if args.ref is not None else inputs[0]
    if ref_path not in inputs:
        inputs = [ref_path] + [p for p in inputs if p != ref_path]
    targets = [p for p in inputs if p != ref_path]
    if len(targets) == 0:
        raise RuntimeError("No target files found after selecting reference.")

    out_csv = args.out_csv if args.out_csv is not None else (base / "variable_candidates_rank.csv")
    out_png = args.out_png if args.out_png is not None else (base / "variable_candidates_rank.png")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    ref_data = fits.getdata(ref_path).astype(float)
    xy_ref, flux_ref = detect_stars(ref_data, max_stars=int(args.max_stars))
    if len(xy_ref) == 0:
        raise RuntimeError(f"No stars detected in reference: {ref_path}")

    n_ref = len(xy_ref)
    # First column uses reference flux as baseline.
    measurements = [np.asarray(flux_ref, dtype=np.float64)]
    used_files = [ref_path]
    matched_counts = []
    failed_files = []

    for p in targets:
        try:
            data = fits.getdata(p).astype(float)
            xy_b, flux_b = detect_stars(data, max_stars=int(args.max_stars))
            if len(xy_b) == 0:
                raise RuntimeError("No stars detected.")

            dx0, dy0 = estimate_translation_from_stars(xy_ref, xy_b, top_n=300, bin_size=2.0)
            ai_idx, bi_idx = build_matches(xy_ref, xy_b, dx0, dy0, match_radius=float(args.match_radius))
            if len(ai_idx) < 10:
                raise RuntimeError(f"Too few matches ({len(ai_idx)}).")

            # Normalize frame-to-frame transparency/exposure scale with robust median ratio.
            ratio = np.asarray(flux_b[bi_idx], dtype=np.float64) / np.maximum(
                np.asarray(flux_ref[ai_idx], dtype=np.float64), 1e-12
            )
            scale = float(np.nanmedian(ratio))
            if not np.isfinite(scale) or scale <= 0.0:
                scale = 1.0

            vals = np.full(n_ref, np.nan, dtype=np.float64)
            vals[ai_idx] = np.asarray(flux_b[bi_idx], dtype=np.float64) / scale
            measurements.append(vals)
            used_files.append(p)
            matched_counts.append((p.name, len(ai_idx)))
        except Exception as exc:
            failed_files.append((p.name, str(exc)))

    flux_mat = np.vstack(measurements).T  # [n_ref, n_frames_used]
    n_obs = np.sum(np.isfinite(flux_mat), axis=1)
    med_flux = np.nanmedian(flux_mat, axis=1)

    valid = (n_obs >= int(args.min_observations)) & np.isfinite(med_flux) & (med_flux > 0.0)
    if np.count_nonzero(valid) == 0:
        raise RuntimeError("No stars meet min observation requirement.")

    rel = flux_mat / np.maximum(med_flux[:, None], 1e-12)
    mad_rel = np.array([robust_mad(row[np.isfinite(row)]) for row in rel], dtype=np.float64)
    p95 = np.nanpercentile(rel, 95, axis=1)
    p05 = np.nanpercentile(rel, 5, axis=1)
    amp_rel = p95 - p05

    # Composite variability score. Higher means more variable-like.
    score = mad_rel + 0.5 * amp_rel
    score[~valid] = np.nan

    idx = np.where(valid)[0]
    order = idx[np.argsort(score[idx])[::-1]]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "x",
                "y",
                "variability_score",
                "mad_rel",
                "amp_rel_p95_p05",
                "n_observations",
                "median_flux_norm",
            ]
        )
        for r, i in enumerate(order, start=1):
            writer.writerow(
                [
                    r,
                    f"{xy_ref[i, 0]:.4f}",
                    f"{xy_ref[i, 1]:.4f}",
                    f"{score[i]:.8f}",
                    f"{mad_rel[i]:.8f}",
                    f"{amp_rel[i]:.8f}",
                    int(n_obs[i]),
                    f"{med_flux[i]:.8f}",
                ]
            )

    save_candidate_scatter(
        ref_data,
        xy_ref[:, 0],
        xy_ref[:, 1],
        np.nan_to_num(score, nan=-1.0),
        int(args.top_k),
        out_png,
    )

    print(f"Reference: {ref_path}")
    print(f"Frames used: {len(used_files)}")
    print(f"Stars in reference: {n_ref}")
    print(f"Candidates ranked: {len(order)}")
    print(f"WROTE {out_csv}")
    print(f"WROTE {out_png}")

    if matched_counts:
        print("Match counts per frame:")
        for name, cnt in matched_counts:
            print(f"  {name}: {cnt}")
    if failed_files:
        print("Failed frames:")
        for name, reason in failed_files:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
