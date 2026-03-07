from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch
from scipy.ndimage import gaussian_filter

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "This script requires OpenCV. Install with: pip install opencv-python"
    ) from exc


def safe_float_image(img):
    arr = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(arr)
    fill = np.nanmedian(arr[finite]) if np.any(finite) else 0.0
    return np.where(finite, arr, fill).astype(np.float32)


def preprocess_for_registration(img):
    arr = safe_float_image(img)
    hp = arr - gaussian_filter(arr, sigma=8.0).astype(np.float32)
    mean, med, std = sigma_clipped_stats(hp, sigma=3.0, maxiters=10)
    _ = mean  # keep explicit for clarity
    scale = max(float(std), 1e-6)
    norm = (hp - float(med)) / scale
    return np.ascontiguousarray(norm.astype(np.float32))


def warp_image_for_eval(img, warp_matrix, out_shape):
    h, w = out_shape
    return cv2.warpAffine(
        img,
        warp_matrix.astype(np.float32),
        (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


def overlap_corrcoef(a, b):
    a1 = np.asarray(a, dtype=np.float32).ravel()
    b1 = np.asarray(b, dtype=np.float32).ravel()
    if a1.size == 0 or b1.size == 0:
        return -1.0
    am = float(np.mean(a1))
    bm = float(np.mean(b1))
    av = a1 - am
    bv = b1 - bm
    denom = float(np.sqrt(np.sum(av * av) * np.sum(bv * bv)) + 1e-12)
    return float(np.sum(av * bv) / denom)


def estimate_translation_phase(template_img, moving_img):
    hann = cv2.createHanningWindow((template_img.shape[1], template_img.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(template_img, moving_img, hann)

    m_plus = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    m_minus = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)

    warped_plus = warp_image_for_eval(moving_img, m_plus, template_img.shape)
    warped_minus = warp_image_for_eval(moving_img, m_minus, template_img.shape)
    score_plus = overlap_corrcoef(template_img, warped_plus)
    score_minus = overlap_corrcoef(template_img, warped_minus)
    if score_minus > score_plus:
        return m_minus, (-dx, -dy), score_minus
    return m_plus, (dx, dy), score_plus


def refine_affine_ecc(template_img, moving_img, init_warp, iterations=300, eps=1e-6):
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(iterations),
        float(eps),
    )
    warp = init_warp.astype(np.float32).copy()
    cc, warp = cv2.findTransformECC(
        template_img,
        moving_img,
        warp,
        cv2.MOTION_AFFINE,
        criteria,
        None,
        5,
    )
    return float(cc), warp


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


def write_preprocessed_outputs(a_path: Path, b_path: Path, outdir: Path, a_reg, b_reg):
    a_reg_fits = outdir / f"{a_path.stem}_preprocessed_for_registration.fits"
    b_reg_fits = outdir / f"{b_path.stem}_preprocessed_for_registration.fits"
    fits.writeto(a_reg_fits, np.asarray(a_reg, dtype=np.float32), overwrite=True)
    fits.writeto(b_reg_fits, np.asarray(b_reg, dtype=np.float32), overwrite=True)

    a_reg_png = outdir / f"{a_path.stem}_preprocessed_for_registration_preview.png"
    b_reg_png = outdir / f"{b_path.stem}_preprocessed_for_registration_preview.png"
    save_png(a_reg, a_reg_png, f"Preprocessed for registration: {a_path.name}")
    save_png(b_reg, b_reg_png, f"Preprocessed for registration: {b_path.name}")
    return a_reg_fits, b_reg_fits, a_reg_png, b_reg_png


def align_pair(a_path: Path, b_path: Path, outdir: Path, phase_only=False, ecc_iterations=300, ecc_eps=1e-6):
    a_data = fits.getdata(a_path).astype(np.float32)
    b_data = fits.getdata(b_path).astype(np.float32)
    a_header = fits.getheader(a_path)
    h, w = a_data.shape

    a_reg = preprocess_for_registration(a_data)
    b_reg = preprocess_for_registration(b_data)
    a_reg_fits, b_reg_fits, a_reg_png, b_reg_png = write_preprocessed_outputs(a_path, b_path, outdir, a_reg, b_reg)

    phase_warp, (dx, dy), phase_score = estimate_translation_phase(a_reg, b_reg)
    used_method = "phase"
    ecc_cc = np.nan
    final_warp = phase_warp

    if not phase_only:
        try:
            ecc_cc, final_warp = refine_affine_ecc(
                a_reg, b_reg, phase_warp, iterations=ecc_iterations, eps=ecc_eps
            )
            used_method = "ecc_affine"
        except cv2.error as exc:
            used_method = f"phase_fallback({exc.__class__.__name__})"
            final_warp = phase_warp

    out = cv2.warpAffine(
        safe_float_image(b_data),
        final_warp.astype(np.float32),
        (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    ).astype(np.float32)

    method_tag = "phase" if phase_only else "ecc_phase"
    out_fits = outdir / f"{b_path.stem}_on_{a_path.stem}_{method_tag}.fits"
    fits.writeto(out_fits, out, a_header, overwrite=True)

    preview_png = outdir / f"{b_path.stem}_on_{a_path.stem}_{method_tag}_preview.png"
    save_png(out, preview_png, f"{method_tag}: B -> A")
    absdiff = np.abs(np.nan_to_num(a_data, nan=0.0) - np.nan_to_num(out, nan=0.0))
    absdiff_png = outdir / f"{b_path.stem}_on_{a_path.stem}_{method_tag}_absdiff.png"
    save_png(absdiff, absdiff_png, f"Abs diff |A - B_{method_tag}|")

    print(f"method_used={used_method}")
    print(f"phase_shift_dxdy=({dx:.3f},{dy:.3f}), phase_corr={phase_score:.6f}")
    if not np.isnan(ecc_cc):
        print(f"ecc_cc={ecc_cc:.6f}")
    print(
        "affine_matrix="
        f"[[{final_warp[0,0]:.6f},{final_warp[0,1]:.6f},{final_warp[0,2]:.6f}],"
        f"[{final_warp[1,0]:.6f},{final_warp[1,1]:.6f},{final_warp[1,2]:.6f}]]"
    )
    print(f"WROTE {out_fits}")
    print(f"WROTE {preview_png}")
    print(f"WROTE {absdiff_png}")
    print(f"WROTE {a_reg_fits}")
    print(f"WROTE {b_reg_fits}")
    print(f"WROTE {a_reg_png}")
    print(f"WROTE {b_reg_png}")


def resolve_path(base: Path, maybe_path):
    if maybe_path is None:
        return None
    p = Path(maybe_path)
    if p.is_absolute():
        return p
    return base / p


def list_inputs(base: Path, patterns):
    files = []
    seen = set()
    for pattern in patterns:
        for p in sorted(base.glob(pattern)):
            if p.is_file() and p not in seen:
                seen.add(p)
                files.append(p)
    return files


def parse_args():
    parser = argparse.ArgumentParser(
        description="FITS alignment using phase correlation + ECC affine (no star matching)."
    )
    parser.add_argument("--base", type=Path, default=Path(r"E:/github/test_align"), help="Input directory.")
    parser.add_argument("--a", type=Path, default=None, help="Reference frame filename or path.")
    parser.add_argument("--b", type=Path, default=None, help="Target frame filename or path.")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory (default: <base>/ecc_phase).")
    parser.add_argument(
        "--pattern",
        nargs="+",
        default=["*.fit", "*.fits", "*.FIT", "*.FITS"],
        help="Glob patterns used in batch mode.",
    )
    parser.add_argument("--batch", action="store_true", help="Align all matching files to the reference frame.")
    parser.add_argument("--phase-only", action="store_true", help="Only run phase-correlation translation.")
    parser.add_argument("--ecc-iterations", type=int, default=300, help="ECC max iteration count.")
    parser.add_argument("--ecc-eps", type=float, default=1e-6, help="ECC convergence epsilon.")
    return parser.parse_args()


def main():
    args = parse_args()
    base = args.base
    outdir = args.outdir if args.outdir is not None else (base / "ecc_phase")
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
                align_pair(
                    ref,
                    tgt,
                    outdir,
                    phase_only=args.phase_only,
                    ecc_iterations=args.ecc_iterations,
                    ecc_eps=args.ecc_eps,
                )
                ok += 1
            except Exception as exc:
                failed += 1
                print(f"FAILED {tgt.name}: {exc}")
        print("-" * 72)
        print(f"Batch done. success={ok}, failed={failed}")
        return

    if a_path is None or b_path is None:
        raise RuntimeError("Single-pair mode requires both --a and --b.")

    align_pair(
        a_path,
        b_path,
        outdir,
        phase_only=args.phase_only,
        ecc_iterations=args.ecc_iterations,
        ecc_eps=args.ecc_eps,
    )


if __name__ == "__main__":
    main()
