from pathlib import Path
import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import requests
from astropy.io import fits
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch
from astropy.wcs import WCS


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cross-match variable_candidates_nonref_only_inner_border.csv with HIP/Variable/MPC "
            "services using a 15-pixel cone radius by default."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("variable_candidates_nonref_only_inner_border.csv"),
        help="Input candidate CSV path.",
    )
    parser.add_argument("--radius-px", type=float, default=15.0, help="Cross-match radius in pixels.")
    parser.add_argument(
        "--arcsec-per-px-fallback",
        type=float,
        default=None,
        help="Fallback arcsec/pixel when arcsec_per_px_mean is missing in a row.",
    )
    parser.add_argument(
        "--find-hip-csv",
        type=Path,
        default=None,
        help="Output matched HIP rows CSV path (default: <input_dir>/find_hip.csv).",
    )
    parser.add_argument(
        "--find-variable-csv",
        type=Path,
        default=None,
        help="Output matched variable-star rows CSV path (default: <input_dir>/find_variable.csv).",
    )
    parser.add_argument(
        "--find-mpc-csv",
        type=Path,
        default=None,
        help="Output matched MPC rows CSV path (default: <input_dir>/find_mpc.csv).",
    )
    parser.add_argument("--hip-host", default="127.0.0.1", help="HIP server host.")
    parser.add_argument("--hip-port", type=int, default=5002, help="HIP server port.")
    parser.add_argument("--var-host", default="127.0.0.1", help="Variable server host.")
    parser.add_argument("--var-port", type=int, default=5000, help="Variable server port.")
    parser.add_argument("--mpc-host", default="127.0.0.1", help="MPC server host.")
    parser.add_argument("--mpc-port", type=int, default=5001, help="MPC server port.")
    parser.add_argument(
        "--ref-fits",
        type=Path,
        default=None,
        help="Reference FITS used to convert matched RA/DEC back to pixel x/y.",
    )
    parser.add_argument("--timeout-sec", type=float, default=8.0, help="HTTP timeout in seconds.")
    return parser.parse_args()


def parse_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def request_search(url, params, timeout_sec):
    r = requests.get(url, params=params, timeout=timeout_sec)
    r.raise_for_status()
    return r.json()


def write_rows_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_reference_wcs(ref_fits: Path | None, input_csv: Path):
    if ref_fits is None:
        return None, None
    p = ref_fits
    if not p.is_absolute():
        p = input_csv.parent / p
    if not p.exists():
        return None, None
    try:
        return WCS(fits.getheader(p)).celestial, p
    except Exception:
        return None, None


def resolve_ref_fits_path(ref_fits: Path | None, input_csv: Path):
    if ref_fits is None:
        return None
    p = ref_fits
    if not p.is_absolute():
        p = input_csv.parent / p
    if not p.exists():
        return None
    return p


def extract_item_ra_dec(item):
    keys = {str(k).lower(): k for k in item.keys()}
    ra_key = None
    dec_key = None
    for k in ("ra_deg", "ra", "raj2000", "raj2000_deg", "raj"):
        if k in keys:
            ra_key = keys[k]
            break
    for k in ("dec_deg", "dec", "dej2000", "dej2000_deg", "dej"):
        if k in keys:
            dec_key = keys[k]
            break
    if ra_key is None or dec_key is None:
        return None, None
    ra = parse_float(item.get(ra_key))
    dec = parse_float(item.get(dec_key))
    if ra is None or dec is None:
        return None, None
    return float(ra), float(dec)


def world_to_pixel_xy(ref_wcs, ra_deg, dec_deg):
    if ref_wcs is None or ra_deg is None or dec_deg is None:
        return None, None
    try:
        x, y = ref_wcs.world_to_pixel_values(float(ra_deg), float(dec_deg))
        return float(x), float(y)
    except Exception:
        return None, None


def rows_to_xy(rows, x_key, y_key):
    xs = []
    ys = []
    for row in rows:
        x = parse_float(row.get(x_key))
        y = parse_float(row.get(y_key))
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack([np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)])


def save_match_overlay_png(ref_img, candidate_xy, hip_xy, var_xy, mpc_xy, out_png: Path):
    finite = np.isfinite(ref_img)
    fill = np.nanmedian(ref_img[finite]) if np.any(finite) else 0.0
    view = np.where(finite, ref_img, fill)
    h, w = view.shape
    dpi = 100

    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, frameon=False)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    norm = ImageNormalize(view, interval=PercentileInterval(99.5), stretch=SqrtStretch())
    ax.imshow(view, origin="lower", cmap="gray", norm=norm, interpolation="nearest")

    if len(candidate_xy) > 0:
        ax.scatter(
            candidate_xy[:, 0],
            candidate_xy[:, 1],
            marker="o",
            s=62,
            facecolors="none",
            edgecolors="#FFD400",
            linewidths=1.0,
            alpha=0.95,
            label="Candidates (inner-border CSV)",
        )
    if len(hip_xy) > 0:
        ax.scatter(
            hip_xy[:, 0],
            hip_xy[:, 1],
            marker="x",
            s=34,
            c="#00E5FF",
            linewidths=1.0,
            alpha=0.95,
            label="HIP matches (match_x/y)",
        )
    if len(var_xy) > 0:
        ax.scatter(
            var_xy[:, 0],
            var_xy[:, 1],
            marker="+",
            s=36,
            c="#57FF57",
            linewidths=1.0,
            alpha=0.95,
            label="Variable matches (match_x/y)",
        )
    if len(mpc_xy) > 0:
        ax.scatter(
            mpc_xy[:, 0],
            mpc_xy[:, 1],
            marker="^",
            s=28,
            c="#FF3EA5",
            alpha=0.9,
            linewidths=0.5,
            edgecolors="white",
            label="MPC matches (match_x/y)",
        )

    if len(candidate_xy) > 0 or len(hip_xy) > 0 or len(var_xy) > 0 or len(mpc_xy) > 0:
        ax.legend(loc="upper right", framealpha=0.65, fontsize=6)

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_axis_off()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def main():
    args = parse_args()
    input_csv = args.input_csv
    if not input_csv.exists():
        raise RuntimeError(f"Input CSV not found: {input_csv}")

    out_hip = args.find_hip_csv if args.find_hip_csv is not None else input_csv.with_name("find_hip.csv")
    out_var = args.find_variable_csv if args.find_variable_csv is not None else input_csv.with_name("find_variable.csv")
    out_mpc = args.find_mpc_csv if args.find_mpc_csv is not None else input_csv.with_name("find_mpc.csv")
    out_match_png = input_csv.with_name("variable_candidates_rank_aligned_to_a_match.png")
    ref_fits_path = resolve_ref_fits_path(args.ref_fits, input_csv)
    ref_wcs, ref_wcs_path = load_reference_wcs(args.ref_fits, input_csv)
    if ref_wcs is None:
        print("WARNING: Reference WCS unavailable, match_x/match_y will be empty in find_*.csv.")
    else:
        print(f"WCS source: {ref_wcs_path}")

    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    for c in ("hip_count", "variable_count", "mpc_count"):
        if c not in fieldnames:
            fieldnames.append(c)

    hip_rows = []
    variable_rows = []
    mpc_rows = []
    n_hip_queried = 0
    n_var_queried = 0
    n_mpc_queried = 0

    hip_url = f"http://{args.hip_host}:{args.hip_port}/search"
    var_url = f"http://{args.var_host}:{args.var_port}/search"
    mpc_url = f"http://{args.mpc_host}:{args.mpc_port}/search"

    for row in rows:
        row["hip_count"] = "-1"
        row["variable_count"] = "-1"
        row["mpc_count"] = "-1"

        rank = row.get("rank", "")
        x = row.get("x", "")
        y = row.get("y", "")
        ra = parse_float(row.get("ra_deg"))
        dec = parse_float(row.get("dec_deg"))
        scale = parse_float(row.get("arcsec_per_px_mean"))
        if scale is None:
            scale = args.arcsec_per_px_fallback
        mjd = parse_float(row.get("mjd"))

        if ra is None or dec is None or scale is None:
            continue
        radius_arcsec = float(args.radius_px) * float(scale)

        # 1) HIP
        hip_success = False
        hip_count = -1
        try:
            n_hip_queried += 1
            hip_resp = request_search(
                hip_url,
                {"ra": ra, "dec": dec, "radius": radius_arcsec, "top": 500},
                timeout_sec=float(args.timeout_sec),
            )
            hip_count = int(hip_resp.get("count", 0))
            hip_success = True
            row["hip_count"] = str(hip_count)
            for j, item in enumerate(hip_resp.get("results", []), start=1):
                item_ra, item_dec = extract_item_ra_dec(item)
                match_x, match_y = world_to_pixel_xy(ref_wcs, item_ra, item_dec)
                hip_rows.append(
                    {
                        "candidate_rank": rank,
                        "candidate_x": x,
                        "candidate_y": y,
                        "candidate_ra_deg": f"{ra:.8f}",
                        "candidate_dec_deg": f"{dec:.8f}",
                        "radius_arcsec": f"{radius_arcsec:.6f}",
                        "result_index": j,
                        "hip": item.get("hip"),
                        "match_ra_deg": f"{item_ra:.8f}" if item_ra is not None else "",
                        "match_dec_deg": f"{item_dec:.8f}" if item_dec is not None else "",
                        "match_x": f"{match_x:.4f}" if match_x is not None else "",
                        "match_y": f"{match_y:.4f}" if match_y is not None else "",
                        "mag": item.get("mag"),
                        "separation_arcsec": item.get("separation_arcsec"),
                        "raw_json": json.dumps(item, ensure_ascii=False),
                    }
                )
        except Exception:
            pass

        # 2) Variable (only when HIP queried successfully and no hit)
        var_success = False
        var_count = -1
        if hip_success and hip_count < 1:
            try:
                n_var_queried += 1
                var_resp = request_search(
                    var_url,
                    {"ra": ra, "dec": dec, "radius": radius_arcsec, "top": 500},
                    timeout_sec=float(args.timeout_sec),
                )
                var_count = int(var_resp.get("count", 0))
                var_success = True
                row["variable_count"] = str(var_count)
                for j, item in enumerate(var_resp.get("results", []), start=1):
                    item_ra, item_dec = extract_item_ra_dec(item)
                    match_x, match_y = world_to_pixel_xy(ref_wcs, item_ra, item_dec)
                    variable_rows.append(
                        {
                            "candidate_rank": rank,
                            "candidate_x": x,
                            "candidate_y": y,
                            "candidate_ra_deg": f"{ra:.8f}",
                            "candidate_dec_deg": f"{dec:.8f}",
                            "radius_arcsec": f"{radius_arcsec:.6f}",
                            "result_index": j,
                            "name": item.get("name"),
                            "match_ra_deg": f"{item_ra:.8f}" if item_ra is not None else "",
                            "match_dec_deg": f"{item_dec:.8f}" if item_dec is not None else "",
                            "match_x": f"{match_x:.4f}" if match_x is not None else "",
                            "match_y": f"{match_y:.4f}" if match_y is not None else "",
                            "mag_min": item.get("mag_min"),
                            "mag_max": item.get("mag_max"),
                            "period": item.get("period"),
                            "separation_arcsec": item.get("separation_arcsec"),
                            "raw_json": json.dumps(item, ensure_ascii=False),
                        }
                    )
            except Exception:
                pass

        # 3) MPC (only when HIP and Variable both queried successfully and both no hit)
        if hip_success and var_success and hip_count < 1 and var_count < 1 and mjd is not None:
            try:
                n_mpc_queried += 1
                mpc_resp = request_search(
                    mpc_url,
                    {"ra": ra, "dec": dec, "epoch": mjd, "radius": radius_arcsec},
                    timeout_sec=float(args.timeout_sec),
                )
                if bool(mpc_resp.get("success", False)):
                    mpc_count = int(mpc_resp.get("count", 0))
                    row["mpc_count"] = str(mpc_count)
                    for j, item in enumerate(mpc_resp.get("results", []), start=1):
                        item_ra, item_dec = extract_item_ra_dec(item)
                        match_x, match_y = world_to_pixel_xy(ref_wcs, item_ra, item_dec)
                        mpc_rows.append(
                            {
                                "candidate_rank": rank,
                                "candidate_x": x,
                                "candidate_y": y,
                                "candidate_ra_deg": f"{ra:.8f}",
                                "candidate_dec_deg": f"{dec:.8f}",
                                "candidate_mjd": f"{mjd:.8f}",
                                "radius_arcsec": f"{radius_arcsec:.6f}",
                                "result_index": j,
                                "name": item.get("name"),
                                "match_ra_deg": f"{item_ra:.8f}" if item_ra is not None else "",
                                "match_dec_deg": f"{item_dec:.8f}" if item_dec is not None else "",
                                "match_x": f"{match_x:.4f}" if match_x is not None else "",
                                "match_y": f"{match_y:.4f}" if match_y is not None else "",
                                "mag": item.get("mag"),
                                "separation": item.get("separation"),
                                "raw_json": json.dumps(item, ensure_ascii=False),
                            }
                        )
            except Exception:
                pass

    write_rows_csv(
        out_hip,
        hip_rows,
        [
            "candidate_rank",
            "candidate_x",
            "candidate_y",
            "candidate_ra_deg",
            "candidate_dec_deg",
            "radius_arcsec",
            "result_index",
            "hip",
            "match_ra_deg",
            "match_dec_deg",
            "match_x",
            "match_y",
            "mag",
            "separation_arcsec",
            "raw_json",
        ],
    )
    write_rows_csv(
        out_var,
        variable_rows,
        [
            "candidate_rank",
            "candidate_x",
            "candidate_y",
            "candidate_ra_deg",
            "candidate_dec_deg",
            "radius_arcsec",
            "result_index",
            "name",
            "match_ra_deg",
            "match_dec_deg",
            "match_x",
            "match_y",
            "mag_min",
            "mag_max",
            "period",
            "separation_arcsec",
            "raw_json",
        ],
    )
    write_rows_csv(
        out_mpc,
        mpc_rows,
        [
            "candidate_rank",
            "candidate_x",
            "candidate_y",
            "candidate_ra_deg",
            "candidate_dec_deg",
            "candidate_mjd",
            "radius_arcsec",
            "result_index",
            "name",
            "match_ra_deg",
            "match_dec_deg",
            "match_x",
            "match_y",
            "mag",
            "separation",
            "raw_json",
        ],
    )

    with input_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if ref_fits_path is None:
        print("WARNING: Reference FITS unavailable, skip variable_candidates_rank_aligned_to_a_match.png.")
    else:
        try:
            ref_data = fits.getdata(ref_fits_path).astype(float)
            candidate_xy = rows_to_xy(rows, "x", "y")
            hip_xy = rows_to_xy(hip_rows, "match_x", "match_y")
            var_xy = rows_to_xy(variable_rows, "match_x", "match_y")
            mpc_xy = rows_to_xy(mpc_rows, "match_x", "match_y")
            save_match_overlay_png(ref_data, candidate_xy, hip_xy, var_xy, mpc_xy, out_match_png)
            print(f"WROTE {out_match_png}")
        except Exception as exc:
            print(f"WARNING: Failed to write {out_match_png}: {exc}")

    print(f"WROTE {input_csv}")
    print(f"WROTE {out_hip}")
    print(f"WROTE {out_var}")
    print(f"WROTE {out_mpc}")
    print(f"queries: hip={n_hip_queried}, variable={n_var_queried}, mpc={n_mpc_queried}")
    print(f"matches: hip={len(hip_rows)}, variable={len(variable_rows)}, mpc={len(mpc_rows)}")


if __name__ == "__main__":
    main()

