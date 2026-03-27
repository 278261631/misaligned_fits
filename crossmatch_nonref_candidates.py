from pathlib import Path
import argparse
import csv
import json

import requests


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


def main():
    args = parse_args()
    input_csv = args.input_csv
    if not input_csv.exists():
        raise RuntimeError(f"Input CSV not found: {input_csv}")

    out_hip = args.find_hip_csv if args.find_hip_csv is not None else input_csv.with_name("find_hip.csv")
    out_var = args.find_variable_csv if args.find_variable_csv is not None else input_csv.with_name("find_variable.csv")
    out_mpc = args.find_mpc_csv if args.find_mpc_csv is not None else input_csv.with_name("find_mpc.csv")

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

    print(f"WROTE {input_csv}")
    print(f"WROTE {out_hip}")
    print(f"WROTE {out_var}")
    print(f"WROTE {out_mpc}")
    print(f"queries: hip={n_hip_queried}, variable={n_var_queried}, mpc={n_mpc_queried}")
    print(f"matches: hip={len(hip_rows)}, variable={len(variable_rows)}, mpc={len(mpc_rows)}")


if __name__ == "__main__":
    main()

