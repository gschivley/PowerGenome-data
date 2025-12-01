#!/usr/bin/env python3
"""Merge AC and non-AC transmission capacity files into a single CSV.

Downloads transmission files from ReEDS GitHub repo and merges them.

Output columns: region_from,region_to,firm_ttc_mw,notes

Usage:
  python merge_transmission_capacity.py
"""
import csv
import sys
from io import StringIO
from pathlib import Path

import requests

# GitHub URLs for transmission files
AC_URL = "https://raw.githubusercontent.com/NREL/ReEDS-2.0/main/inputs/transmission/transmission_capacity_init_AC_ba_NARIS2024.csv"
NONAC_URL = "https://raw.githubusercontent.com/NREL/ReEDS-2.0/main/inputs/transmission/transmission_capacity_init_nonAC_ba.csv"


def download_file(url, cache_path):
    """Download file from URL and cache it locally."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Using cached file: {cache_path}")
        with open(cache_path, "r") as f:
            return f.read()

    print(f"Downloading from {url}...")
    response = requests.get(url)
    response.raise_for_status()

    # Cache the file
    with open(cache_path, "w") as f:
        f.write(response.text)
    print(f"  Cached to {cache_path}")

    return response.text


def parse_ac(csv_text):
    rows = []
    reader = csv.DictReader(StringIO(csv_text))
    for r in reader:
        try:
            mw_f0 = float(r.get("MW_f0", "") or 0.0)
        except ValueError:
            mw_f0 = 0.0
        try:
            mw_r0 = float(r.get("MW_r0", "") or 0.0)
        except ValueError:
            mw_r0 = 0.0
        avg = (mw_f0 + mw_r0) / 2.0
        rows.append(
            {
                "region_from": r.get("r", "").strip(),
                "region_to": r.get("rr", "").strip(),
                "firm_ttc_mw": f"{avg:.6f}",
                "notes": "",
            }
        )
    return rows


def parse_nonac(csv_text):
    rows = []
    reader = csv.DictReader(StringIO(csv_text))
    for r in reader:
        mw_raw = r.get("MW", "")
        try:
            mw = float(mw_raw) if mw_raw not in (None, "") else 0.0
        except ValueError:
            mw = 0.0
        notes = (r.get("Notes") or "").strip()
        if not notes:
            # fallback to Project(s) if Notes empty
            notes = (r.get("Project(s)") or "").strip()
        rows.append(
            {
                "region_from": r.get("r", "").strip(),
                "region_to": r.get("rr", "").strip(),
                "firm_ttc_mw": f"{mw:.6f}",
                "notes": notes,
            }
        )
    return rows


def write_output(out_path, rows):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        fieldnames = ["region_from", "region_to", "firm_ttc_mw", "notes"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    """Main execution function."""
    print("=" * 60)
    print("Merging ReEDS Transmission Capacity Data")
    print("=" * 60)

    # Download files
    ac_text = download_file(
        AC_URL, "cache_data/transmission_capacity_init_AC_ba_NARIS2024.csv"
    )
    nonac_text = download_file(
        NONAC_URL, "cache_data/transmission_capacity_init_nonAC_ba.csv"
    )

    # Parse data
    print("\nParsing AC transmission data...")
    ac_rows = parse_ac(ac_text)
    print(f"  Loaded {len(ac_rows):,} AC transmission rows")

    print("Parsing non-AC transmission data...")
    nonac_rows = parse_nonac(nonac_text)
    print(f"  Loaded {len(nonac_rows):,} non-AC transmission rows")

    # Combine
    combined = ac_rows + nonac_rows

    # Write output
    out_file = "data/transmission_capacity_merged.csv"
    write_output(out_file, combined)

    print(f"\n✓ Wrote {len(combined):,} total rows to {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
