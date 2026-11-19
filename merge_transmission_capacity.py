#!/usr/bin/env python3
"""Merge AC and non-AC transmission capacity files into a single CSV. Add the average of
MW_f0 and MW_r0 for AC lines to MW for non-AC lines when they share the same region pair.

Downloads and merges historical ReEDS-2.0 transmission inputs. For current
z134 and z90 data, use build_reeds_transmission.py instead.

Output: transmission_capacity_reeds_legacy.csv

Usage:
  python merge_transmission_capacity.py
"""
import argparse
from pathlib import Path

import pandas as pd

from locked_sources import SourceLock, add_source_options

AC_SOURCE = "transmission/transmission_capacity_init_AC_ba_NARIS2024.csv"
NONAC_SOURCE = "transmission/transmission_capacity_init_nonAC_ba.csv"


def parse_ac(csv_text):
    """Parse AC transmission CSV and return DataFrame."""
    df = pd.read_csv(pd.io.common.StringIO(csv_text))

    # Convert MW columns to float, handling missing/invalid values
    df["MW_f0"] = pd.to_numeric(df.get("MW_f0", 0), errors="coerce").fillna(0)
    df["MW_r0"] = pd.to_numeric(df.get("MW_r0", 0), errors="coerce").fillna(0)

    # Calculate average
    df["firm_ttc_mw"] = (df["MW_f0"] + df["MW_r0"]) / 2.0

    # Rename and select columns
    df = df.rename(columns={"r": "region_from", "rr": "region_to"})
    df["notes"] = ""

    return df[["region_from", "region_to", "firm_ttc_mw", "notes"]]


def parse_nonac(csv_text):
    """Parse non-AC transmission CSV and return DataFrame."""
    df = pd.read_csv(pd.io.common.StringIO(csv_text))

    # Convert MW to float
    df["firm_ttc_mw"] = pd.to_numeric(df.get("MW", 0), errors="coerce").fillna(0)

    # Get notes, fallback to Project(s) if empty
    df["notes"] = df.get("Notes", "").fillna("")
    mask = df["notes"] == ""
    df.loc[mask, "notes"] = df.get("Project(s)", "").fillna("")

    # Rename and select columns
    df = df.rename(columns={"r": "region_from", "rr": "region_to"})

    return df[["region_from", "region_to", "firm_ttc_mw", "notes"]]


def main():
    """Main execution function."""
    print("=" * 60)
    print("Merging ReEDS Transmission Capacity Data")
    print("=" * 60)

    parser = argparse.ArgumentParser(description=__doc__)
    add_source_options(parser)
    args = parser.parse_args()
    sources = SourceLock(args.cache_dir, args.sources_lock)
    ac_text = sources.fetch(AC_SOURCE).read_text(encoding="utf-8")
    nonac_text = sources.fetch(NONAC_SOURCE).read_text(encoding="utf-8")

    # Parse data
    print("\nParsing AC transmission data...")
    ac_df = parse_ac(ac_text)
    print(f"  Loaded {len(ac_df):,} AC transmission rows")

    print("Parsing non-AC transmission data...")
    nonac_df = parse_nonac(nonac_text)
    print(f"  Loaded {len(nonac_df):,} non-AC transmission rows")

    # Combine dataframes
    print("\nAggregating capacities by region pair...")
    combined_df = pd.concat([ac_df, nonac_df], ignore_index=True)

    # Aggregate by region pair
    agg_df = combined_df.groupby(["region_from", "region_to"], as_index=False).agg(
        {
            "firm_ttc_mw": "sum",
            "notes": lambda x: "; ".join(filter(None, x.unique())),
        }
    )

    print(f"  Aggregated to {len(agg_df):,} unique region pairs")

    # Write output
    out_file = "transmission_capacity_reeds_legacy.csv"
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sources.output(out_path, (AC_SOURCE, NONAC_SOURCE)) as output:
        agg_df.to_csv(output, index=False)

    print(f"\n✓ Wrote {len(agg_df):,} total rows to {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
