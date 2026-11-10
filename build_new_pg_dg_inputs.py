"""
Build distributed generation inputs for PowerGenome.

Creates two tables:
1. distributed_capacity: region, capacity_mw, year
2. distributed_profiles: time_index, region, weather_year, value, datetime_utc

UTC timestamps follow the source's offset labels, not a verified physical clock.
Legacy ReEDS BA distributed-PV profiles have an unresolved regional timing anomaly.

Supports inputs from either a local directory or directly from the
NREL ReEDS GitHub repository (raw URLs under inputs/).

Uses the same county-to-BA aggregation as build_dg_file.py.

Note that capacity is in MWdc. Use an ILR of 1.1 (from ReEDS discussion)
https://github.com/NREL/ReEDS-2.0/discussions/227#discussioncomment-13069190
"""

import argparse
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import requests

INPUTS_DIR = Path(__file__).parent.parent

# Base URL for ReEDS inputs on GitHub (raw content)
GITHUB_INPUTS_BASE = "https://raw.githubusercontent.com/NREL/ReEDS-2.0/main/inputs"
DISTPV_PROFILES_URL = (
    "https://media.githubusercontent.com/media/NatLabRockies/ReEDS-2.0/"
    "v2025.1.0/inputs/variability/multi_year/distpv-reference_ba.h5"
)


def is_url(path: Union[str, Path]) -> bool:
    return isinstance(path, str) and path.startswith(("http://", "https://"))


def _download_to_temp(url: str) -> str:
    """Download a URL to a temporary file and return its path."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        fd, tmp_path = tempfile.mkstemp(suffix=Path(url).suffix)
        with os.fdopen(fd, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    # If this is supposed to be an HDF5 file, verify the signature
    if tmp_path.endswith(".h5"):
        try:
            with open(tmp_path, "rb") as fh:
                sig = fh.read(8)
            # HDF5 signature should be b"\x89HDF\r\n\x1a\n"
            if sig != b"\x89HDF\r\n\x1a\n":
                raise ValueError(
                    "Downloaded file does not look like a valid HDF5 (unexpected signature)."
                )
        except Exception as e:
            # Leave file for debugging, but raise with helpful message
            raise RuntimeError(
                f"Failed to download a valid HDF5 from URL: {url}.\n"
                "This may be a Git LFS pointer. Try using the GitHub /raw URL or provide a local path."
            ) from e
    return tmp_path


def _to_github_raw_redirect(url: str) -> str:
    """Convert raw.githubusercontent.com URL to github.com ... /raw ... which resolves LFS binaries."""
    # Example:
    # https://raw.githubusercontent.com/OWNER/REPO/BRANCH/path ->
    # https://github.com/OWNER/REPO/raw/BRANCH/path
    if "raw.githubusercontent.com" in url:
        parts = url.split("raw.githubusercontent.com/")[-1].split("/")
        if len(parts) >= 3:
            owner, repo, branch, *rest = parts
            return f"https://github.com/{owner}/{repo}/raw/{branch}/{'/'.join(rest)}"
    return url


def build_distributed_capacity(
    county_capacity_path: Union[str, Path],
    mapping_path: Union[str, Path],
    ilr: float = 1.1,
) -> pd.DataFrame:
    """
    Build distributed capacity table with columns: region, capacity_mw, year.

    Aggregates county-level capacity to BA regions.
    Converts DC capacity to AC capacity by dividing by ILR.
    """
    # Load the county DG capacity data (supports local path or URL)
    county_capacity = pd.read_csv(str(county_capacity_path))

    # Load county to BA mapping, columns are FIPS,ba,county_name,state -- append "p" to FIPS
    mapping = pd.read_csv(str(mapping_path))
    mapping["r"] = "p" + mapping["FIPS"].astype(str).str.zfill(5)

    # Assign the BA to the FIPS code
    county_capacity = county_capacity.merge(mapping[["r", "ba"]], on="r", how="left")

    # Calculate BA capacity -- columns are year, r, ba
    ba_capacity = (
        county_capacity.drop(columns=["r"]).groupby(["ba"], as_index=False).sum()
    ).melt(id_vars=["ba"], var_name="year", value_name="capacity_mw")

    # Convert DC capacity to AC capacity by dividing by ILR
    ba_capacity["capacity_mw"] = ba_capacity["capacity_mw"] / ilr

    # Rename ba to region and ensure proper types
    ba_capacity = ba_capacity.rename(columns={"ba": "region"})
    ba_capacity["year"] = ba_capacity["year"].astype("int32")
    ba_capacity["capacity_mw"] = ba_capacity["capacity_mw"].astype("float32").round(1)
    ba_capacity["region"] = ba_capacity["region"].astype("category")

    return ba_capacity[["region", "capacity_mw", "year"]]


def build_distributed_profiles(
    generation_profiles_path: Union[str, Path],
) -> pd.DataFrame:
    """Build capacity-factor profiles with timezone-aware ``datetime_utc`` labels.

    Convert each 8760-hour weather year independently using a cyclic shift based
    on the source's fixed UTC offset. Preserve the source calendar (including
    Feb 29 and excluding Dec 31 in leap years) and the legacy 0-based time_index
    spanning all weather years. Naive or non-fixed-offset clocks are rejected.

    The legacy BA source has a regional timing anomaly: these UTC labels encode
    the source-offset assumption, not a verified physical clock correction.
    """
    # Defer heavy imports so script can run without them if skipping profiles
    import h5py
    from tqdm import tqdm

    # Load the generation profiles from .h5 file (download if URL)
    tmp_file = None
    h5_path = generation_profiles_path
    if is_url(generation_profiles_path):
        # Handle Git LFS assets by using the /raw redirect URL
        url = _to_github_raw_redirect(str(generation_profiles_path))
        tmp_file = _download_to_temp(url)
        h5_path = tmp_file

    try:
        with h5py.File(str(h5_path), "r") as f:
            timestamps = [
                datetime.fromisoformat(x.decode("utf-8")) for x in f["index_0"][:]
            ]
            offsets = {timestamp.utcoffset() for timestamp in timestamps}
            if None in offsets:
                raise ValueError(
                    "Distributed profiles require a timezone-aware source index."
                )
            if len(offsets) > 1 or any(
                offset.total_seconds() % 3600 for offset in offsets
            ):
                raise ValueError(
                    "Distributed profiles require a fixed, whole-hour UTC offset."
                )
            generation_profiles = pd.DataFrame(
                f["data"][:],
                columns=[x.decode("utf-8") for x in f["columns"][:]],
                index=pd.DatetimeIndex(timestamps),
            )
    finally:
        if tmp_file is not None:
            os.remove(tmp_file)

    generation_profiles = generation_profiles.loc[
        generation_profiles.index.year <= 2013, :
    ]
    index = generation_profiles.index
    if index.empty:
        raise ValueError("No distributed profiles found for weather years <= 2013.")
    utc_offset_hours = int(next(iter(offsets)).total_seconds() / 3600)

    shifted_years = []
    for year, profiles in generation_profiles.groupby(index.year, sort=True):
        expected = pd.date_range(f"{year}-01-01", periods=8760, freq="h")
        if not profiles.index.tz_localize(None).equals(expected):
            raise ValueError(
                f"Weather year {year} must contain 8760 consecutive hours from Jan 1."
            )
        # Rebase onto the same UTC weather-year grid; wrap only within this year.
        shifted_years.append(
            pd.DataFrame(
                np.roll(profiles.to_numpy(), -utc_offset_hours, axis=0),
                index=expected.tz_localize("UTC"),
                columns=profiles.columns,
            )
        )
    generation_profiles = pd.concat(shifted_years)

    df_list = []
    for region in tqdm(generation_profiles.columns, desc="Processing regions"):
        df_list.append(
            pd.DataFrame(
                {
                    "time_index": range(len(generation_profiles)),
                    "region": region,
                    "weather_year": generation_profiles.index.year,
                    "value": generation_profiles[region].to_numpy(),
                    "datetime_utc": generation_profiles.index,
                }
            )
        )

    distributed_profiles = pd.concat(df_list, ignore_index=True)
    distributed_profiles["region"] = distributed_profiles["region"].astype("category")
    distributed_profiles["weather_year"] = distributed_profiles["weather_year"].astype(
        "int32"
    )
    distributed_profiles["value"] = distributed_profiles["value"].astype("float32")

    return distributed_profiles[
        ["time_index", "region", "weather_year", "value", "datetime_utc"]
    ]


def main():
    """
    Main function to build and save distributed capacity and profiles.
    """
    parser = argparse.ArgumentParser(description="Build distributed generation inputs.")
    parser.add_argument(
        "--skip-profiles",
        action="store_true",
        help="Skip building distributed profiles (avoids large HDF5 download)",
    )
    parser.add_argument(
        "--ilr",
        type=float,
        default=1.1,
        help="Inverter loading ratio to convert DC to AC (default: 1.1)",
    )
    parser.add_argument(
        "--profiles-path",
        default=DISTPV_PROFILES_URL,
        help="Local HDF5 path or URL (default: pinned ReEDS v2025.1.0 BA profiles)",
    )
    args = parser.parse_args()

    ilr = args.ilr  # Inverter Loading Ratio

    # Build distributed capacity table
    print("Building distributed capacity table...")
    county_capacity_url = f"{GITHUB_INPUTS_BASE}/dgen_model_inputs/stscen2023_mid_case/distpvcap_stscen2023_mid_case.csv"
    mapping_url = f"{GITHUB_INPUTS_BASE}/county2zone.csv"

    distributed_capacity = build_distributed_capacity(
        county_capacity_path=county_capacity_url,
        mapping_path=mapping_url,
        ilr=ilr,
    )

    # Save distributed capacity
    output_capacity_path = "distributed_capacity.parquet"
    distributed_capacity.to_parquet(
        output_capacity_path, index=False, compression="zstd", compression_level=3
    )
    print(f"Saved distributed capacity to {output_capacity_path}")
    print(f"  Shape: {distributed_capacity.shape}")
    print(f"  Columns: {list(distributed_capacity.columns)}")
    print(f"  Sample:\n{distributed_capacity.head()}\n")

    # Build distributed profiles table
    if args.skip_profiles:
        print("Skipping distributed profiles as requested (--skip-profiles).")
    else:
        print("Building distributed profiles table...")
        distributed_profiles = build_distributed_profiles(
            generation_profiles_path=args.profiles_path,
        )
        # Save distributed profiles
        output_profiles_path = "distributed_profiles.parquet"
        distributed_profiles.to_parquet(
            output_profiles_path, index=False, compression="zstd", compression_level=3
        )
        print(f"Saved distributed profiles to {output_profiles_path}")
        print(f"  Shape: {distributed_profiles.shape}")
        print(f"  Columns: {list(distributed_profiles.columns)}")
        print(f"  Sample:\n{distributed_profiles.head()}\n")


if __name__ == "__main__":
    main()
