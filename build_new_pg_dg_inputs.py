"""
Build distributed generation inputs for PowerGenome.

Creates two tables:
1. distributed_capacity: region, capacity_mw, year
2. distributed_profiles: region, weather_year, value

Supports inputs from either a local directory or directly from the
NREL ReEDS GitHub repository (raw URLs under inputs/).

Uses the same county-to-BA aggregation as build_dg_file.py.

Note that capacity is in MWdc. Use an ILR of 1.1 (from ReEDS discussion)
https://github.com/NREL/ReEDS-2.0/discussions/227#discussioncomment-13069190
"""

import argparse
import os
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import requests

INPUTS_DIR = Path(__file__).parent.parent

# Base URL for ReEDS inputs on GitHub (raw content)
GITHUB_INPUTS_BASE = "https://raw.githubusercontent.com/NREL/ReEDS-2.0/main/inputs"


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
    """
    Build distributed profiles table with columns: time_index, region, weather_year, value.

    Profiles represent capacity factors (0-1) without any conversion.
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

    with h5py.File(str(h5_path), "r") as f:
        generation_profiles = pd.DataFrame(
            f["data"][:],
            columns=[x.decode("utf-8") for x in f["columns"][:]],
            index=pd.to_datetime([x.decode("utf-8") for x in f["index_0"][:]]),
        )
        # Filter to years <= 2013
        generation_profiles = generation_profiles.loc[
            generation_profiles.index.year <= 2013, :
        ]
    # Clean up temp file if used
    if tmp_file is not None and os.path.exists(tmp_file):
        # Keep the file until we finish using 'generation_profiles'
        pass

    # Adjust for UTC offset (assume 0 if index is tz-naive)
    if (
        generation_profiles.index.tz is not None
        and generation_profiles.index[0].utcoffset() is not None
    ):
        utc_offset_hours = int(
            generation_profiles.index[0].utcoffset().total_seconds() / 3600
        )
    else:
        utc_offset_hours = 0

    # Build tidy dataframe
    df_list = []
    for region in tqdm(generation_profiles.columns, desc="Processing regions"):
        # Get the generation profile for the region (roll to adjust UTC offset)
        gen_profile = np.roll(generation_profiles[region].values, -utc_offset_hours)

        # Create dataframe with weather_year
        _df = pd.DataFrame(
            {
                "time_index": range(len(gen_profile)),
                "region": region,
                "weather_year": generation_profiles.index.year,
                "value": gen_profile,
            }
        )

        df_list.append(_df)

    distributed_profiles = pd.concat(df_list, ignore_index=True)
    distributed_profiles["region"] = distributed_profiles["region"].astype("category")
    distributed_profiles["weather_year"] = distributed_profiles["weather_year"].astype(
        "int32"
    )
    distributed_profiles["value"] = distributed_profiles["value"].astype("float32")

    # Remove temp file if created
    if tmp_file is not None and os.path.exists(tmp_file):
        try:
            os.remove(tmp_file)
        except OSError:
            pass

    return distributed_profiles[["time_index", "region", "weather_year", "value"]]


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
    distributed_capacity.to_parquet(output_capacity_path, index=False)
    print(f"Saved distributed capacity to {output_capacity_path}")
    print(f"  Shape: {distributed_capacity.shape}")
    print(f"  Columns: {list(distributed_capacity.columns)}")
    print(f"  Sample:\n{distributed_capacity.head()}\n")

    # Build distributed profiles table
    if args.skip_profiles:
        print("Skipping distributed profiles as requested (--skip-profiles).")
    else:
        print("Building distributed profiles table...")
        # Use the GitHub /raw URL to ensure LFS-backed binaries are served
        distpv_h5_url = "https://github.com/NREL/ReEDS-2.0/raw/main/inputs/variability/multi_year/distpv-reference_ba.h5"
        distributed_profiles = build_distributed_profiles(
            generation_profiles_path=distpv_h5_url,
        )
        # Save distributed profiles
        output_profiles_path = "distributed_profiles.parquet"
        distributed_profiles.to_parquet(output_profiles_path, index=False)
        print(f"Saved distributed profiles to {output_profiles_path}")
        print(f"  Shape: {distributed_profiles.shape}")
        print(f"  Columns: {list(distributed_profiles.columns)}")
        print(f"  Sample:\n{distributed_profiles.head()}\n")


if __name__ == "__main__":
    main()
