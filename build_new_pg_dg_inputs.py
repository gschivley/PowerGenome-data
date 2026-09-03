"""
Build distributed generation inputs for PowerGenome.

Creates two tables:
1. distributed_capacity: region, capacity_mw, year
2. distributed_profiles: region, weather_year, value

Supports inputs from either a local directory or directly from the
ReEDS GitHub repository (raw URLs under inputs/). ReEDS now provides
distributed PV data at the *county* level rather than by BA region:

- county-level DG capacity: inputs/dgen_model_inputs/<scenario>/distpvcap_<scenario>.csv
- hourly county-level capacity factors: cf_distpv_county.h5 (hosted on
  Zenodo; the download URL is resolved from inputs/remote_files.csv)
- county population: inputs/disaggregation/county_population.csv
- county-to-BA mapping: inputs/zones/z134/county2zone.csv

Outputs are kept at the BA (ReEDS region) level: county capacities are
summed to each BA, and the county capacity-factor profiles are averaged
to each BA weighted by county capacity (the same weighting ReEDS uses in
calculate_regional_distpv_cf) or optionally by county population.

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
GITHUB_INPUTS_BASE = "https://raw.githubusercontent.com/ReEDS-Model/ReEDS/main/inputs"

# Paths (relative to GITHUB_INPUTS_BASE) to the county-level inputs
REMOTE_FILES_PATH = "remote_files.csv"
DISTPV_SCEN = "stscen2023_mid_case"
DISTPV_CAP_PATH = f"dgen_model_inputs/{DISTPV_SCEN}/distpvcap_{DISTPV_SCEN}.csv"
COUNTY2ZONE_PATH = "zones/z134/county2zone.csv"
COUNTY_POPULATION_PATH = "disaggregation/county_population.csv"
DISTPV_CF_FILENAME = "cf_distpv_county.h5"

# A tiny capacity floor used when weighting profiles, matching ReEDS, so
# counties/BAs with (near) zero capacity don't produce divide-by-zero NaNs.
CAP_MIN = 0.0001


def is_url(path: Union[str, Path]) -> bool:
    return isinstance(path, str) and path.startswith(("http://", "https://"))


def _stream_request(url: str, dest: str) -> None:
    """Stream a URL download to dest, raising a helpful error on failure."""
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    except requests.RequestException as e:
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(f"Failed to download {url}: {e}") from e


def download_file(
    url: str, cache_dir: Union[str, Path, None] = None
) -> str:
    """Download a URL to a local file and return its path.

    If cache_dir is given, the file is cached there (reusing it on later
    calls to avoid re-downloading large files); otherwise a temporary file
    is used that the caller is expected to clean up.
    """
    fname = Path(url.split("/")[-1] or "download")

    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        dest = str(Path(cache_dir) / fname)
        if os.path.exists(dest):
            return dest
    else:
        fd, dest = tempfile.mkstemp(suffix=fname.suffix)
        os.close(fd)

    _stream_request(url, dest)

    # If this is supposed to be an HDF5 file, verify the signature
    if fname.suffix == ".h5":
        with open(dest, "rb") as fh:
            sig = fh.read(8)
        # HDF5 signature should be b"\x89HDF\r\n\x1a\n"
        if sig != b"\x89HDF\r\n\x1a\n":
            if os.path.exists(dest):
                os.remove(dest)
            raise ValueError(
                f"Downloaded file from {url} does not look like a valid HDF5 "
                "(unexpected signature). This may be an HTML error page or a "
                "Git LFS pointer. Check the URL or provide a local path."
            )
    return dest


def resolve_zenodo_url(remote_files_path: Union[str, Path], filename: str) -> str:
    """Resolve the Zenodo download URL for a file from ReEDS remote_files.csv.

    remote_files.csv (inputs/remote_files.csv in the ReEDS repo) lists the
    Zenodo record hosting each large model input. Each row's url_base
    contains {record_id} and {filename} placeholders to format.
    """
    remote_files = pd.read_csv(str(remote_files_path))
    matches = remote_files[remote_files["filename"] == filename]
    if matches.empty:
        raise ValueError(
            f"'{filename}' is not listed in {remote_files_path}. "
            "The file may have been renamed or moved in a newer ReEDS release."
        )
    row = matches.iloc[0]
    return str(row["url_base"]).format(
        record_id=row["record_id"], filename=row["filename"]
    )


def read_county2zone(mapping_path: Union[str, Path]) -> pd.Series:
    """Return county ('p'+FIPS) -> BA mapping as a Series.

    The new county2zone.csv has columns FIPS, r where r is the ReEDS BA
    (e.g. 'p1'..'p134'). FIPS is stored without the leading 'p' (and
    sometimes without leading zeros), so it is reformatted to match the
    county keys used in the county capacity and profile files.
    """
    mapping = pd.read_csv(str(mapping_path), dtype={"FIPS": str})
    mapping = mapping.drop_duplicates("FIPS").set_index("FIPS")
    # Build the county key ('p'+FIPS, zero-padded) and pair it with the BA.
    # Pass values as an array so the mapping is positional, not label-aligned.
    county2zone = pd.Series(
        mapping["r"].astype(str).to_numpy(),
        index="p" + mapping.index.str.zfill(5),
        name="r",
    )
    county2zone.index.name = "r"
    return county2zone


def build_distributed_capacity(
    county_capacity_path: Union[str, Path],
    mapping_path: Union[str, Path],
    ilr: float = 1.1,
) -> pd.DataFrame:
    """
    Build distributed capacity table with columns: region, capacity_mw, year.

    Aggregates county-level capacity to BA regions (sum) and converts DC
    capacity to AC capacity by dividing by ILR.
    """
    county_capacity = pd.read_csv(str(county_capacity_path))
    county2zone = read_county2zone(mapping_path)

    # Attach each county's BA using the county key from the capacity file
    county_capacity = county_capacity.merge(
        county2zone.rename("ba").reset_index(), on="r", how="left"
    )

    unmapped = county_capacity["ba"].isna()
    if unmapped.any():
        missing = sorted(county_capacity.loc[unmapped, "r"].head().tolist())
        raise ValueError(
            f"{unmapped.sum()} counties in the capacity file are not found in "
            f"the county2zone mapping (e.g. {missing}). Check that the mapping "
            "and capacity files cover the same county set."
        )

    # Sum county capacities by BA and reshape to year/capacity rows
    ba_capacity = (
        county_capacity.drop(columns=["r"])
        .groupby(["ba"], as_index=False)
        .sum()
        .melt(id_vars=["ba"], var_name="year", value_name="capacity_mw")
    )

    # Convert DC capacity to AC capacity by dividing by ILR
    ba_capacity["capacity_mw"] = ba_capacity["capacity_mw"] / ilr

    # Rename ba to region and ensure proper types
    ba_capacity = ba_capacity.rename(columns={"ba": "region"})
    ba_capacity["year"] = ba_capacity["year"].astype("int32")
    ba_capacity["capacity_mw"] = ba_capacity["capacity_mw"].astype("float32").round(1)
    ba_capacity["region"] = ba_capacity["region"].astype("category")

    return ba_capacity[["region", "capacity_mw", "year"]]


def _county_weights(
    counties,
    county_capacity_path: Union[str, Path],
    county_population_path: Union[str, Path],
    county2zone: pd.Series,
    weight_year: int,
    weight_by: str,
    cap_min: float = CAP_MIN,
) -> pd.Series:
    """
    Return per-county weights (indexed by county) for averaging profiles to BA.

    weight_by='capacity' uses the county DG capacity in weight_year (falling
    back to weight_year + 1 if that year is absent from the file, matching
    ReEDS' GSw_HourlyClusterYear handling). weight_by='population' uses the
    county population. Weights are floored at cap_min to avoid
    divide-by-zero / NaN in BAs with no capacity.
    """
    if weight_by == "capacity":
        capacities = pd.read_csv(str(county_capacity_path)).set_index("r")
        year = weight_year
        if str(year) not in capacities.columns:
            year = weight_year + 1
        if str(year) not in capacities.columns:
            available = sorted(
                int(c) for c in capacities.columns if str(c).isdigit()
            )
            raise ValueError(
                f"No capacity data for weight years {weight_year}/{weight_year + 1} "
                f"in {county_capacity_path}. Available years: {available}."
            )
        weights = capacities.loc[counties, str(year)].clip(lower=cap_min)
    elif weight_by == "population":
        population = pd.read_csv(
            str(county_population_path), dtype={"FIPS": str}
        )
        population["FIPS"] = population["FIPS"].str.removeprefix("p")
        population = population.set_index("FIPS")["value"]
        weights = population.loc[counties].clip(lower=cap_min)
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unknown weight_by: {weight_by}")

    return weights.astype("float64")


def build_distributed_profiles(
    generation_profiles_path: Union[str, Path],
    county_capacity_path: Union[str, Path],
    county_population_path: Union[str, Path],
    mapping_path: Union[str, Path],
    weight_year: int = 2035,
    weight_by: str = "capacity",
    max_weather_year: int = 2013,
    cache_dir: Union[str, Path, None] = None,
) -> pd.DataFrame:
    """
    Build distributed profiles table with columns: time_index, region, weather_year, value.

    Reads county-level capacity factors, aggregates them to BA regions using
    a capacity-weighted (or population-weighted) average, then reshapes to a
    tidy frame. Profiles represent capacity factors (0-1) without conversion.
    """
    # Defer heavy imports so script can run without them if skipping profiles
    import h5py
    from tqdm import tqdm

    county2zone = read_county2zone(mapping_path)

    # Download the .h5 file if a URL is given
    tmp_file = None
    h5_path = generation_profiles_path
    if is_url(generation_profiles_path):
        tmp_file = download_file(str(generation_profiles_path), cache_dir=cache_dir)
        h5_path = tmp_file

    with h5py.File(str(h5_path), "r") as f:
        county_cf = pd.DataFrame(
            f["data"][:],
            columns=[x.decode("utf-8") for x in f["columns"][:]],
            index=pd.to_datetime([x.decode("utf-8") for x in f["index_0"][:]]),
        )
        county_cf = county_cf.loc[county_cf.index.year <= max_weather_year, :]

    # Downselect profiles to counties present in the mapping (all of them in
    # practice) and align the weights to the same county set.
    counties = county_cf.columns.intersection(county2zone.index)
    if counties.empty:
        raise ValueError(
            "No county-level capacity factor columns match the county2zone "
            "mapping. Check that both files use the same county keys."
        )
    county_cf = county_cf[counties]

    county_weights = _county_weights(
        counties=counties,
        county_capacity_path=county_capacity_path,
        county_population_path=county_population_path,
        county2zone=county2zone,
        weight_year=weight_year,
        weight_by=weight_by,
    )

    # BA-level weights: sum of the county weights within each BA
    regional_weights = county_weights.groupby(county2zone.loc[county_weights.index]).sum()

    # Capacity-weighted average capacity factor per BA, mirroring ReEDS'
    # calculate_regional_distpv_cf: regional generation / regional capacity.
    # Drop the temporary (downloaded) file only after the CF frame is built.
    regional_cf = (
        county_cf.mul(county_weights, axis=1)
        .rename(columns=county2zone)
        .T.groupby(level=0)
        .sum()
        .T.div(regional_weights, axis=1)
    )

    # Clean up temporary file once we no longer need the downloaded data
    if tmp_file is not None and cache_dir is None and os.path.exists(tmp_file):
        try:
            os.remove(tmp_file)
        except OSError:
            pass

    # Adjust for UTC offset (assume 0 if index is tz-naive)
    if (
        county_cf.index.tz is not None
        and county_cf.index[0].utcoffset() is not None
    ):
        utc_offset_hours = int(county_cf.index[0].utcoffset().total_seconds() / 3600)
    else:
        utc_offset_hours = 0

    # Build tidy dataframe
    df_list = []
    for region in tqdm(regional_cf.columns, desc="Processing regions"):
        # Roll the profile to adjust for the UTC offset
        gen_profile = np.roll(regional_cf[region].values, -utc_offset_hours)

        _df = pd.DataFrame(
            {
                "time_index": range(len(gen_profile)),
                "region": region,
                "weather_year": regional_cf.index.year,
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
    parser.add_argument(
        "--weight-year",
        type=int,
        default=2035,
        help="Year of county capacity used to weight profiles to BA averages "
        "(default: 2035; falls back to year+1 if absent, like ReEDS)",
    )
    parser.add_argument(
        "--weight-by",
        choices=["capacity", "population"],
        default="capacity",
        help="Weight county capacity factors by county capacity (default) or "
        "county population when aggregating to BA averages",
    )
    parser.add_argument(
        "--max-weather-year",
        type=int,
        default=2013,
        help="Include weather years up to and including this year (default: 2013)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional directory to cache remote downloads (e.g. the large "
        "distpv HDF5) between runs",
    )
    args = parser.parse_args()

    ilr = args.ilr  # Inverter Loading Ratio

    # Build distributed capacity table
    print("Building distributed capacity table...")
    county_capacity_url = f"{GITHUB_INPUTS_BASE}/{DISTPV_CAP_PATH}"
    mapping_url = f"{GITHUB_INPUTS_BASE}/{COUNTY2ZONE_PATH}"

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
        # Resolve the Zenodo download URL for the county-level distpv HDF5
        # from ReEDS' remote_files.csv
        remote_files_url = f"{GITHUB_INPUTS_BASE}/{REMOTE_FILES_PATH}"
        distpv_h5_url = resolve_zenodo_url(remote_files_url, DISTPV_CF_FILENAME)
        print(f"  Resolved {DISTPV_CF_FILENAME} URL: {distpv_h5_url}")
        county_population_url = f"{GITHUB_INPUTS_BASE}/{COUNTY_POPULATION_PATH}"

        distributed_profiles = build_distributed_profiles(
            generation_profiles_path=distpv_h5_url,
            county_capacity_path=county_capacity_url,
            county_population_path=county_population_url,
            mapping_path=mapping_url,
            weight_year=args.weight_year,
            weight_by=args.weight_by,
            max_weather_year=args.max_weather_year,
            cache_dir=args.cache_dir,
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
