"""
Build distributed generation inputs for PowerGenome.

Creates two tables:
1. distributed_capacity: region, capacity_mw, year
2. distributed_profiles: time_index, region, weather_year, value, datetime_utc

The command verifies its ReEDS inputs against sources.lock.json.
The transformation functions accept local paths. ReEDS provides
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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from locked_sources import SourceLock, add_source_options

REMOTE_FILES_PATH = "remote_files.csv"
DISTPV_SCEN = "stscen2023_mid_case"
DISTPV_CAP_PATH = f"dgen_model_inputs/{DISTPV_SCEN}/distpvcap_{DISTPV_SCEN}.csv"
COUNTY2ZONE_PATH = "zones/z134/county2zone.csv"
COUNTY_POPULATION_PATH = "disaggregation/county_population.csv"
DISTPV_CF_FILENAME = "cf_distpv_county.h5"

# A tiny capacity floor used when weighting profiles, matching ReEDS, so
# counties/BAs with (near) zero capacity don't produce divide-by-zero NaNs.
CAP_MIN = 0.0001
HOURS_PER_WEATHER_YEAR = 8760
SOURCE_UTC_OFFSET_HOURS = -6


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
    county_population_path: Union[str, Path, None],
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
        population["FIPS"] = "p" + population["FIPS"].str.zfill(5)
        population = population.set_index("FIPS")["value"]
        weights = population.loc[counties].clip(lower=cap_min)
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unknown weight_by: {weight_by}")

    return weights.astype("float64")


def _county_source_index(labels: np.ndarray) -> pd.DatetimeIndex:
    try:
        timestamps = [datetime.fromisoformat(label.decode("utf-8")) for label in labels]
    except ValueError as exc:
        raise ValueError("Invalid timestamp in the county profile HDF5 index_0.") from exc
    source_offset = timedelta(hours=SOURCE_UTC_OFFSET_HOURS)
    if any(stamp.utcoffset() != source_offset for stamp in timestamps):
        raise ValueError("County profile timestamps must use fixed UTC−06:00 (no DST).")
    return pd.DatetimeIndex(timestamps)


def _utc_profile_calendar(source_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Validate retained weather years and label their cyclic UTC slots."""
    if source_index.empty:
        raise ValueError("No county profile hours match --max-weather-year.")
    if not source_index.is_monotonic_increasing:
        raise ValueError("County profile timestamps must be in chronological order.")

    local_hours = source_index.tz_localize(None)
    for year, hours in pd.Series(local_hours).groupby(local_hours.year, sort=False):
        # 8,760 consecutive hours from Jan 1 include Feb 29 and omit Dec 31 in leap years.
        expected = pd.date_range(f"{year}-01-01", periods=HOURS_PER_WEATHER_YEAR, freq="h")
        if not expected.equals(pd.DatetimeIndex(hours)):
            raise ValueError(
                f"County profile year {year} does not match the 8,760-hour ReEDS calendar."
            )
    return local_hours.tz_localize("UTC")


def build_distributed_profiles(
    generation_profiles_path: Union[str, Path],
    county_capacity_path: Union[str, Path],
    county_population_path: Union[str, Path, None],
    mapping_path: Union[str, Path],
    weight_year: int = 2035,
    weight_by: str = "capacity",
    max_weather_year: int = 2013,
) -> pd.DataFrame:
    """
    Build hourly BA capacity-factor profiles with timezone-aware UTC labels.

    Reads county-level capacity factors, aggregates them to BA regions using
    a capacity-weighted (or population-weighted) average, and cyclically shifts
    each weather year from fixed UTC−06:00 to UTC. Values are not rescaled.
    """
    # Defer heavy imports so script can run without them if skipping profiles
    import h5py
    from tqdm import tqdm

    county2zone = read_county2zone(mapping_path)

    with h5py.File(generation_profiles_path, "r") as f:
        source_index = _county_source_index(f["index_0"][:])
        retained = source_index.year <= max_weather_year
        utc_index = _utc_profile_calendar(source_index[retained])
        county_cf = pd.DataFrame(
            f["data"][:],
            columns=[x.decode("utf-8") for x in f["columns"][:]],
            index=source_index,
        ).loc[retained]

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
    regional_cf = (
        county_cf.mul(county_weights, axis=1)
        .rename(columns=county2zone)
        .T.groupby(level=0)
        .sum()
        .T.div(regional_weights, axis=1)
    )

    # Rotate each validated year independently, including across the year boundary.
    hours_by_year = np.arange(len(utc_index)).reshape(-1, HOURS_PER_WEATHER_YEAR)
    source_rows = np.roll(hours_by_year, -SOURCE_UTC_OFFSET_HOURS, axis=1).ravel()

    df_list = []
    for region in tqdm(regional_cf.columns, desc="Processing regions"):
        gen_profile = regional_cf[region].to_numpy()[source_rows]

        _df = pd.DataFrame(
            {
                "time_index": range(len(gen_profile)),
                "region": region,
                "weather_year": utc_index.year,
                "value": gen_profile,
                "datetime_utc": utc_index,
            }
        )
        df_list.append(_df)

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
    add_source_options(parser)
    args = parser.parse_args()
    sources = SourceLock(args.cache_dir, args.sources_lock)

    print("Building distributed capacity table...")
    county_capacity_path = sources.fetch(DISTPV_CAP_PATH)
    mapping_path = sources.fetch(COUNTY2ZONE_PATH)
    distributed_capacity = build_distributed_capacity(
        county_capacity_path=county_capacity_path,
        mapping_path=mapping_path,
        ilr=args.ilr,
    )

    # Save distributed capacity
    output_capacity_path = "distributed_capacity.parquet"
    with sources.output(output_capacity_path, (DISTPV_CAP_PATH, COUNTY2ZONE_PATH)) as output:
        distributed_capacity.to_parquet(output, index=False, compression="zstd")
    print(f"Saved distributed capacity to {output_capacity_path}")
    print(f"  Shape: {distributed_capacity.shape}")
    print(f"  Columns: {list(distributed_capacity.columns)}")
    print(f"  Sample:\n{distributed_capacity.head()}\n")

    # Build distributed profiles table
    if args.skip_profiles:
        print("Skipping distributed profiles as requested (--skip-profiles).")
    else:
        print("Building distributed profiles table...")
        distpv_h5 = sources.fetch_from_zenodo_catalog(DISTPV_CF_FILENAME, REMOTE_FILES_PATH)
        weight_pin = COUNTY_POPULATION_PATH if args.weight_by == "population" else DISTPV_CAP_PATH
        population_path = sources.fetch(weight_pin) if args.weight_by == "population" else None

        distributed_profiles = build_distributed_profiles(
            generation_profiles_path=distpv_h5,
            county_capacity_path=county_capacity_path,
            county_population_path=population_path,
            mapping_path=mapping_path,
            weight_year=args.weight_year,
            weight_by=args.weight_by,
            max_weather_year=args.max_weather_year,
        )
        # Save distributed profiles
        output_profiles_path = "distributed_profiles.parquet"
        profile_pins = (DISTPV_CF_FILENAME, REMOTE_FILES_PATH, COUNTY2ZONE_PATH, weight_pin)
        with sources.output(output_profiles_path, profile_pins) as output:
            distributed_profiles.to_parquet(output, index=False, compression="zstd")
        print(f"Saved distributed profiles to {output_profiles_path}")
        print(f"  Shape: {distributed_profiles.shape}")
        print(f"  Columns: {list(distributed_profiles.columns)}")
        print(f"  Sample:\n{distributed_profiles.head()}\n")


if __name__ == "__main__":
    main()
