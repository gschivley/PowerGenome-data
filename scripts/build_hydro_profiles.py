"""
Generate hourly hydroelectric generation profiles by region from 2007-13.

This script:
1. Merges plant-region mappings with generator data to identify hydro plants
2. Queries PUDL monthly EIA data from S3 to calculate monthly capacity factors
3. Fills every month of 2007-13 for each region/type (see below)
4. Interpolates to hourly values and applies 1-week moving average smoothing
5. Checks that every site has a full, nonzero 8760-hour series for every year
6. Outputs tidy parquet files with site_id, weather_year, time_index, value
7. Writes one resource-group metadata CSV per hydro type (region, capacity_mw, id)

Months come from, in order of preference:

* ``reported``: the region's own 2007-13 data. Generators with no reported
  generation in a month are left out of that month's capacity factor rather than
  counted as zero output. If at least ``MIN_WEATHER_MONTH_SHARE`` of the months
  are reported, gaps are filled with the region's mean for that calendar month.
* ``later_years``: the region has too little 2007-13 data (e.g. its plants came
  online or began reporting later). Each calendar month uses the mean capacity
  factor of the same plants in FALLBACK_START_YEAR-FALLBACK_END_YEAR, repeated in
  every weather year.
* ``donor_state`` / ``donor_national``: no usable data at all. The capacity-weighted
  monthly capacity factor of regions of the same type with reported data, in
  the same state if there are any, otherwise nationally.

Conventional and run-of-river hydro are separate PowerGenome resource groups, so
each needs its own metadata. Sharing one file makes each group request sites that
only exist in the other group's profiles, and overstates each group's capacity.

Pass ``--metadata-only`` to rebuild just the metadata from local files (no PUDL).
"""

import argparse
import logging
from datetime import timedelta
from pathlib import Path

import duckdb
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# PUDL data is published to a public S3 bucket. DuckDB needs these settings to
# load the httpfs extension and reach the bucket. The bucket currently lives in
# us-west-2; if this ever changes, DuckDB will surface a 301 with the correct
# region to use here.
PUDL_S3_SETTINGS = [
    "INSTALL httpfs",
    "LOAD httpfs",
    "SET s3_region='us-west-2'",
    "SET s3_use_ssl=true",
    "SET s3_url_style='path'",
]

# Constants
PUDL_PARQUET_URL = "s3://pudl.catalyst.coop/nightly/out_eia__monthly_generators.parquet"
START_YEAR = 2007
END_YEAR = 2013
HOURS_PER_WEEK = 168
HOURS_PER_YEAR = 8760

# Years used to build a monthly shape for regions without usable 2007-13 data.
# Fixed (not "latest") so rebuilds from a newer PUDL nightly stay comparable.
FALLBACK_START_YEAR = 2014
FALLBACK_END_YEAR = 2025
# Share of the 2007-13 months a region must report to use its own data.
MIN_WEATHER_MONTH_SHARE = 0.75

HIERARCHY_PATH = Path("cache_data/hierarchy.csv")

# Hydro generator types to include
HYDRO_TYPES = {
    "run_of_river": "Run of River Hydroelectric",
    "conventional": "Conventional Hydroelectric",
}


METADATA_FILENAME = "existing_hydro_{hydro_type}_reeds_ba_metadata.csv"


def hours_in_month(report_date) -> float:
    """Number of real calendar hours in the month of ``report_date``.

    December must return 744/736 hours like any other month. The original
    inline lambda built December's "next month" as Jan 1 of the *same* year,
    producing a negative divisor that flipped the capacity factor's sign and
    zeroed out every December after the 0-1 clip.
    """
    ts = pd.Timestamp(report_date)
    month_start = pd.Timestamp(ts.year, ts.month, 1)
    next_month_start = month_start + pd.DateOffset(months=1)
    return (next_month_start - month_start).total_seconds() / 3600


def load_plant_region_data() -> pd.DataFrame:
    """Load plant-to-region mapping."""
    logger.info("Loading plant-region mappings...")
    return pd.read_csv("data/plant_region_map.csv")


def load_generator_data() -> pd.DataFrame:
    """Load transformed REEDS generators and filter for hydroelectric plants."""
    logger.info("Loading generator data...")
    generators = pd.read_csv("data/reeds_generators_transformed.csv")

    # Filter for hydroelectric plants only
    hydro_mask = generators["technology"].isin(HYDRO_TYPES.values())
    generators = generators[hydro_mask].copy()

    # Classify by hydro type
    generators["hydro_type"] = generators["technology"].map(
        {v: k for k, v in HYDRO_TYPES.items()}
    )

    logger.info(f"Found {len(generators)} hydroelectric generators")
    return generators[["plant_id", "hydro_type"]].drop_duplicates()


def build_hydro_metadata(
    generators: pd.DataFrame, plant_region: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Operating hydro capacity by region for each hydro type.

    Returns ``{hydro_type: DataFrame[region, capacity_mw, id]}`` where ``id`` is the
    profile ``site_id`` (the region). Regions with no capacity of a type are left
    out, matching the regions that get a profile for that type.
    """
    hydro = generators.loc[
        generators["technology"].isin(HYDRO_TYPES.values()),
        ["plant_id", "technology", "capacity_mw"],
    ].merge(plant_region[["plant_id", "region"]], on="plant_id", how="left")
    unmapped = hydro["region"].isna()
    if unmapped.any():
        raise ValueError(
            "Hydro plants missing from plant_region_map: "
            f"{sorted(hydro.loc[unmapped, 'plant_id'].unique())}"
        )

    metadata = {}
    for hydro_type, technology in HYDRO_TYPES.items():
        cap = (
            hydro.loc[hydro["technology"] == technology]
            .groupby("region", as_index=False)["capacity_mw"]
            .sum()
        )
        cap = cap.loc[cap["capacity_mw"] > 0].copy()
        cap["capacity_mw"] = cap["capacity_mw"].round(1)
        cap["id"] = cap["region"]
        cap["_n"] = cap["region"].str.extract(r"(\d+)$", expand=False).astype(float)
        metadata[hydro_type] = (
            cap.sort_values(["_n", "region"]).drop(columns="_n").reset_index(drop=True)
        )
    return metadata


def write_hydro_metadata(output_dir: Path = Path("existing_resource_groups")) -> None:
    """Build and save one metadata CSV per hydro type from local inputs."""
    output_dir.mkdir(exist_ok=True)
    generators = pd.read_csv("data/reeds_generators_transformed.csv")
    metadata = build_hydro_metadata(generators, load_plant_region_data())
    for hydro_type, df in metadata.items():
        output_file = output_dir / METADATA_FILENAME.format(hydro_type=hydro_type)
        df.to_csv(output_file, index=False)
        logger.info(
            f"Saved {hydro_type} metadata to {output_file} "
            f"({len(df)} regions, {df['capacity_mw'].sum():,.1f} MW)"
        )


def merge_plant_region_hydro_data() -> pd.DataFrame:
    """Merge plant-region and generator data."""
    logger.info("Merging plant-region and generator data...")

    plant_region = load_plant_region_data()
    generators = load_generator_data()

    # Merge on plant_id
    merged = plant_region.merge(generators, on="plant_id", how="inner")

    logger.info(f"Merged data has {len(merged)} plant-region-hydro_type combinations")
    return merged[["plant_id", "region", "hydro_type"]].drop_duplicates()


def query_pudl_monthly_capacity_factors(
    parquet_url: str = PUDL_PARQUET_URL,
    s3_settings: list[str] | None = None,
) -> pd.DataFrame:
    """
    Query PUDL EIA monthly generators data to calculate regional monthly capacity factors.

    Args:
        parquet_url: Location of the PUDL monthly generators parquet. Defaults
            to the public S3 nightly path; tests pass a local file.
        s3_settings: DuckDB statements to run before the query (httpfs setup
            for S3). ``None`` uses ``PUDL_S3_SETTINGS``; an empty list skips
            them for local files.

    Returns:
        DataFrame with columns: region, hydro_type, report_date, capacity_factor,
        capacity_mw for START_YEAR through FALLBACK_END_YEAR. ``capacity_mw`` is the
        capacity of the generators that reported generation that month.
    """
    if s3_settings is None:
        s3_settings = PUDL_S3_SETTINGS
    logger.info(
        f"Querying PUDL parquet from {parquet_url} for "
        f"{START_YEAR}-{FALLBACK_END_YEAR}..."
    )

    # Get plant-region mappings
    plant_hydro_map = merge_plant_region_hydro_data()

    # Use DuckDB to query parquet with filtering
    conn = duckdb.connect()
    for setting in s3_settings:
        conn.execute(setting)

    # Read PUDL data and join with region mapping in DuckDB.
    #
    # Hydro profiles should reflect only hydroelectric generation, so we
    # restrict to generators with prime_mover_code = 'HY'. Some hydro plants
    # share a plant_id with co-located pumped-storage ('PS', which reports
    # negative net generation) or thermal ('GT'/'IC'/'ST') generators. Since
    # net_generation_mwh is summed across the whole plant, including those
    # generators corrupts the hydro capacity factor and (for pumped storage)
    # can even drive it to zero via the clip below.
    #
    # Generators that did not report generation for a month (NULL) are dropped
    # so their capacity does not count as zero output in that month.
    pudl_data = conn.execute(
        f"""
        SELECT 
            pg.plant_id_eia,
            pg.report_date,
            pg.net_generation_mwh,
            pg.capacity_mw
        FROM 
            read_parquet('{parquet_url}') pg
        WHERE 
            EXTRACT(YEAR FROM pg.report_date) >= {START_YEAR}
            AND EXTRACT(YEAR FROM pg.report_date) <= {FALLBACK_END_YEAR}
            AND pg.capacity_mw > 0
            AND pg.prime_mover_code = 'HY'
            AND pg.net_generation_mwh IS NOT NULL
    """
    ).df()

    logger.info(f"Retrieved {len(pudl_data)} records from PUDL")

    # Convert plant_id to match our mapping
    pudl_data["plant_id"] = pudl_data["plant_id_eia"].astype(int)

    # Join with plant-region mapping
    merged = pudl_data.merge(plant_hydro_map, on="plant_id", how="inner")

    logger.info(f"Matched {len(merged)} records to regions")

    # Calculate monthly capacity factors by region and hydro type
    monthly_cf = (
        merged.groupby(["region", "hydro_type", "report_date"])
        .agg({"net_generation_mwh": "sum", "capacity_mw": "sum"})
        .reset_index()
    )

    # Capacity factor = monthly generation / (capacity * hours in month),
    # using true calendar-month hours (see hours_in_month for the December
    # regression this guards against).
    monthly_cf["hours_in_month"] = monthly_cf["report_date"].apply(hours_in_month)

    monthly_cf["capacity_factor"] = (
        monthly_cf["net_generation_mwh"]
        / (monthly_cf["capacity_mw"] * monthly_cf["hours_in_month"])
    ).clip(0, 1)  # Ensure 0-1 range

    monthly_cf = monthly_cf[
        ["region", "hydro_type", "report_date", "capacity_factor", "capacity_mw"]
    ]

    logger.info(f"Calculated {len(monthly_cf)} monthly capacity factors")
    return monthly_cf


def _calendar_month_means(cf: pd.DataFrame) -> pd.Series:
    """Mean capacity factor by calendar month (1-12)."""
    return cf.groupby(cf["report_date"].dt.month)["capacity_factor"].mean()


def _weighted_monthly_cf(cf: pd.DataFrame) -> pd.Series:
    """Capacity-weighted capacity factor by report_date across regions."""
    weighted = cf["capacity_factor"] * cf["capacity_mw"]
    grouped = pd.DataFrame(
        {"report_date": cf["report_date"], "w": weighted, "cap": cf["capacity_mw"]}
    ).groupby("report_date")
    return grouped["w"].sum() / grouped["cap"].sum()


def complete_weather_year_months(
    monthly_cf: pd.DataFrame,
    sites: pd.DataFrame,
    region_state: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a capacity factor for every 2007-13 month of every site.

    Args:
        monthly_cf: Output of :func:`query_pudl_monthly_capacity_factors`.
        sites: One row per (region, hydro_type) that needs a profile.
        region_state: Region -> state, used to pick donor regions.

    Returns:
        ``(completed, methods)``. ``completed`` has columns region, hydro_type,
        report_date, capacity_factor with 12 * (END_YEAR - START_YEAR + 1) rows per
        site. ``methods`` has one row per site with the ``method`` used (see the
        module docstring) and ``reported_months``.
    """
    region_state = region_state or {}
    weather_months = pd.date_range(
        f"{START_YEAR}-01-01", f"{END_YEAR}-12-01", freq="MS"
    )
    in_window = monthly_cf["report_date"].dt.year.between(START_YEAR, END_YEAR)
    window = monthly_cf.loc[in_window]
    later = monthly_cf.loc[
        monthly_cf["report_date"].dt.year.between(
            FALLBACK_START_YEAR, FALLBACK_END_YEAR
        )
    ]

    def usable(cf: pd.DataFrame) -> bool:
        return (
            len(cf) >= MIN_WEATHER_MONTH_SHARE * len(weather_months)
            and cf["capacity_factor"].mean() > 0
        )

    own = {key: cf for key, cf in window.groupby(["region", "hydro_type"])}
    usable_keys = {key for key, cf in own.items() if usable(cf)}

    completed, methods = [], []
    for region, hydro_type in (
        sites[["region", "hydro_type"]].drop_duplicates().itertuples(index=False)
    ):
        key = (region, hydro_type)
        reported = own.get(key, window.iloc[0:0])
        later_cf = later.loc[
            (later["region"] == region) & (later["hydro_type"] == hydro_type)
        ]
        later_means = _calendar_month_means(later_cf)

        if key in usable_keys:
            values = reported.set_index("report_date")["capacity_factor"].reindex(
                weather_months
            )
            fill = (
                weather_months.month.map(_calendar_month_means(reported))
                .to_series(index=weather_months)
                .fillna(
                    pd.Series(
                        weather_months.month.map(later_means), index=weather_months
                    )
                )
                .fillna(reported["capacity_factor"].mean())
            )
            values = values.fillna(fill)
            method = "reported"
        elif len(later_means) == 12 and later_means.mean() > 0:
            values = pd.Series(
                weather_months.month.map(later_means).to_numpy(), index=weather_months
            )
            method = "later_years"
        else:
            state = region_state.get(region)
            donors = [
                k
                for k in usable_keys
                if k[1] == hydro_type
                and k[0] != region
                and state is not None
                and region_state.get(k[0]) == state
            ]
            method = "donor_state"
            if not donors:
                donors = [
                    k for k in usable_keys if k[1] == hydro_type and k[0] != region
                ]
                method = "donor_national"
            if not donors:
                raise ValueError(f"No data or donor regions for {region}/{hydro_type}")
            donor_cf = pd.concat([own[k] for k in donors])
            values = (
                _weighted_monthly_cf(donor_cf)
                .reindex(weather_months)
                .fillna(
                    pd.Series(
                        weather_months.month.map(_calendar_month_means(donor_cf)),
                        index=weather_months,
                    )
                )
            )

        if values.isna().any():
            raise ValueError(f"Could not fill every month for {region}/{hydro_type}")
        if method != "reported":
            logger.info(f"{region}/{hydro_type}: using {method} capacity factors")
        completed.append(
            pd.DataFrame(
                {
                    "region": region,
                    "hydro_type": hydro_type,
                    "report_date": weather_months,
                    "capacity_factor": values.to_numpy(),
                }
            )
        )
        methods.append(
            {
                "region": region,
                "hydro_type": hydro_type,
                "method": method,
                "reported_months": len(reported),
            }
        )

    return pd.concat(completed, ignore_index=True), pd.DataFrame(methods)


def check_profile_coverage(profiles: pd.DataFrame, site_ids) -> None:
    """Raise if any site lacks a full, nonzero series for every weather year.

    Mirrors PowerGenome's resource-group coverage check, so gaps fail the build
    instead of a model run.
    """
    years = range(START_YEAR, END_YEAR + 1)
    expected_index = set(range(1, HOURS_PER_YEAR + 1))
    problems = []
    for site in site_ids:
        site_rows = profiles.loc[profiles["site_id"] == site]
        if site_rows.empty:
            problems.append(f"{site}: no profile")
            continue
        for year in years:
            year_rows = site_rows.loc[site_rows["weather_year"] == year, "time_index"]
            if len(year_rows) != HOURS_PER_YEAR or set(year_rows) != expected_index:
                problems.append(f"{site}/{year}: {len(year_rows)} hours")
        if not site_rows["value"].between(0, 1).all():
            problems.append(f"{site}: values outside 0-1")
        if site_rows["value"].mean() <= 0:
            problems.append(f"{site}: all-zero profile")
    if problems:
        raise ValueError("Incomplete hydro profiles: " + "; ".join(problems))


def interpolate_monthly_to_hourly(
    monthly_cf: pd.DataFrame, region: str, hydro_type: str
) -> pd.DataFrame:
    """
    Convert monthly capacity factors to hourly values.

    Repeats each monthly value for all hours in that month, then applies
    1-week moving average smoothing. Removes Dec 31 from leap years to ensure
    8760 hours per year.
    """
    region_data = (
        monthly_cf[
            (monthly_cf["region"] == region) & (monthly_cf["hydro_type"] == hydro_type)
        ]
        .sort_values("report_date")
        .copy()
    )

    if len(region_data) == 0:
        return None

    # Create date range covering all months
    date_range = pd.date_range(
        start=region_data["report_date"].min(),
        end=region_data["report_date"].max(),
        freq="MS",  # Month start
    )

    hourly_list = []

    for i, month_start in enumerate(date_range):
        # Get next month start for day count
        if i < len(date_range) - 1:
            month_end = date_range[i + 1]
        else:
            month_end = month_start + pd.DateOffset(months=1)

        # Days in this month
        days_in_month = (month_end - month_start).days

        # For December in leap years, use 30 days instead of 31 to ensure 8760 hours/year
        is_leap_year = month_start.year % 4 == 0 and (
            month_start.year % 100 != 0 or month_start.year % 400 == 0
        )
        if month_start.month == 12 and is_leap_year:
            days_in_month = 30

        hours_in_month = days_in_month * 24

        # Get capacity factor for this month
        month_cf = region_data[region_data["report_date"] == month_start]
        if len(month_cf) == 0:
            # Dropping the month silently would shift every later hour onto
            # the wrong time_index, so make the gap visible in the logs.
            logger.warning(
                f"{region}/{hydro_type}: no capacity factor for "
                f"{month_start:%Y-%m}; skipping month"
            )
            continue

        cf_value = month_cf["capacity_factor"].values[0]

        # Create hourly values for this month
        for hour in range(hours_in_month):
            hourly_list.append(
                {
                    "report_date": month_start + timedelta(hours=hour),
                    "capacity_factor": cf_value,
                }
            )

    hourly_df = pd.DataFrame(hourly_list)

    # Apply 1-week (168-hour) moving average
    hourly_df["capacity_factor"] = (
        hourly_df["capacity_factor"]
        .rolling(window=HOURS_PER_WEEK, center=True, min_periods=1)
        .mean()
    )

    # Forward fill first 84 hours (half week), backward fill last 84 hours
    hourly_df["capacity_factor"] = hourly_df["capacity_factor"].ffill().bfill()

    return hourly_df


def extract_weather_years_and_time_indices(
    hourly_data: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Extract weather_year and time_index (1-8760) from hourly timestamps."""
    weather_year = hourly_data["report_date"].dt.year

    # Calculate day of year and hour
    day_of_year = hourly_data["report_date"].dt.dayofyear
    hour_of_day = hourly_data["report_date"].dt.hour + 1  # 1-24

    # Time index: (day - 1) * 24 + hour
    time_index = (day_of_year - 1) * 24 + hour_of_day

    return weather_year, time_index


def build_hydro_profiles():
    """Main function to build and save hydroelectric generation profiles."""

    try:
        # Query monthly capacity factors
        monthly_cf = query_pudl_monthly_capacity_factors()

        if len(monthly_cf) == 0:
            logger.warning(
                "No PUDL data matched. Ensure S3 access and parquet file exists."
            )
            return

        # Every site in the metadata needs a profile, whether or not it has
        # usable PUDL data for 2007-13.
        metadata = build_hydro_metadata(
            pd.read_csv("data/reeds_generators_transformed.csv"),
            load_plant_region_data(),
        )
        sites = pd.concat(
            [df.assign(hydro_type=t) for t, df in metadata.items()],
            ignore_index=True,
        )
        hierarchy = pd.read_csv(HIERARCHY_PATH)
        monthly_cf, methods = complete_weather_year_months(
            monthly_cf, sites, dict(zip(hierarchy["ba"], hierarchy["st"]))
        )
        filled = methods.loc[methods["method"] != "reported"]
        if not filled.empty:
            logger.info(
                "Sites without usable 2007-13 data:\n" + filled.to_string(index=False)
            )

        regions = monthly_cf["region"].unique()
        hydro_types = ["run_of_river", "conventional"]

        # Ensure output directory exists
        output_dir = Path("existing_resource_groups")
        output_dir.mkdir(exist_ok=True)

        # Build profiles for each hydro type
        for hydro_type in hydro_types:
            logger.info(f"\nBuilding {hydro_type} hydroelectric profiles...")

            all_hourly_data = []

            for region in tqdm(regions, desc=f"Processing {hydro_type} regions"):
                hourly_data = interpolate_monthly_to_hourly(
                    monthly_cf, region, hydro_type
                )

                if hourly_data is None or len(hourly_data) == 0:
                    continue

                # Extract weather_year and time_index
                weather_year, time_index = extract_weather_years_and_time_indices(
                    hourly_data
                )

                hourly_data["site_id"] = region
                hourly_data["weather_year"] = weather_year
                hourly_data["time_index"] = time_index
                hourly_data["value"] = hourly_data["capacity_factor"]

                # Select final columns
                hourly_data = hourly_data[
                    ["site_id", "weather_year", "time_index", "value"]
                ]

                all_hourly_data.append(hourly_data)

            if not all_hourly_data:
                logger.warning(f"No data for {hydro_type}. Skipping.")
                continue

            # Combine all regions
            output_df = pd.concat(all_hourly_data, ignore_index=True)

            # Validate: no nulls, time_index 1-8760, groups by site_id/weather_year
            null_count = output_df.isnull().sum().sum()
            if null_count > 0:
                logger.warning(f"{hydro_type}: Found {null_count} null values")
            check_profile_coverage(output_df, metadata[hydro_type]["id"])

            # Save as parquet
            output_file = output_dir / f"hydro_{hydro_type}_2007_2013.parquet"
            output_df.to_parquet(output_file, compression="zstd")

            logger.info(f"\nSaved {hydro_type} profiles to {output_file}")
            logger.info(f"  Rows: {len(output_df)}")
            logger.info(f"  Regions: {output_df['site_id'].nunique()}")
            logger.info(f"  Years: {sorted(output_df['weather_year'].unique())}")
            logger.info(f"  Nulls: {null_count}")
            logger.info(
                f"  Value range: {output_df['value'].min():.3f} - {output_df['value'].max():.3f}"
            )

    except Exception as e:
        logger.error(f"Error building hydro profiles: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only rebuild the per-type metadata CSVs from local files.",
    )
    args = parser.parse_args()
    if not args.metadata_only:
        build_hydro_profiles()
    write_hydro_metadata()
