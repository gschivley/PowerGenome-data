#!/usr/bin/env python3
"""
Transform ReEDS generator database to PowerGenome format.

This script:
1. Fetches ReEDS generator data from GitHub
2. Fetches EIA technology names from PUDL
3. Maps technology names with special handling for run-of-river hydro
4. Filters out retired units
5. Performs data quality checks
6. Exports to CSV matching the target format
"""

from io import StringIO

import numpy as np
import pandas as pd
import requests

# URLs
REEDS_URL = "https://raw.githubusercontent.com/NREL/ReEDS-2.0/refs/heads/main/inputs/capacity_exogenous/ReEDS_generator_database_final_EIA-NEMS.csv"
COUNTY2ZONE_URL = "https://raw.githubusercontent.com/NREL/ReEDS-2.0/refs/heads/main/inputs/county2zone.csv"
EIA_URL = "https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/nightly/out_eia__yearly_generators.parquet"


def reverse_excel_date_conversion(value, zero_pad=False, uppercase_format=False):
    """
    Reverse Excel's auto-conversion of strings like "1-2" to "2-Jan".

    Excel converts patterns like:
    - "1-2" -> "2-Jan" (original: month-day, Excel: day-month name)
    - "1-02" -> "2-Jan" (original with zero-padding)
    - "1-98" -> "Jan-98" (original: month-year)
    - "MAR1" -> "1-Mar" (original: uppercase month + day, no dash)

    Args:
        value: The Excel-converted string to reverse
        zero_pad: If True, pad numeric parts to 2 digits (e.g., "1-2" -> "01-02")
        uppercase_format: If True, return uppercase month+day format (e.g., "MAR1")
    """
    if pd.isna(value) or not isinstance(value, str):
        return value

    # Month name to number mapping
    month_map = {
        "jan": "1",
        "feb": "2",
        "mar": "3",
        "apr": "4",
        "may": "5",
        "jun": "6",
        "jul": "7",
        "aug": "8",
        "sep": "9",
        "oct": "10",
        "nov": "11",
        "dec": "12",
    }

    value_lower = value.lower().strip()

    # Pattern 1: "2-Jan" -> "1-2" or "1-02" or "MAR1" (day-month name -> various formats)
    for month_name, month_num in month_map.items():
        if f"-{month_name}" in value_lower:
            parts = value_lower.split("-")
            if len(parts) == 2 and parts[1] == month_name:
                day = parts[0]
                if uppercase_format:
                    # Return uppercase month + day with no dash (e.g., "MAR1")
                    return f"{month_name.upper()}{day}"
                elif zero_pad:
                    return f"{month_num.zfill(2)}-{day.zfill(2)}"
                else:
                    return f"{month_num}-{day}"

    # Pattern 2: "Jan-98" -> "1-98" (month name-year -> month-year)
    for month_name, month_num in month_map.items():
        if month_name in value_lower and "-" in value_lower:
            parts = value_lower.split("-")
            if len(parts) == 2 and parts[0] == month_name:
                year = parts[1]
                if zero_pad:
                    return f"{month_num.zfill(2)}-{year.zfill(2)}"
                else:
                    return f"{month_num}-{year}"

    return value


def fetch_reeds_data():
    """Fetch ReEDS generator database from GitHub."""
    print("Fetching ReEDS data...")
    response = requests.get(REEDS_URL)
    response.raise_for_status()
    # Force T_UID to be read as string to prevent further Excel date conversion
    df = pd.read_csv(StringIO(response.text), dtype={"T_UID": str})
    print(f"  Loaded {len(df):,} rows from ReEDS")
    return df


def fetch_eia_technology_names():
    """Fetch and process EIA technology names from PUDL."""
    print("Fetching EIA technology names...")
    df = pd.read_parquet(EIA_URL)
    print(f"  Loaded {len(df):,} rows from PUDL")

    # Keep only relevant columns
    df = df[
        ["plant_id_eia", "generator_id", "technology_description", "report_date"]
    ].copy()

    # Drop duplicates, keeping the most recent report_date
    df = df.sort_values("report_date", ascending=False)
    df = df.drop_duplicates(subset=["plant_id_eia", "generator_id"], keep="first")

    # Strip whitespace from generator_id to match ReEDS format
    df["generator_id"] = df["generator_id"].astype(str).str.strip().str.replace(" ", "")

    print(f"  Deduplicated to {len(df):,} unique plant/generator combinations")
    return df[["plant_id_eia", "generator_id", "technology_description"]]


def fetch_county_to_zone_mapping():
    """Fetch county to BA zone mapping from ReEDS."""
    print("Fetching county to zone mapping...")
    response = requests.get(COUNTY2ZONE_URL)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    print(f"  Loaded {len(df):,} county mappings")

    # Keep only FIPS and ba columns
    df = df[["FIPS", "ba"]].copy()

    # Convert FIPS to string with leading zeros (5 digits)
    df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)

    return df


def map_reeds_tech_to_technology(tech):
    """
    Map ReEDS tech codes to technology names.
    Used as fallback when EIA match is not found.
    """
    # Common mappings
    tech_map = {
        "gas-cc": "Natural Gas Fired Combined Cycle",
        "gas-ct": "Natural Gas Fired Combustion Turbine",
        "coaloldscr": "Conventional Steam Coal",
        "coalolduns": "Conventional Steam Coal",
        "coal-new": "Conventional Steam Coal",
        "nuclear": "Nuclear",
        "pumped-hydro": "Hydroelectric Pumped Storage",
        "o-g-s": "Natural Gas Steam Turbine",
        "biopower": "Wood/Wood Waste Biomass",
        "lfill-gas": "Landfill Gas",
        "wind-ons": "Onshore Wind Turbine",
        "wind-ofs": "Offshore Wind Turbine",
        "upv": "Solar Photovoltaic",
        "dupv": "Solar Photovoltaic",
        "csp": "Solar Thermal with Energy Storage",
        "geothermal": "Geothermal",
    }

    # Check for battery
    if "battery" in tech.lower():
        return "Batteries"

    # Check for run-of-river hydro (tech contains "ND")
    if "ND" in tech and "hyd" in tech.lower():
        return "Run of River Hydroelectric"

    # Check for conventional hydro
    if "hyd" in tech.lower() and "pumped" not in tech.lower():
        return "Conventional Hydroelectric"

    # Use mapping or return as-is
    return tech_map.get(tech, tech)


def transform_reeds_data(reeds_df, eia_tech_df):
    """Transform ReEDS data to target format."""
    print("\nTransforming data...")

    # Filter out retired units
    initial_count = len(reeds_df)
    if "status" in reeds_df.columns:
        reeds_df = reeds_df[reeds_df["status"] != "Retired"].copy()
        print(f"  Filtered out {initial_count - len(reeds_df):,} retired units")
    else:
        print("  Warning: 'status' column not found, skipping retirement filter")

    # Create base dataframe with column mapping
    transformed = pd.DataFrame()

    # Map columns
    transformed["plant_id"] = reeds_df["T_PID"]
    # Strip whitespace from generator_id to ensure clean merge with EIA data
    transformed["generator_id"] = reeds_df["T_UID"].astype(str).str.strip()
    transformed["capacity_mw"] = reeds_df["summer_power_capacity_MW"]
    transformed["nameplate_capacity_mw"] = reeds_df["TC_NP"]
    transformed["winter_capacity_mw"] = reeds_df["TC_WIN"]
    transformed["capacity_mwh"] = reeds_df.get("energy_capacity_MWh", np.nan)
    transformed["operating_year"] = reeds_df["StartYear"]
    transformed["retirement_year"] = reeds_df["RetireYear"]
    transformed["historical_capacity_factor"] = reeds_df.get("T_CF", np.nan)

    # Convert heat rate from BTU/kWh to MMBtu/MWh
    if "HeatRate" in reeds_df.columns:
        transformed["heat_rate_mmbtu_mwh"] = reeds_df["HeatRate"] / 1000
    else:
        transformed["heat_rate_mmbtu_mwh"] = np.nan

    # Map O&M costs from ReEDS columns
    if "T_VOM" in reeds_df.columns:
        transformed["vom_per_mwh"] = pd.to_numeric(reeds_df["T_VOM"], errors="coerce")
    else:
        transformed["vom_per_mwh"] = np.nan

    if "T_FOM" in reeds_df.columns:
        # ReEDS FOM is in $/kW-yr, convert to $/MW-yr
        transformed["fom_per_mwyr"] = (
            pd.to_numeric(reeds_df["T_FOM"], errors="coerce") * 1000
        )
    else:
        transformed["fom_per_mwyr"] = np.nan

    # Store ReEDS tech for fallback
    transformed["reeds_tech"] = reeds_df["tech"]

    # Merge with EIA technology names
    print("  Joining with EIA technology names...")
    transformed = transformed.merge(
        eia_tech_df,
        left_on=["plant_id", "generator_id"],
        right_on=["plant_id_eia", "generator_id"],
        how="left",
        suffixes=("", "_eia"),
    )

    # For unmatched rows, try reversing Excel date conversion and re-merge
    unmatched_mask = transformed["technology_description"].isna()
    unmatched_count_initial = unmatched_mask.sum()

    if unmatched_count_initial > 0:
        print(
            f"  Attempting Excel date reversal for {unmatched_count_initial:,} unmatched generators..."
        )

        # For each unmatched row, try reversing Excel date conversion with various padding combinations
        matched_count = 0
        for idx in transformed[unmatched_mask].index:
            original_id = str(transformed.at[idx, "generator_id"])
            plant_id = int(transformed.at[idx, "plant_id"])

            # Generate all possible reversed formats
            candidate_ids = []

            # Try base reversal (no padding)
            base_reversed = reverse_excel_date_conversion(original_id, zero_pad=False)
            if base_reversed != original_id:
                candidate_ids.append(base_reversed)

                # If it's a month-day pattern (e.g., "1-2"), try partial padding variants
                if "-" in base_reversed:
                    parts = base_reversed.split("-")
                    if len(parts) == 2:
                        month, day = parts
                        # Try: month-day (no pad), month-dayPadded, monthPadded-day, monthPadded-dayPadded
                        candidate_ids.extend(
                            [
                                f"{month}-{day}",  # Already added as base_reversed
                                f"{month}-{day.zfill(2)}",  # Pad day only
                                f"{month.zfill(2)}-{day}",  # Pad month only
                                f"{month.zfill(2)}-{day.zfill(2)}",  # Pad both
                            ]
                        )

                # Also try uppercase month+day format (e.g., "1-Mar" -> "MAR1")
                uppercase_variant = reverse_excel_date_conversion(
                    original_id, zero_pad=False, uppercase_format=True
                )
                if (
                    uppercase_variant != original_id
                    and uppercase_variant not in candidate_ids
                ):
                    candidate_ids.append(uppercase_variant)

            # Remove duplicates while preserving order
            seen = set()
            unique_candidates = []
            for cand in candidate_ids:
                if cand not in seen:
                    seen.add(cand)
                    unique_candidates.append(cand)

            # Try each candidate ID
            for candidate_id in unique_candidates:
                match = eia_tech_df[
                    (eia_tech_df["plant_id_eia"] == plant_id)
                    & (eia_tech_df["generator_id"] == candidate_id)
                ]

                if not match.empty:
                    transformed.at[idx, "technology_description"] = match.iloc[0][
                        "technology_description"
                    ]
                    transformed.at[idx, "generator_id"] = candidate_id
                    matched_count += 1
                    break  # Stop after first match

        if matched_count > 0:
            print(
                f"    - Matched {matched_count:,} additional generators after correction"
            )

    # Handle run-of-river hydro special case
    # If ReEDS tech contains "ND", override EIA technology name
    run_of_river_mask = transformed["reeds_tech"].str.contains(
        "ND", na=False
    ) & transformed["reeds_tech"].str.contains("hyd", case=False, na=False)

    # Apply technology mapping
    def get_technology(row):
        # First check for run-of-river override
        if "ND" in str(row["reeds_tech"]) and "hyd" in str(row["reeds_tech"]).lower():
            return "Run of River Hydroelectric"
        # Then use EIA technology if available
        elif pd.notna(row["technology_description"]):
            return row["technology_description"]
        # Fall back to mapped ReEDS tech
        else:
            return map_reeds_tech_to_technology(row["reeds_tech"])

    transformed["technology"] = transformed.apply(get_technology, axis=1)

    # Reorder columns and drop helper columns
    output_columns = [
        "technology",
        "plant_id",
        "generator_id",
        "capacity_mw",
        "capacity_mwh",
        "nameplate_capacity_mw",
        "winter_capacity_mw",
        "operating_year",
        "retirement_year",
        "historical_capacity_factor",
        "heat_rate_mmbtu_mwh",
        "vom_per_mwh",
        "fom_per_mwyr",
    ]

    result = transformed[output_columns].copy()

    # Calculate join statistics
    total_rows = len(transformed)
    eia_matched = transformed["technology_description"].notna().sum()
    fallback_used = total_rows - eia_matched
    run_of_river_count = run_of_river_mask.sum()

    print(f"  Technology name sources:")
    print(f"    - EIA matched: {eia_matched:,} ({eia_matched/total_rows*100:.1f}%)")
    print(f"    - Run-of-river override: {run_of_river_count:,}")
    print(
        f"    - ReEDS fallback: {fallback_used:,} ({fallback_used/total_rows*100:.1f}%)"
    )

    return result


def perform_data_quality_checks(df):
    """Perform data quality checks and report issues."""
    print("\nData Quality Checks:")

    issues = []

    # Check for missing required fields
    required_fields = ["technology", "plant_id", "generator_id", "capacity_mw"]
    for field in required_fields:
        missing = df[field].isna().sum()
        if missing > 0:
            issues.append(f"  ⚠️  {missing:,} rows missing {field}")

    # Check capacity > 0
    if "capacity_mw" in df.columns:
        invalid_capacity = (df["capacity_mw"] <= 0).sum()
        if invalid_capacity > 0:
            issues.append(f"  ⚠️  {invalid_capacity:,} rows with capacity_mw <= 0")

    # Check operating year range
    if "operating_year" in df.columns:
        out_of_range = (
            (df["operating_year"] < 1800) | (df["operating_year"] > 2030)
        ).sum()
        if out_of_range > 0:
            issues.append(
                f"  ⚠️  {out_of_range:,} rows with operating_year outside 1800-2030"
            )

    # Check retirement year
    if "retirement_year" in df.columns and "operating_year" in df.columns:
        invalid_retire = (df["retirement_year"] <= df["operating_year"]).sum()
        if invalid_retire > 0:
            issues.append(
                f"  ⚠️  {invalid_retire:,} rows with retirement_year <= operating_year"
            )

        far_future = (df["retirement_year"] > 2150).sum()
        if far_future > 0:
            issues.append(f"  ⚠️  {far_future:,} rows with retirement_year > 2150")

    # Check capacity factor range
    if "historical_capacity_factor" in df.columns:
        invalid_cf = (
            (df["historical_capacity_factor"] < 0)
            | (df["historical_capacity_factor"] > 1)
        ).sum()
        if invalid_cf > 0:
            issues.append(
                f"  ⚠️  {invalid_cf:,} rows with capacity_factor outside [0, 1]"
            )

    # Check heat rate
    if "heat_rate_mmbtu_mwh" in df.columns:
        negative_hr = (df["heat_rate_mmbtu_mwh"] < 0).sum()
        if negative_hr > 0:
            issues.append(f"  ⚠️  {negative_hr:,} rows with negative heat_rate")

    if issues:
        print("  Issues found:")
        for issue in issues:
            print(issue)
    else:
        print("  ✓ All quality checks passed!")

    # Summary statistics by technology
    print("\nSummary by Technology:")
    tech_summary = (
        df.groupby("technology")
        .agg(
            {"capacity_mw": ["count", "sum", "mean"], "operating_year": ["min", "max"]}
        )
        .round(2)
    )

    tech_summary.columns = ["Count", "Total_MW", "Avg_MW", "Min_Year", "Max_Year"]
    tech_summary = tech_summary.sort_values("Count", ascending=False)
    print(tech_summary.head(20))

    return len(issues) == 0


def create_plant_region_map(reeds_df, county_zone_df):
    """
    Create plant to region mapping.

    Args:
        reeds_df: ReEDS generator dataframe with FIPS and T_PID columns
        county_zone_df: County to zone mapping with FIPS and ba columns

    Returns:
        DataFrame with plant_id and region columns
    """
    print("\nCreating plant-region mapping...")

    # Extract unique plant-FIPS mappings from ReEDS data
    plant_fips = reeds_df[["T_PID", "FIPS"]].copy()
    plant_fips = plant_fips.rename(columns={"T_PID": "plant_id"})

    # Strip 'p' prefix from FIPS in ReEDS data if present
    plant_fips["FIPS"] = (
        plant_fips["FIPS"].astype(str).str.replace("^p", "", regex=True).str.zfill(5)
    )

    # Drop duplicates
    plant_fips = plant_fips.drop_duplicates(subset=["plant_id"])

    # Merge with county to zone mapping
    plant_region = plant_fips.merge(county_zone_df, on="FIPS", how="left")

    # Rename ba to region
    plant_region = plant_region.rename(columns={"ba": "region"})

    # Select final columns
    plant_region = plant_region[["plant_id", "region"]]

    # Check for unmapped plants
    unmapped = plant_region["region"].isna().sum()
    if unmapped > 0:
        print(f"  ⚠️  Warning: {unmapped:,} plants could not be mapped to a region")

    print(f"  Created mapping for {len(plant_region):,} plants")
    print(f"  Unique regions: {plant_region['region'].nunique()}")

    return plant_region


def main():
    """Main execution function."""
    print("=" * 60)
    print("ReEDS to PowerGenome Generator Data Transformation")
    print("=" * 60)

    # Fetch data
    reeds_df = fetch_reeds_data()
    eia_tech_df = fetch_eia_technology_names()
    county_zone_df = fetch_county_to_zone_mapping()

    # Transform
    transformed_df = transform_reeds_data(reeds_df, eia_tech_df)

    # Create plant-region mapping
    plant_region_map = create_plant_region_map(reeds_df, county_zone_df)

    # Quality checks
    all_checks_passed = perform_data_quality_checks(transformed_df)

    # Export generators
    output_file = "reeds_generators_transformed.csv"
    transformed_df.to_csv(output_file, index=False)
    print(f"\n✓ Exported {len(transformed_df):,} rows to {output_file}")

    # Export plant-region mapping
    region_map_file = "plant_region_map.csv"
    plant_region_map.to_csv(region_map_file, index=False)
    print(
        f"✓ Exported {len(plant_region_map):,} plant-region mappings to {region_map_file}"
    )

    # Verify all plants in generators file are in plant-region mapping
    print("\nVerifying plant-region mapping coverage...")
    generator_plants = set(transformed_df["plant_id"].unique())
    mapped_plants = set(plant_region_map["plant_id"].unique())

    missing_plants = generator_plants - mapped_plants
    if missing_plants:
        print(
            f"  ❌ ERROR: {len(missing_plants):,} plants in generators file are missing from plant-region mapping:"
        )
        print(
            f"     Missing plant IDs: {sorted(list(missing_plants))[:10]}{'...' if len(missing_plants) > 10 else ''}"
        )
        all_checks_passed = False
    else:
        print(
            f"  ✓ All {len(generator_plants):,} plants in generators file are mapped to regions"
        )

    if not all_checks_passed:
        print("\n⚠️  Warning: Some data quality issues were detected. Review above.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
