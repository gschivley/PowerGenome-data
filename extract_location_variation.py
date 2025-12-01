#!/usr/bin/env python3
"""
Extract location variation data from EIA capital cost PDF.

This script:
1. Reads Tables 1-1 through 1-19 from the PDF (pages 162-180)
2. Extracts technology-specific location variation multipliers by city
3. Combines with ReEDS region-to-city mapping
4. Outputs location variation values for each technology in each region
"""

import json
from pathlib import Path

import pandas as pd
import pdfplumber

# Paths
PDF_PATH = Path(
    "cache/capital_cost_AEO2025.pdf"
)  # https://www.eia.gov/analysis/studies/powerplants/capitalcost/pdf/capital_cost_AEO2025.pdf
REGION_CITY_MAP_PATH = Path("cache/reeds_region_to_eia_city.csv")
OUTPUT_PATH = Path("data/regional_cost_multipliers.csv")

# Table configuration: page number and technology name
# Based on actual table titles from EIA capital cost study PDF
TABLES = {
    162: "Ultra-Supercritical Coal w/o Carbon Capture",
    163: "Ultra-Supercritical Coal 95% Carbon Capture",
    164: "Combustion Turbine - Simple Cycle (Aeroderivative)",
    165: "Combustion Turbine - Simple Cycle",
    166: "Combined-Cycle 2x2x1",
    167: "Combined-Cycle 1x1x1, Single Shaft",
    168: "Combined Cycle 1x1x1, Single Shaft 95% Carbon Capture",
    169: "Bio Energy 95% Carbon Capture",
    170: "Advanced Nuclear (Brownfield)",
    171: "Small Modular Reactor Nuclear Power Plant",
    172: "Geothermal",
    173: "Hydroelectric Power Plant",
    174: "Onshore Wind - Large Plant Footprint: Great Plains Region",
    175: "Onshore Wind Repowering/Retrofit",
    176: "Fixed-bottom Offshore Wind",
    177: "Solar PV w/ Single Axis Tracking",
    178: "Solar PV w/ Single Axis Tracking + AC Coupled Battery Storage",
    179: "Solar PV, Single-Axis Tracking (1.6 ILR) with Battery Hybrid",
    180: "Battery Storage: 4 hours",
}

atb_tech_map = {
    "Ultra-Supercritical Coal w/o Carbon Capture": [
        "Coal_new",
    ],
    "Ultra-Supercritical Coal 95% Carbon Capture": [
        "Coal_95%-CCS",
        "Coal_99%-CCS",
        "Coal_Retrofits_95%-CCS",
        "Coal_Retrofits_90%-CCS",
    ],
    "Combustion Turbine - Simple Cycle": [
        "NaturalGas_Combustion Turbine",
        "NaturalGas_F-Frame CT",
    ],
    "Combined-Cycle 2x2x1": [
        "NaturalGas_Combined Cycle 2-on-1",
        "NaturalGas_F-Frame CC",
        "NaturalGas_H-Frame CC",
    ],
    "Combined-Cycle 1x1x1, Single Shaft": [
        "NaturalGas_Combined Cycle 1-on-1",
    ],
    "Combined Cycle 1x1x1, Single Shaft 95% Carbon Capture": [
        "NaturalGas_Combined Cycle 1-on-1 (H-Frame) 95% CCS",
        "NaturalGas_Combined Cycle 1-on-1 (H-Frame) 97% CCS",
        "NaturalGas_Combined Cycle 2-on-1 (F-Frame) 95% CCS",
        "NaturalGas_Combined Cycle 2-on-1 (F-Frame) 97% CCS",
        "NaturalGas_Combined Cycle 2-on-1 (H-Frame) 95% CCS",
        "NaturalGas_Combined Cycle 2-on-1 (H-Frame) 97% CCS",
        "NaturalGas_F-Frame CC 95% CCS",
        "NaturalGas_H-Frame CC 95% CCS",
        "NaturalGas_F-Frame CC 97% CCS",
        "NaturalGas_H-Frame CC 97% CCS",
    ],
    "Bio Energy 95% Carbon Capture": [
        "Biopower_Dedicated",
    ],
    "Advanced Nuclear (Brownfield)": [
        "Nuclear_Nuclear",
        "Nuclear_Nuclear - Large",
    ],
    "Small Modular Reactor Nuclear Power Plant": [
        "Nuclear_SMR",
        "Nuclear_Nuclear - Small",
    ],
    "Geothermal": [
        "Geothermal",
    ],
    "Hydroelectric Power Plant": [
        "Hydropower",
        "Pumped Storage Hydropower",
    ],
    "Onshore Wind - Large Plant Footprint: Great Plains Region": [
        "LandBasedWind",
    ],
    "Fixed-bottom Offshore Wind": [
        "OffshoreWind",
    ],
    "Solar PV w/ Single Axis Tracking": [
        "UtilityPV",
    ],
    "Battery Storage: 4 hours": [
        "Battery_",
        "Utility-Scale Battery Storage",
    ],
}


def extract_table_from_page(pdf_path, page_number):
    """
    Extract location variation table from a specific page.

    Args:
        pdf_path: Path to PDF file
        page_number: Page number (1-indexed)

    Returns:
        DataFrame with city, state, and location_variation columns
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]  # Convert to 0-indexed

        # Extract tables
        tables = page.extract_tables()

        if not tables:
            print(f"  Warning: No tables found on page {page_number}")
            return pd.DataFrame()

        # Use the first table
        table = tables[0]

        # Table structure: State | City | Base Project Cost | Location Variation | Delta Cost | Total Cost
        # We need columns: State (0), City (1), Location Variation (3)
        city_data = []

        for row in table[1:]:  # Skip header row
            if len(row) < 4:
                continue

            state = row[0]
            city = row[1]
            # Location Variation multiplier is in column 3
            location_var_str = row[3]

            location_var = None
            if location_var_str and location_var_str.strip():
                # Parse the multiplier value
                val = location_var_str.strip().replace(",", "")
                try:
                    location_var = float(val)
                except ValueError:
                    pass  # Keep as None if can't parse

            # Basic validation - ensure city and state look valid
            if city and state:
                if not any(char.isdigit() for char in city[:3]) and not any(
                    char.isdigit() for char in state
                ):
                    city_data.append(
                        {
                            "city": city.strip(),
                            "state": state.strip(),
                            "location_variation": location_var,
                        }
                    )

        result_df = pd.DataFrame(city_data)
        return result_df


def extract_all_tables(pdf_path, tables_config):
    """
    Extract all technology location variation tables from PDF.

    Args:
        pdf_path: Path to PDF file
        tables_config: Dict mapping page number to technology name

    Returns:
        DataFrame with technology, city, state, location_variation columns
    """
    print("Extracting location variation tables from PDF...")

    all_data = []

    for page_num, technology in tables_config.items():
        print(f"\n  Page {page_num}: {technology}")

        table_df = extract_table_from_page(pdf_path, page_num)

        if table_df.empty:
            print(f"    Warning: No data extracted")
            continue

        # Map EIA technology to PowerGenome technologies
        if technology in atb_tech_map:
            # Create a list of mapped technologies for each row
            table_df["technology"] = [atb_tech_map[technology]] * len(table_df)
            # Explode to create one row per mapped technology
            table_df = table_df.explode("technology")
            print(
                f"    Extracted {len(table_df) // len(atb_tech_map[technology])} cities"
            )
            print(
                f"    Expanded to {len(table_df)} rows ({len(atb_tech_map[technology])} PowerGenome technologies)"
            )
        else:
            # Keep original EIA technology name
            table_df["technology"] = technology
            print(f"    Extracted {len(table_df)} cities")
            print(f"    Keeping original EIA technology name (not in atb_tech_map)")

        # Reorder columns
        table_df = table_df[["technology", "city", "state", "location_variation"]]

        print(f"    Sample: {table_df.head(3).to_dict('records')}")

        all_data.append(table_df)

    # Combine all tables
    combined_df = pd.concat(all_data, ignore_index=True)

    print(f"\n  Total rows extracted: {len(combined_df)}")
    print(f"  Unique technologies: {combined_df['technology'].nunique()}")
    print(f"  Unique cities: {combined_df['city'].nunique()}")

    return combined_df


def merge_with_region_mapping(location_var_df, region_city_map_path):
    """
    Merge location variation data with ReEDS region-to-city mapping.

    Args:
        location_var_df: DataFrame with technology, city, state, location_variation
        region_city_map_path: Path to CSV with region, city, state mapping

    Returns:
        DataFrame with region, technology, location_variation columns
    """
    print("\nMerging with ReEDS region mapping...")

    # Load region-to-city mapping
    region_map = pd.read_csv(region_city_map_path)
    print(f"  Loaded {len(region_map)} region mappings")

    # Merge on city and state
    merged = region_map.merge(location_var_df, on=["city", "state"], how="left")

    print(f"  Merged result: {len(merged)} rows")

    # Select final columns
    result = merged[
        [
            "region",
            "technology",
            "city",
            "state",
            "location_variation",
            "match_method",
            "distance_km",
        ]
    ]
    result = result.rename(columns={"location_variation": "value"})

    # Check for missing values
    missing_count = result["value"].isna().sum()
    if missing_count > 0:
        print(f"\n  Warning: {missing_count} rows missing location variation data")
        print(f"  Missing combinations:")
        missing = result[result["value"].isna()][
            ["region", "technology", "city", "state"]
        ].drop_duplicates()
        print(f"    {missing.head(10)}")

    return result


def create_pivot_table(df):
    """
    Create a pivot table with regions as rows and technologies as columns.

    Args:
        df: DataFrame with region, technology, location_variation columns

    Returns:
        DataFrame pivoted with regions × technologies
    """
    pivot = df.pivot_table(
        values="value",
        index="region",
        columns="technology",
        aggfunc="first",  # Use first value if duplicates
    )

    return pivot


def main():
    """Main execution function."""
    print("=" * 70)
    print("EIA Capital Cost Location Variation Extraction")
    print("=" * 70)

    # Extract all tables
    location_var_df = extract_all_tables(PDF_PATH, TABLES)

    # Save raw extracted data
    raw_output_path = OUTPUT_PATH.parent / "eia_location_variation_raw.csv"
    location_var_df.to_csv(raw_output_path, index=False)
    print(f"\n✓ Saved raw location variation data to {raw_output_path}")

    # Merge with region mapping
    region_tech_df = merge_with_region_mapping(location_var_df, REGION_CITY_MAP_PATH)

    # Save detailed output
    region_tech_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved region-technology location variation to {OUTPUT_PATH}")
    print(f"  Total rows: {len(region_tech_df)}")
    print(f"  Unique regions: {region_tech_df['region'].nunique()}")
    print(f"  Unique technologies: {region_tech_df['technology'].nunique()}")

    # Create and save pivot table
    pivot_output_path = (
        OUTPUT_PATH.parent / "reeds_region_technology_location_variation_pivot.csv"
    )
    pivot_df = create_pivot_table(region_tech_df)
    pivot_df.to_csv(pivot_output_path)
    print(f"\n✓ Saved pivot table (regions × technologies) to {pivot_output_path}")
    print(f"  Shape: {pivot_df.shape[0]} regions × {pivot_df.shape[1]} technologies")

    # Summary statistics
    print("\nSummary Statistics:")
    print(
        f"  Location Variation Range: {region_tech_df['value'].min():.2f} to {region_tech_df['value'].max():.2f}"
    )
    print(f"  Mean: {region_tech_df['value'].mean():.2f}")
    print(f"  Median: {region_tech_df['value'].median():.2f}")

    # Show sample of pivot table
    print("\nSample of pivot table (first 10 regions, first 5 technologies):")
    print(pivot_df.iloc[:10, :5])

    print("\n" + "=" * 70)
    print("✓ Extraction complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
