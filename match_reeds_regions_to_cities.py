#!/usr/bin/env python3
"""
Match ReEDS model regions to EIA capital cost cities.

This script:
1. Extracts cities from EIA capital cost PDF (Table 1-1, page 162)
2. Downloads ReEDS US_PCA shapefile with region geometries
3. Geocodes city locations to coordinates
4. Matches each region to a city using spatial analysis:
   - Cities within the region polygon
   - Nearest city (preferring same-state cities by geographic proximity)
5. Exports region-to-city mapping to CSV
"""

import json
import time
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pdfplumber
import requests
from geopy.geocoders import Nominatim
from shapely.geometry import Point
from tqdm import tqdm

# URLs for ReEDS shapefiles
SHAPEFILE_BASE_URL = (
    "https://github.com/NREL/ReEDS-2.0/raw/main/inputs/shapefiles/US_PCA"
)
SHAPEFILE_FILES = ["US_PCA.shp", "US_PCA.shx", "US_PCA.dbf", "US_PCA.prj"]

# PDF path
PDF_PATH = Path(
    "cache/capital_cost_AEO2025.pdf"
)  # https://www.eia.gov/analysis/studies/powerplants/capitalcost/pdf/capital_cost_AEO2025.pdf

# Output path
OUTPUT_PATH = Path("cache/reeds_region_to_eia_city.csv")

# Cache file for geocoded cities
GEOCODE_CACHE_FILE = Path("cache_data/geocoded_cities.json")


def extract_cities_from_pdf(pdf_path, page_number=162):
    """
    Extract cities and states from Table 1-1 in the EIA capital cost PDF.

    Args:
        pdf_path: Path to the PDF file
        page_number: Page number containing Table 1-1 (1-indexed)

    Returns:
        DataFrame with columns: city, state
    """
    print(f"Extracting cities from PDF page {page_number}...")

    cities_data = []

    with pdfplumber.open(pdf_path) as pdf:
        # Convert to 0-indexed
        page = pdf.pages[page_number - 1]

        # Extract tables from the page
        tables = page.extract_tables()

        if not tables:
            print("  No tables found on the page. Trying text extraction...")
            text = page.extract_text()
            print(f"  Page text preview:\n{text[:500]}")
            return pd.DataFrame(columns=["city", "state"])

        print(f"  Found {len(tables)} table(s) on page {page_number}")

        # Process the first table (Table 1-1)
        # Expected format: [State, City, Base Project Cost, Location Variation, Delta Cost, Total]
        table = tables[0]

        # Skip header row, extract state (column 0) and city (column 1)
        for row in table[1:]:
            if row and len(row) >= 2:
                state = row[0]
                city = row[1]

                # Basic validation - skip empty rows and verify valid data
                if state and city and state.strip() and city.strip():
                    # Skip if state contains numbers (data rows should have text states)
                    if not any(char.isdigit() for char in state):
                        cities_data.append(
                            {"city": city.strip(), "state": state.strip()}
                        )

    if not cities_data:
        print("  Warning: No cities extracted. Manual entry may be needed.")
        # Provide manual fallback with common EIA reference cities
        print("  Using common EIA reference cities as fallback...")
        cities_data = [
            {"city": "Atlanta", "state": "GA"},
            {"city": "Boston", "state": "MA"},
            {"city": "Chicago", "state": "IL"},
            {"city": "Dallas", "state": "TX"},
            {"city": "Denver", "state": "CO"},
            {"city": "Detroit", "state": "MI"},
            {"city": "Houston", "state": "TX"},
            {"city": "Los Angeles", "state": "CA"},
            {"city": "Miami", "state": "FL"},
            {"city": "New York", "state": "NY"},
            {"city": "Philadelphia", "state": "PA"},
            {"city": "Phoenix", "state": "AZ"},
            {"city": "San Francisco", "state": "CA"},
            {"city": "Seattle", "state": "WA"},
            {"city": "Washington", "state": "DC"},
        ]

    df = pd.DataFrame(cities_data)
    df = df.drop_duplicates().reset_index(drop=True)

    print(f"\n  Extracted {len(df)} unique cities")
    print(f"  Sample cities: {df.head(10).to_dict('records')}")

    return df


def download_shapefile(output_dir="cache_data/shapefiles"):
    """
    Download ReEDS US_PCA shapefile components from GitHub.

    Args:
        output_dir: Directory to save shapefile components

    Returns:
        Path to the .shp file
    """
    print("\nDownloading ReEDS US_PCA shapefile...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for filename in SHAPEFILE_FILES:
        file_path = output_path / filename

        if file_path.exists():
            print(f"  {filename} already exists, skipping download")
            continue

        url = f"{SHAPEFILE_BASE_URL}/{filename}"
        print(f"  Downloading {filename}...")

        response = requests.get(url)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            f.write(response.content)

        print(f"    Saved to {file_path}")

    shp_path = output_path / "US_PCA.shp"
    print(f"\n  Shapefile ready at: {shp_path}")

    return shp_path


def load_reeds_regions(shapefile_path):
    """
    Load ReEDS regions from shapefile.

    Args:
        shapefile_path: Path to the .shp file

    Returns:
        GeoDataFrame with region geometries and 'rb' column
    """
    print("\nLoading ReEDS regions from shapefile...")

    gdf = gpd.read_file(shapefile_path)

    print(f"  Loaded {len(gdf)} regions")
    print(f"  Columns: {list(gdf.columns)}")
    print(f"  CRS: {gdf.crs}")

    # Check for 'rb' column
    if "rb" not in gdf.columns:
        print("  Warning: 'rb' column not found. Available columns:")
        print(f"  {list(gdf.columns)}")
        # Try to find alternative region identifier
        for col in ["region", "ba", "zone", "id", "name"]:
            if col in gdf.columns:
                print(f"  Using '{col}' as region identifier instead")
                gdf = gdf.rename(columns={col: "rb"})
                break

    # Ensure we're using a projected CRS for distance calculations
    if gdf.crs.is_geographic:
        print("  Converting to projected CRS (EPSG:5070 - NAD83 / Conus Albers)...")
        gdf = gdf.to_crs(epsg=5070)

    return gdf


def geocode_cities(cities_df, delay=1.5):
    """
    Geocode cities to coordinates using Nominatim with caching.

    Args:
        cities_df: DataFrame with 'city' and 'state' columns
        delay: Delay between requests in seconds (Nominatim requires 1 req/sec)

    Returns:
        GeoDataFrame with city points
    """
    print("\nGeocoding cities to coordinates...")

    # Load cache if it exists
    cache = {}
    if GEOCODE_CACHE_FILE.exists():
        with open(GEOCODE_CACHE_FILE, "r") as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached locations")
    else:
        # Ensure cache directory exists
        GEOCODE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    geolocator = Nominatim(user_agent="reeds_region_matcher", timeout=10)

    geocoded_data = []
    cache_updated = False

    for idx, row in tqdm(list(cities_df.iterrows()), desc="Geocoding cities"):
        city = row["city"]
        state = row["state"]
        cache_key = f"{city}, {state}"

        # Check cache first
        if cache_key in cache:
            geocoded_data.append(
                {
                    "city": city,
                    "state": state,
                    "latitude": cache[cache_key]["latitude"],
                    "longitude": cache[cache_key]["longitude"],
                }
            )
            continue

        # Create query
        query = f"{city}, {state}, USA"

        try:
            print(f"  Geocoding {query}...", end="")
            location = geolocator.geocode(query)

            if location:
                geocoded_data.append(
                    {
                        "city": city,
                        "state": state,
                        "latitude": location.latitude,
                        "longitude": location.longitude,
                        "geometry": Point(location.longitude, location.latitude),
                    }
                )

                # Add to cache
                cache[cache_key] = {
                    "latitude": location.latitude,
                    "longitude": location.longitude,
                }
                cache_updated = True

                print(f" ✓ ({location.latitude:.4f}, {location.longitude:.4f})")
            else:
                print(f" ✗ Not found")

        except Exception as e:
            print(f" ✗ Error: {e}")

        # Respect rate limit
        if idx < len(cities_df) - 1:
            time.sleep(delay)

    # Save updated cache
    if cache_updated:
        with open(GEOCODE_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"\n  Saved {len(cache)} locations to cache")

    if not geocoded_data:
        raise ValueError(
            "No cities could be geocoded. Check city names and internet connection."
        )

    gdf = gpd.GeoDataFrame(geocoded_data, crs="EPSG:4326")

    print(f"\n  Successfully geocoded {len(gdf)} / {len(cities_df)} cities")

    return gdf


def match_regions_to_cities(regions_gdf, cities_gdf):
    """
    Match each ReEDS region to an EIA city.

    Strategy:
    1. Check if any city falls within the region polygon
    2. If no match, find nearest city (preferring same-state cities)
    3. For same-state preference, filter to same state first if possible

    Args:
        regions_gdf: GeoDataFrame of ReEDS regions (projected CRS)
        cities_gdf: GeoDataFrame of EIA cities (geographic CRS)

    Returns:
        DataFrame with columns: region, city, state, match_method, distance_km
    """
    print("\nMatching regions to cities...")

    # Convert cities to same CRS as regions for distance calculations
    cities_projected = cities_gdf.to_crs(regions_gdf.crs)

    # Get state for each region (spatial join with state boundaries would be ideal,
    # but we'll infer from nearest city or use region centroids)
    matches = []

    for idx, region_row in regions_gdf.iterrows():
        region_id = region_row["rb"]
        region_geom = region_row["geometry"]

        print(f"\n  Processing region {region_id}...", end="")

        # Method 1: Check if any city is within the region
        cities_within = cities_projected[cities_projected.within(region_geom)]

        if len(cities_within) > 0:
            # Use the first city within the region
            matched_city = cities_within.iloc[0]
            match_method = "within"
            distance_km = 0
            print(
                f" ✓ City within region: {matched_city['city']}, {matched_city['state']}"
            )
        else:
            # Method 2: Find nearest city
            # Calculate centroid of region
            centroid = region_geom.centroid

            # Calculate distances to all cities
            cities_projected["distance"] = cities_projected.distance(centroid)
            cities_sorted = cities_projected.sort_values("distance")

            # Try to find nearest city in same state
            # First, we need to determine the region's state
            # Use the nearest city's state as the region's state
            nearest_city_overall = cities_sorted.iloc[0]
            region_state = nearest_city_overall["state"]

            # Filter to same-state cities
            same_state_cities = cities_sorted[cities_sorted["state"] == region_state]

            if len(same_state_cities) > 0:
                matched_city = same_state_cities.iloc[0]
                match_method = "nearest_same_state"
                distance_km = matched_city["distance"] / 1000  # Convert meters to km
                print(
                    f" → Nearest in {region_state}: {matched_city['city']} ({distance_km:.1f} km)"
                )
            else:
                # No same-state city available, use overall nearest
                matched_city = nearest_city_overall
                match_method = "nearest"
                distance_km = matched_city["distance"] / 1000
                print(
                    f" → Nearest overall: {matched_city['city']}, {matched_city['state']} ({distance_km:.1f} km)"
                )

        matches.append(
            {
                "region": region_id,
                "city": matched_city["city"],
                "state": matched_city["state"],
                "match_method": match_method,
                "distance_km": round(distance_km, 2),
            }
        )

    df = pd.DataFrame(matches)

    print(f"\n\nMatching Summary:")
    print(f"  Total regions matched: {len(df)}")
    print(f"\n  Match method breakdown:")
    print(df["match_method"].value_counts().to_string())
    print(f"\n  States represented:")
    print(df["state"].value_counts().to_string())

    return df


def main():
    """Main execution function."""
    print("=" * 70)
    print("ReEDS Region to EIA City Matching")
    print("=" * 70)

    # Step 1: Extract cities from PDF
    cities_df = extract_cities_from_pdf(PDF_PATH)

    # Step 2: Download and load ReEDS shapefile
    shapefile_path = download_shapefile()
    regions_gdf = load_reeds_regions(shapefile_path)

    # Step 3: Geocode cities
    cities_gdf = geocode_cities(cities_df)

    # Step 4: Match regions to cities
    mapping_df = match_regions_to_cities(regions_gdf, cities_gdf)

    # Step 5: Export mapping
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'=' * 70}")
    print(f"✓ Exported mapping to {OUTPUT_PATH}")
    print(f"  Total regions: {len(mapping_df)}")
    print(f"  Total unique cities: {mapping_df['city'].nunique()}")
    print(f"\n  Sample mappings:")
    print(mapping_df.head(10).to_string(index=False))
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
