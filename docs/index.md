# PowerGenome Data Inputs

This short guide explains how the top-level Python scripts fetch, extract, and transform data to create the files that live in `data/` (plus a few closely related outputs). Many CSVs in `data/` ship pre-populated; the scripts here only regenerate a subset and sometimes cache external downloads. When a file is not listed below, it is not produced by these scripts.

## Quickstart

- Preview docs locally: `uv run mkdocs serve`
- Build docs: `uv run mkdocs build`
- Most scripts rely on large external datasets (PUDL, ReEDS, EIA PDFs) and will download/cache them as needed under `cache_data/` or the working directory.

## Script-to-output map

| Script | Main outputs | External inputs & sources | Notes |
| --- | --- | --- | --- |
| `transform_reeds_generators.py` | `data/reeds_generators_transformed.csv`, `data/plant_region_map.csv` | ReEDS generator database CSV from NREL GitHub; PUDL yearly generators parquet (`out_eia__yearly_generators.parquet`); ReEDS `county2zone.csv` | Cleans generator tech names (handles run-of-river), filters retired units, computes capacities, builds plant→region map. |
| `build_hydro_profiles.py` | `resource_profiles/hydro_run_of_river_2007_2013.parquet`, `resource_profiles/hydro_conventional_2007_2013.parquet` | `data/plant_region_map.csv`, `data/reeds_generators_transformed.csv`; PUDL monthly generators parquet (`s3://pudl.catalyst.coop/nightly/out_eia__monthly_generators.parquet`) | Derives monthly capacity factors per region/type, interpolates to hourly, smooths with a 1-week window. |
| `transform_reeds_load.py` | `reeds_load_transformed.parquet` | ReEDS hourly load HDF5 (`inputs/load/EER_IRAlow_load_hourly.h5` from ReEDS GitHub, LFS-backed) | Downloads HDF5 (with LFS handling), filters weather years 2007–2013 and model years 2020–2050, melts to tidy hourly load with time_index. |
| `build_new_pg_dg_inputs.py` | `distributed_capacity.parquet`, `distributed_profiles.parquet` | ReEDS `dgen_model_inputs/stscen2023_mid_case/distpvcap_stscen2023_mid_case.csv`; ReEDS `county2zone.csv`; optional distpv HDF5 (`variability/multi_year/distpv-reference_ba.h5` from ReEDS GitHub raw) | Capacity is aggregated county→BA and converted from DC to AC (ILR=1.1). Profiles download can be skipped via `--skip-profiles` to avoid the large HDF5. |
| `extract_location_variation.py` | `data/eia_location_variation_raw.csv`, `data/regional_cost_multipliers.csv`, `data/reeds_region_technology_location_variation_pivot.csv` | EIA capital cost PDF (`cache_data/capital_cost_AEO2025.pdf`); region→city map (`cache_data/reeds_region_to_eia_city.csv` from `match_reeds_regions_to_cities.py`) | Reads tables 1-1–1-19, maps EIA technologies to PowerGenome techs, merges with regions, and pivots to region×technology multipliers. |
| `match_reeds_regions_to_cities.py` | `cache_data/reeds_region_to_eia_city.csv` (and `cache_data/geocoded_cities.json` cache) | Same EIA PDF (table 1-1); ReEDS US_PCA shapefile from NREL GitHub | Geocodes PDF cities, downloads shapefiles, then matches each ReEDS region to a city (within-region or nearest). Feeds location-variation extraction. |
| `merge_transmission_capacity.py` | `data/transmission_capacity_reeds.csv` | ReEDS AC and non-AC transmission CSVs from NREL GitHub | Downloads and caches the raw files, averages forward/reverse AC MW, sums AC+non-AC per region pair, keeps notes. |
| `build_emission_policies.py` | `emission_policies_wecc.csv` (path hard-coded) | ESR Excel workbook (`ESR_Inputs_ReEDS_WECC.xlsx`); `settings/model_definition.yml` for model years and aggregations | Pop-weighted ESR values by region/year; enforces CES ≥ RPS when both apply. |

## Running the builders

All scripts are regular Python entry points. Examples (run from repo root):

```bash
uv run python transform_reeds_generators.py
uv run python build_hydro_profiles.py
uv run python transform_reeds_load.py
uv run python build_new_pg_dg_inputs.py --skip-profiles  # avoid large HDF5
uv run python extract_location_variation.py
uv run python merge_transmission_capacity.py
uv run python build_emission_policies.py
```

`match_reeds_regions_to_cities.py` is usually run once before `extract_location_variation.py` to build the region→city map and geocode cache. Many scripts expect the referenced PDFs/CSVs to already exist under `cache_data/` or will download them on first run.

## Data files not regenerated here

Files such as `cpi_data.csv`, `fuel_prices.csv`, `technology_heat_rates_nrelatb.csv`, and other static CSVs in `data/` are provided as-is and are not built by the scripts listed above.
