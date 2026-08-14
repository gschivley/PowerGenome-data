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
| `fetch_gdp_ipd_data.py` | `data/dollar_year_adjustment.csv` | FRED (series `A191RD3A086NBEA`); BEA NIPA API (fallback, requires `BEA_API_KEY` env var) | Fetches annual GDP Implicit Price Deflator (GDP-IPD, index 2017=100) from 1980 onward. GDP-IPD is used by AEO and ATB for dollar-year adjustments. Intended to supersede `cpi_data.csv`; some configs/consumers may still reference the older file until migration is complete. Run weekly via GitHub Actions. |
| `build_operational_constraints_reeds.py` | `data/operational_constraints_reeds.csv` | ReEDS `pcm_defaults.json` (`inputs/plant_characteristics/` from NREL GitHub) | Builds PowerGenome operational constraints from ReEDS PCM defaults: drops all CSP technologies, maps ReEDS tech names to ATB/EIA new-build names (see `TECH_MAP` in the script), applies battery/pumped-storage efficiency and duration overrides, and appends curated existing-technology rows preserved from the historical file. |

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
uv run python fetch_gdp_ipd_data.py
uv run python build_operational_constraints_reeds.py  # optional --pcm-path <local pcm_defaults.json> to avoid downloading
```

`match_reeds_regions_to_cities.py` is usually run once before `extract_location_variation.py` to build the region→city map and geocode cache. Many scripts expect the referenced PDFs/CSVs to already exist under `cache_data/` or will download them on first run.

## Data files not regenerated here

Files such as `fuel_prices.csv`, `technology_heat_rates_nrelatb.csv`, and other static CSVs in `data/` are provided as-is and are not built by the scripts listed above.

## Versioned data-source manifest (`data/manifest.json`)

`data/manifest.json` records, for every file in `data/`, where the data came from, when it was last
updated, and an MD5 content hash, along with a versioned history that advances whenever a file changes.
This is the provenance + change-detection layer that will eventually drive scheduled update workflows
and automatic Zenodo publishing.

### Read it

```json
{
  "manifest_version": 1,
  "data_version": "2026.08.11",
  "generated_at_utc": "2026-08-11T19:55:29.157504+00:00",
  "files": {
    "fuel_prices.parquet": {
      "sources": [
        {
          "source": "EIA Annual Energy Outlook ...",
          "source_url": "https://www.eia.gov/opendata/bulk/AEO2026.zip"
        }
      ],
      "last_updated": "2026-08-11",
      "version": "2026-08-11",
      "md5": "a5393175e8eabda1134e849ceeb6f5e1",
      "history": []
    }
  }
}
```

- `manifest_version` — schema version of the manifest format itself; fixed at `1` unless the schema changes.
- `data_version` — the **overall dataset** version in calendar-versioning format `YYYY.MM.DD`. It
  advances to the current date whenever any file is added or its contents change, and is otherwise
  preserved from the previous run. Use this to tag a dataset release (e.g. on Zenodo).
- Per file — `sources` is a list of `{source, source_url}` objects describing the upstream origin
  (human-maintained; a file may rely on more than one upstream, e.g. `plant_region_map.csv` uses both
  the ReEDS generator database and `county2zone.csv`). `source_url` is optional per source. `version` /
  `last_updated` are the ISO date (`YYYY-MM-DD`) the file last changed, `md5` is the content hash, and
  `history` holds prior `{version, last_updated, md5}` entries for every change that has been seen.

### Update it

Run `update_data_manifest.py` from the repo root after any data file changes:

```bash
uv run python update_data_manifest.py            # rescan data/ and bump versions on change
uv run python update_data_manifest.py --dry-run  # preview changes without writing
uv run python update_data_manifest.py --date 2026-08-11  # override the change date (reproducible runs)
```

Behavior:

- Unchanged files keep their existing version, date, and source.
- Changed or newly added files get `version`/`last_updated` set to the current date, the previous entry
  moves onto `history`, and `data_version` advances (CalVer `YYYY.MM.DD`).
- Hand-edited `sources` entries (including `source_url`) are preserved across runs; the script only fills
  them in for files it has never seen (from a built-in seed map — genuinely unknown origins are marked
  `"Unknown - document me"` so they can be filled in).
- Files that are no longer present in `data/` are dropped from the manifest (with a warning).

Run the tests with:

```bash
python -m unittest discover -s tests
```
