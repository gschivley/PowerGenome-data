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
This is the provenance + change-detection layer: it drives `publish_zenodo.py` (below), which publishes
manifest-listed files to Zenodo and can be wired into scheduled update workflows.

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
      "license": "public-domain",
      "history": []
    }
  }
}
```

- `manifest_version` — schema version of the manifest format itself; fixed at `1` unless the schema changes.
- `data_version` — the **overall dataset** version in calendar-versioning format `YYYY.MM.DD`. It
  advances to the current date whenever a data input is added or its contents change, and is otherwise
  preserved from the previous run. The manifest file itself is excluded from scanning, so it never
  triggers a `data_version` bump on its own. Use this to tag a dataset release (e.g. on Zenodo).
- Per file — `sources` is a list of `{source, source_url}` objects describing the upstream origin
  (human-maintained; a file may rely on more than one upstream, e.g. `plant_region_map.csv` uses both
  the ReEDS generator database and `county2zone.csv`). `source_url` is optional per source. `version` /
  `last_updated` are the ISO date (`YYYY-MM-DD`) the file last changed, `md5` is the content hash,
  `license` is a human-maintained identifier for the file's underlying-source license
  (`public-domain`, `cc-by-4.0`, or `cc-zero`), and `history` holds prior
  `{version, last_updated, md5}` entries for every change that has been seen.

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
  `"Unknown - document me"` so they can be filled in). If an entry still carries the
  `"Unknown - document me"` placeholder and a real seed now exists for that file, a later run upgrades the
  placeholder to the seed, so provenance backfills automatically once it has been documented.
- Hand-maintained `license` values are preserved when a file changes (only `version`/`last_updated`/`md5`
  are rewritten); files that carry no `license` are flagged with a warning so the attribution can be
  added.
- Files that are no longer present in `data/` are dropped from the manifest (with a warning).

Run the tests with:

```bash
python -m unittest discover -s tests
```

## Publishing to Zenodo

`publish_zenodo.py` publishes the manifest-listed files to Zenodo, **one deposit per data
collection**, and keeps each Zenodo record in sync with the manifest across releases. It is the
publishing counterpart to the manifest described above.

### Collections and deposits

The manifest is split into sections; each non-empty section becomes its own Zenodo deposit with its
own title, description, version history, and DOI:

| Manifest section | Source folder | Deposit title | Contents |
| --- | --- | --- | --- |
| `files` (top level, `core`) | `data/` | PowerGenome Input Data | Core input tables (generators, load, fuels, costs, transmission, ...) |
| `sections.profiles` | `resource_profiles/` | PowerGenome Renewable Resource Profiles | Hourly generation profiles for new-build renewable resources (PowerGenome `RESOURCE_GROUP_PROFILES`) |
| `sections.existing_resource_groups` | `existing_resource_groups/` | PowerGenome Existing Renewable Resource Groups | Resource group files for existing renewables (PowerGenome `RESOURCE_GROUPS`) |

Inside a deposit, file keys are prefixed by collection (`profiles/…`,
`existing_resource_groups/…`), so downloads self-organize and filenames can never collide across
collections. Sections with no files are skipped, so a deposit is only created once a collection
actually has data. The flat top-level `files` object is the core section, which keeps older
manifests valid.

Each section's files use the same entry schema as the core `files` object (`sources`, `version`,
`last_updated`, `md5`, `license`, `history`).

A collection can supply its own methodology prose for the Zenodo description via
`metadata.sections.<section>.description` in `.zenodo.json`; when present it is prepended to the
auto-generated per-file description (e.g. the Renewable Resource Profiles deposit documents how the
WTK/NSRDB weather data were turned into generation profiles and what the site-mapping files
(`CPA_ID` → `Site` → `profile_dist`) contain).

### What it does

- Reads `data/manifest.json` and derives each Zenodo `version` from its top-level `data_version`
  (CalVer `YYYY.MM.DD`) with a sequential per-day suffix (`2026.08.14.v1`, `2026.08.14.v2`, ...) so
  that multiple releases on the same date never share a version number. The suffix is computed by
  combining the already-published versions of the dataset's concept carrying the same `data_version`
  with the release versions recorded locally (see `versions` under Release state below), so a fast
  follow-up release never reuses a suffix even while Zenodo's search index is still catching up.
- Builds the Zenodo description from the manifest: a change note listing the files **added**,
  **updated**, and **removed** in this release, a licensing paragraph (the compilation is CC0 while
  individual files retain their underlying-source license), plus a per-file table of each data element's own
  `version` (the date that element was last updated), `last_updated`, `md5`, its `license`, and its `sources`.
- Uploads **only files that changed** since the last published release (compared by `md5`), so the
  initial release uploads everything and later releases upload just the new/updated files. Files that
  were released before but are no longer in the manifest are removed from the draft.
- Creates a **new version** of each existing record on subsequent releases, so Zenodo keeps the full file
  history. The first release of a collection creates a new deposition.
- Refuses to run when any release file differs from git HEAD (so the Zenodo record always corresponds to
  a committed state of the repository); pass `--allow-dirty` to override.
- Defaults to the **Zenodo sandbox**; pass `--production` (or set `USE_PRODUCTION=true`) for production.

### Requirements

- A plain Python environment with the `requests` package (e.g. `uv run python publish_zenodo.py` or a
  `.venv` with `requests` installed).
- A Zenodo personal access token with `deposit:write` and `deposit:actions` scopes, supplied via the
  `ZENODO_TOKEN` environment variable (or `ZENODO_SANDBOX_API_KEY` for sandbox). It may be loaded from a
  local `.env` file without extra arguments — `.env` is git-ignored and never committed.

### Usage

```bash
# Show what the release would contain (no API calls)
uv run python publish_zenodo.py --dry-run

# Show only one collection's release plan
uv run python publish_zenodo.py --dry-run --collection profiles

# Create/update the Zenodo draft (sandbox) and print the draft URL + DOI.
uv run python publish_zenodo.py

# Resolve just one collection's draft (repeat --collection for several)
uv run python publish_zenodo.py --collection profiles

# Publish the draft (irreversible in production) and record release state.
uv run python publish_zenodo.py --publish

# Publish to production Zenodo.
uv run python publish_zenodo.py --publish --production

# Publish anyway even though release files are uncommitted (not recommended).
uv run python publish_zenodo.py --publish --allow-dirty
```

By default every collection with files in the manifest is released. Pass `--collection`
(repeatable; choices: `core`, `profiles`, `existing_resource_groups`) to limit the run to a
single deposit or a subset.

Without `--publish` the script leaves each deposition as a draft. If a draft already exists it is resumed
(no duplicate is created), and files whose checksum already matches the draft are skipped. Re-running
`--publish` after an identical release is a no-op.

The script verifies that every file listed in the manifest is committed to git before it contacts Zenodo.
If any release file differs from HEAD it prints the offending paths and exits; commit the changes first or
pass `--allow-dirty`. This keeps each Zenodo record reproducible from the repository history.

### Release state (`.zenodo.json`)

On the first publish the script writes `.zenodo.json` in the repo root with shared Zenodo metadata plus
a `releases` object keyed by collection section (`core`, `profiles`, `existing_resource_groups`).
Each `releases.<section>` block tracks the environment (`sandbox` or `production`), the published
`data_version`, the `release_version` (the version string actually recorded on Zenodo, suffix
included), the `deposition_id`, the resolved `doi`, the `versions` history (every release version
string published in this environment, used to keep same-day suffixes unique even while Zenodo's
search index lags), and the per-file `md5`s that were released. The legacy single-deposit
`zenodo_release` block is migrated into `releases.core` on load. The metadata block carries the fields
shared by every deposit — `creators`, `access_right`, and the compilation `license` (e.g. `cc-zero`) —
plus per-deposit title overrides under `metadata.sections.<section>`; each file's source license is
recorded in `data/manifest.json` and surfaced in the description. `.zenodo.json` contains no secrets
and is committed to the repo so later runs know what changed and can create new versions against the
same records.

The release state is tracked **per environment**. Sandbox and production deposition ids, DOIs, and version
suffix counters are independent, so the script records which environment a release belongs to and ignores
stored release state from the other environment (keeping shared metadata such as creators). This lets you
publish to both sites from the same `.zenodo.json` without one release interfering with the other.
