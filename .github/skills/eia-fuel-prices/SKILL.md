---
name: eia-fuel-prices
description: "Download EIA Annual Energy Outlook (AEO) bulk data and update data/fuel_prices.parquet with electric power sector fuel price scenarios. Use when: fetching AEO fuel prices for a new year, adding EIA fuel price scenarios, updating fuel_prices.parquet with AEO2025 or AEO2026 data, getting coal/gas/distillate/uranium prices by ReEDS BA."
argument-hint: "AEO year(s) to fetch, e.g. 2026 or 2025 2026"
---

# EIA Fuel Prices

## What This Skill Does

Downloads EIA AEO bulk ZIP files, extracts "Energy Prices : Electric Power" series for each
census division / scenario / fuel, maps census divisions to ReEDS Balancing Areas, and appends
the results to `data/fuel_prices.parquet`, deduplicating on `(year, data_year, scenario, fuel, region)`.

## Procedure

### 1. Determine years to fetch

Ask the user which AEO year(s) they want, or read from their request. The bulk file URL is:
`https://www.eia.gov/opendata/bulk/AEO{year}.zip`

### 2. Run the script

From the repo root:

```bash
uv run .github/skills/eia-fuel-prices/scripts/fetch_eia_fuel_prices.py --years <year(s)>
```

Full options:
```
--years         AEO year(s), e.g. --years 2026 or --years 2025 2026   [required]
--output        Path to output parquet file     [default: data/fuel_prices.parquet]
--hierarchy     ReEDS hierarchy CSV              [default: cache_data/hierarchy.csv]
--cache-dir     Directory for cached ZIP files   [default: cache_data/eia_bulk]
```

Downloaded ZIPs are cached in `cache_data/eia_bulk/` so re-runs are fast.

### 3. Verify the output

```bash
# Count new rows by data_year and scenario
uv run python -c "
import pandas as pd
df = pd.read_parquet('data/fuel_prices.parquet')
print(df.groupby(['data_year','scenario','fuel'])['region'].count().reset_index())
"
```

Expected: ~130 rows per (data_year, scenario, fuel) combination — one per ReEDS BA.

### 4. Confirm idempotency

Re-running the same command should produce no changes (duplicates are dropped, new rows take
precedence over old via `keep='last'`).

---

## Key Mappings

### Scenario name normalization

EIA scenario names are normalized to snake_case. Special overrides (both years):

| EIA Name | Output `scenario` |
|---|---|
| Alternative Electricity | `no_111d` |
| Counterfactual Baseline | `baseline` |
| Reference | `reference` |
| Low Price | `low_price` |
| High Price | `high_price` |
| Low Economic Growth | `low_economic_growth` |
| High Economic Growth | `high_economic_growth` |

To add a new override, edit `SCENARIO_OVERRIDES` in the script.

### Fuel mapping

| EIA series component | `fuel` column |
|---|---|
| Steam Coal | `coal` |
| Natural Gas | `naturalgas` |
| Distillate Fuel Oil | `distillate` |
| Uranium | `uranium` |

Unrecognized fuels are logged as INFO and skipped. To add one, edit `FUEL_MAP` in the script.

### Census division → `cendiv` (hierarchy.csv)

| EIA name | `cendiv` value |
|---|---|
| New England | `New_England` |
| Middle Atlantic | `Middle_Atlantic` |
| East North Central | `East_North_Central` |
| West North Central | `West_North_Central` |
| South Atlantic | `South_Atlantic` |
| East South Central | `East_South_Central` |
| West South Central | `West_South_Central` |
| Mountain | `Mountain` |
| Pacific | `Pacific` |

National/U.S. aggregates are logged as INFO and skipped.

### Dollar year convention

`dollar_year = aeo_year - 1`  (AEO2025 → 2024, AEO2026 → 2025)

---

## Troubleshooting

**"Unmapped EIA fuel names" in output** — EIA has added a new fuel type. Add it to `FUEL_MAP` if
needed for your use case.

**"Unmapped census divisions" in output** — EIA has renamed or added a division. Add it to
`CENDIV_MAP`.

**"No .txt file found in ZIP"** — EIA changed the file format. Inspect the ZIP:
```python
import zipfile; print(zipfile.ZipFile("cache_data/eia_bulk/AEO2026.zip").namelist())
```

**HTTP error on download** — Delete the (empty) cached ZIP and retry:
```bash
rm cache_data/eia_bulk/AEO<year>.zip
uv run .github/skills/eia-fuel-prices/scripts/fetch_eia_fuel_prices.py --years <year>
```

**Scenario names don't match existing fuel_prices.parquet** — Compare unique scenario values:
```python
import pandas as pd
df = pd.read_parquet('data/fuel_prices.parquet')
print(df.groupby('data_year')['scenario'].unique())
```
Then add entries to `SCENARIO_OVERRIDES` in the script as needed.
