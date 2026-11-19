# Existing Renewable Resource Groups

Resource group files for existing renewable generators, used by PowerGenome via `RESOURCE_GROUPS`. These describe existing hydro, onshore wind, offshore wind, and solar resources aggregated to ReEDS balancing areas (BA).

## Method

Generation profiles for existing wind and solar resources were generated with NREL's **reV** framework and the **SAM** (System Advisor Model) engine, using the Wind Integration National Dataset Toolkit (WTK) for wind and the National Solar Radiation Database (NSRDB) for solar. For each existing facility the nearest NREL weather site was used, and hourly weather data were retrieved for weather years 2007–2013. The site-specific SAM technology configurations used were:

- **Existing solar** — PVWatts v8, single-axis tracking, standard modules, DC/AC ratio 1.34, 2 MW per site, inverter efficiency 96%, total system losses ≈16%.
- **Existing onshore wind** — Vestas V82-1.65 MW turbine (rotor diameter 82 m, hub height 80 m), representative of the older existing onshore fleet, modeled as a 32-turbine, 52.8 MW farm.
- **Existing offshore wind** — NREL Reference 12 MW turbine (rotor diameter 214 m, hub height 137 m) modeled as a 32-turbine, 384 MW farm.

**Hydro (conventional and run-of-river)** was not simulated with reV: hourly profiles are based on monthly generation within each balancing area, with resources categorized by ReEDS hydro technology (see `scripts/transform_reeds_generators.py`), interpolated to hourly and smoothed with a 1-week window.

Monthly capacity factors are derived from PUDL `out_eia__monthly_generators.parquet`. Only generators with `prime_mover_code = 'HY'` are used (pumped-storage `'PS'` and co-located thermal generators are excluded, since they report negative or non-hydro net generation that would corrupt the hydro profile). The hours-in-month divisor is computed as the true calendar-month length, so December (and every month) gets a valid non-zero capacity factor.

Because the profiles are built from the PUDL `nightly` parquet, the exact numbers depend on the snapshot fetched at build time; the 2026.09.24 profiles were built from the nightly available on 2026-09-24.

All resulting hourly profiles are aggregated to ReEDS balancing areas. Existing hydro/onshore wind/offshore wind/solar metadata (capacity, plant identifiers, ReEDS BA assignment) is carried in the `existing_<tech>_reeds_ba_metadata.csv` files.

## Rebuild hydro for z90

From the repository root, run:

```bash
uv run python scripts/build_hydro_profiles.py --zone-set z90
```

The default (`--zone-set z134`) still writes to `existing_resource_groups/`.
The z90 build filters the plant map before aggregating monthly PUDL generation
and capacity. It writes both profile Parquets and their PowerGenome resource-group
JSON descriptors and metadata CSVs to the separate, locally generated
`existing_resource_groups_z90/` directory. Point `RESOURCE_GROUPS` there for z90
hydro, not at the z134 hydro descriptors.

Each z90 descriptor references its own metadata CSV (`region,capacity_mw,id`).
`id` and `region` both match the profile's `site_id`; `capacity_mw` is the sum
of transformed ReEDS generator capacity for that hydro technology and zone,
not historical PUDL capacity. Only sites with a profile are included. Missing
PUDL months or years are logged, not filled; inspect coverage before selecting
weather years for a model. The z90 build does not add files to the published
z134 resource group.

## Files

**Resource metadata (per technology)** — `existing_<tech>_reeds_ba_metadata.csv`: existing renewable plant/resource metadata by ReEDS balancing area (e.g. capacity, plant identifiers), renamed to PowerGenome conventions (`region`, `capacity_mw`).

**Hourly generation profiles (per technology)** — `existing_<tech>_reeds_ba_profiles_20250513_tidy.parquet` and `existing_osw_profiles_tidy.parquet`: hourly generation profiles for existing renewables by ReEDS balancing area, generated as described in *Method* above.

**Hydro profiles** — `hydro_conventional_2007_2013.parquet` and `hydro_run_of_river_2007_2013.parquet`: hourly conventional and run-of-river hydro generation profiles by ReEDS balancing area for weather years 2007–2013.

**Regional flags** — `regional_<tech>_True.json`: per-technology flags controlling regional (balancing-area-level) aggregation of the corresponding existing resource groups.

## Notes

- All resources are keyed by ReEDS balancing area; join to PowerGenome model regions through the balancing-area hierarchy.
- The wind and solar weather years span 2007–2013; profile vintage is 2025-05-13.
