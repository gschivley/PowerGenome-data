# Existing Renewable Resource Groups

Resource group files for existing renewable generators, used by PowerGenome via `RESOURCE_GROUPS`. These describe existing hydro, onshore wind, offshore wind, and solar resources aggregated to ReEDS balancing areas (BA).

## Method

Generation profiles for existing wind and solar resources were generated with NREL's **reV** framework and the **SAM** (System Advisor Model) engine, using the Wind Integration National Dataset Toolkit (WTK) for wind and the National Solar Radiation Database (NSRDB) for solar. For each existing facility the nearest NREL weather site was used, and hourly weather data were retrieved for weather years 2007–2013. The site-specific SAM technology configurations used were:

- **Existing solar** — PVWatts v8, single-axis tracking, standard modules, DC/AC ratio 1.34, 2 MW per site, inverter efficiency 96%, total system losses ≈16%.
- **Existing onshore wind** — Vestas V82-1.65 MW turbine (rotor diameter 82 m, hub height 80 m), representative of the older existing onshore fleet, modeled as a 32-turbine, 52.8 MW farm.
- **Existing offshore wind** — NREL Reference 12 MW turbine (rotor diameter 214 m, hub height 137 m) modeled as a 32-turbine, 384 MW farm.

**Hydro (conventional and run-of-river)** was not simulated with reV: hourly profiles are derived from PUDL monthly generation data for each region and hydro type, interpolated to hourly and smoothed with a 1-week window.

All resulting hourly profiles are aggregated to ReEDS balancing areas. Existing hydro/onshore wind/offshore wind/solar metadata (capacity, plant identifiers, ReEDS BA assignment) is carried in the `existing_<tech>_reeds_ba_metadata.csv` files.

## Files

**Resource metadata (per technology)** — `existing_<tech>_reeds_ba_metadata.csv`: existing renewable plant/resource metadata by ReEDS balancing area (e.g. capacity, plant identifiers), renamed to PowerGenome conventions (`region`, `capacity_mw`).

**Hourly generation profiles (per technology)** — `existing_<tech>_reeds_ba_profiles_20250513_tidy.parquet` and `existing_osw_profiles_tidy.parquet`: hourly generation profiles for existing renewables by ReEDS balancing area, generated as described in *Method* above.

**Hydro profiles** — `hydro_conventional_2007_2013.parquet` and `hydro_run_of_river_2007_2013.parquet`: hourly conventional and run-of-river hydro generation profiles by ReEDS balancing area for weather years 2007–2013.

**Regional flags** — `regional_<tech>_True.json`: per-technology flags controlling regional (balancing-area-level) aggregation of the corresponding existing resource groups.

## Notes

- All resources are keyed by ReEDS balancing area; join to PowerGenome model regions through the balancing-area hierarchy.
- The wind and solar weather years span 2007–2013; profile vintage is 2025-05-13.
