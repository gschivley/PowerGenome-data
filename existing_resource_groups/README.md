# Existing Renewable Resource Groups

Resource group files for existing renewable generators, used by PowerGenome via `RESOURCE_GROUPS`. These describe existing hydro, onshore wind, offshore wind, and solar resources aggregated to ReEDS balancing areas (BA).

## Files

**Resource metadata (per technology)** — `existing_<tech>_reeds_ba_metadata.csv`: existing renewable plant/resource metadata by ReEDS balancing area (e.g. capacity, plant identifiers), renamed to PowerGenome conventions (`region`, `capacity_mw`).

**Hourly generation profiles (per technology)** — `existing_<tech>_reeds_ba_profiles_20250513_tidy.parquet` and `existing_osw_profiles.{csv,tidy.parquet}`: hourly generation profiles for existing renewables by ReEDS balancing area, generated with NREL reV/SAM from WTK/NSRDB weather data (see `resource_profiles/README.md` for the simulation method) and aggregated to balancing areas.

**Hydro profiles** — `hydro_conventional_2007_2013.parquet` and `hydro_run_of_river_2007_2013.parquet`: hourly conventional and run-of-river hydro generation profiles by ReEDS balancing area for weather years 2007–2013, derived from PUDL monthly generation data interpolated to hourly.

**Regional flags** — `regional_<tech>_True.json`: per-technology flags controlling regional (balancing-area-level) aggregation of the corresponding existing resource groups.

## Notes

- All resources are keyed by ReEDS balancing area; join to PowerGenome model regions through the balancing-area hierarchy.
- The offshore wind profiles are available in both a wide CSV and a tidy parquet representation of the same data.
