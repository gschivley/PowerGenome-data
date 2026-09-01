# Renewable Resource Profiles

Hourly generation profiles for new-build renewable resources, used by PowerGenome via `RESOURCE_GROUP_PROFILES`. This folder holds one parquet per technology plus the site-mapping files that join profiles to Princeton capacity areas.

## Method

Profiles were generated with NREL's **reV** framework and the **SAM** (System Advisor Model) engine using wind resource data from the Wind Integration National Dataset Toolkit (WTK) and solar resource data from the National Solar Radiation Database (NSRDB).

**Site selection.** A 10 km × 10 km grid was generated over the contiguous United States and each grid point was assigned to its Princeton capacity area (CPA). For every grid point the nearest NREL WTK (wind) or NSRDB (solar) weather site was identified, and multi-year hourly weather data were retrieved for those sites for weather years 2007–2013.

**Simulation.** For each weather year and site, reV executed SAM's PVWatts v8 model (solar) or wind power model (wind) to produce an hourly capacity-factor profile per site:

- **Utility-scale solar** — PVWatts v8, single-axis tracking with bifacial modules (bifaciality 0.7), DC/AC ratio 1.34, 5 MW per site, total system losses ≈14%.
- **Onshore wind** — NREL Reference 5.5 MW turbine (rotor diameter 175 m, hub height 115 m) modeled as a 32-turbine, 176 MW farm.
- **Offshore wind** — NREL Reference 12 MW turbine (rotor diameter 214 m, hub height 137 m) modeled as a 32-turbine, 384 MW farm.

## Files

| File | Contents |
| --- | --- |
| `<tech>_rev_profiles_20240801_tidy.parquet` | Hourly capacity-factor profiles in tidy form. One row per (`site_id`, `weather_year`, `time_index`) with column `value` (capacity factor 0–1). |
| `<tech>_site_mapping_20240801.parquet` | Site mapping with one row per grid point: `CPA_ID` (the Princeton capacity area the grid point was assigned to), `Site` (the NREL WTK/NSRDB resource site used to generate the profile), and `profile_dist` (the great-circle distance between the grid point and that site, in km). |

Profile records are keyed by `site_id`, so a site map joins the hourly profiles to the Princeton capacity areas they represent. Weather years span 2007–2013.
