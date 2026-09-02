# PowerGenome-data

Utilities and reference data for preparing PowerGenome inputs. Top-level Python scripts download and transform ReEDS/PUDL/EIA sources into the CSVs and parquet files used by PowerGenome.

- Documentation: [https://gschivley.github.io/PowerGenome-data/](https://gschivley.github.io/PowerGenome-data/)
- Repo: [https://github.com/gschivley/PowerGenome-data](https://github.com/gschivley/PowerGenome-data)

## Working with the docs

- Preview locally: `uv run mkdocs serve`
- Build once: `uv run mkdocs build`
- GitHub Actions publishes the docs to GitHub Pages on pushes to `main` (`.github/workflows/docs.yml`).

See the docs site for a script-by-script map of which files are generated and the upstream data sources they pull from.

## Publishing to Zenodo

Data is published to Zenodo as one deposit per collection via `publish_zenodo.py` (sandbox by default, `--publish --production` for a real release):

- **PowerGenome Input Data** — core input tables from `data/`
- **PowerGenome Renewable Resource Profiles** — hourly new-build renewable profiles from `resource_profiles/`
- **PowerGenome Existing Renewable Resource Groups** — existing renewable resource group files from `existing_resource_groups/`

Each collection keeps its own manifest (`<folder>/manifest.json`) with its own `data_version`, so
collections version independently and deposits are only created once a collection has files. See the docs site for details.
