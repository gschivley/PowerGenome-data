---
name: zenodo-upload
description: >
  Upload and publish files from the current project to a Zenodo repository.
  Use this skill whenever the user wants to deposit, publish, archive, or share
  research outputs on Zenodo, get a DOI for their data or code, or automate
  Zenodo submissions from VSCode. Trigger on phrases like "upload to Zenodo",
  "publish my dataset", "get a DOI for this", "deposit to Zenodo", or
  "archive this on Zenodo".
compatibility:
  tools: [bash, file_create, str_replace]
  requires: python3 with requests, or curl
---

# Zenodo Upload

This project publishes its data to Zenodo with `publish_zenodo.py` (repo root),
which creates **one Zenodo deposit per data collection** and keeps each record
in sync with the collection's manifest across releases. Use this skill to
prepare and run a release, or to fall back to the generic helper for ad-hoc
uploads.

## Collections and deposits

| Manifest | Source folder | Deposit title | Contents |
| --- | --- | --- | --- |
| `data/manifest.json` (core) | `data/` | PowerGenome Input Data | Core input tables (generators, load, fuels, costs, transmission, ...) |
| `resource_profiles/manifest.json` | `resource_profiles/` | PowerGenome Renewable Resource Profiles | Hourly generation profiles for new-build renewable resources |
| `existing_resource_groups/manifest.json` | `existing_resource_groups/` | PowerGenome Existing Renewable Resource Groups | Resource group files for existing renewables |

Each collection has its own `manifest.json` with its own `data_version`, so
collections version independently. A collection without a local manifest falls
back to a top-level `sections` object in the core manifest (legacy layout).
Collections with no files are skipped, so a deposit is only created once a
collection actually has data.

## Defaults

- Default to the Zenodo sandbox unless the user explicitly opts into production.
- Never hardcode tokens or write them into files.
- Prefer `publish_zenodo.py`; invoke it from bash.
- Stop before publish unless the user explicitly confirms publish.
- Reuse release state and metadata from the workspace `.zenodo.json`.

## Ask First

Before running the release workflow, collect the minimum required choices from the user:

1. Target environment: sandbox or production.
2. Which collection(s) to release: `core`, `profiles`, `existing_resource_groups`, or all.
3. Publish behavior: confirm before publish or leave as draft.
4. Token source: `ZENODO_SANDBOX_API_KEY` (sandbox) / `ZENODO_API_KEY` (production)
   environment variables are preferred; a local `.env` file is acceptable if it
   is already ignored and not committed.

Recommended defaults if the user does not specify:

- Sandbox.
- All collections with files.
- Require explicit publish confirmation.
- Use the collection manifests as-is (no manual file lists).

## Safety Rules

- Use `https://sandbox.zenodo.org/api` by default.
- Only use `https://zenodo.org/api` when the user explicitly requests production
  or sets `USE_PRODUCTION=true`.
- Treat publish as irreversible in production.
- Refuse to proceed if the required token is missing.
- The script refuses to run when any release file differs from git HEAD; commit
  first or pass `--allow-dirty` (not recommended).
- The script verifies manifest md5s against the files on disk; run
  `update_data_manifest.py` before releasing if they mismatch.
- Zenodo limits: at most 100 files and 50 GB per record (checked by the script).

## Workflow

### 1. Update the manifest (if data changed)

Each collection's manifest records `sources`, `version`, `last_updated`, `md5`,
`license`, and `history` per file, plus the collection `data_version`. Update it
after changing data files:

```bash
uv run python update_data_manifest.py
uv run python update_data_manifest.py --data-dir resource_profiles --manifest resource_profiles/manifest.json
uv run python update_data_manifest.py --data-dir existing_resource_groups --manifest existing_resource_groups/manifest.json
```

### 2. Dry run

Show what the release would contain without calling the Zenodo API:

```bash
uv run python publish_zenodo.py --dry-run
uv run python publish_zenodo.py --dry-run --collection profiles
```

### 3. Create/update the draft

```bash
# All collections (sandbox by default); prints draft URL + pre-reserved DOI.
uv run python publish_zenodo.py

# Limit to one collection (repeatable).
uv run python publish_zenodo.py --collection profiles
```

Without `--publish` the script leaves each deposition as a draft. If a draft
already exists it is resumed (no duplicate is created), and files whose
checksum already matches the draft are skipped. Pass `--save-state` to write
`.zenodo.json` even for a draft-only run.

### 4. Publish

```bash
# Publish the draft (irreversible in production) and record release state.
uv run python publish_zenodo.py --publish

# Publish to production Zenodo.
uv run python publish_zenodo.py --publish --production
```

Re-running `--publish` after an identical release is a no-op.

## What the script does

- Derives each Zenodo `version` from the collection's own `data_version`
  (CalVer `YYYY.MM.DD`) with a sequential per-day suffix (`2026.08.14.v1`,
  `2026.08.14.v2`, ...) so multiple releases on the same date never share a
  version number. The suffix combines already-published versions of the
  concept with the release versions recorded locally, so a fast follow-up
  release never reuses a suffix even while Zenodo's search index lags.
- Builds the Zenodo description from the manifest: a change note listing files
  **added**, **updated**, and **removed** in this release, the collection's
  `README.md` rendered as HTML (the descriptive body, if present), a licensing
  paragraph (compilation is CC0; individual files retain their source license),
  and a per-file table of each element's `version`, `last_updated`, `md5`,
  `license`, and `sources`. Custom prose from
  `metadata.sections.<section>.description` in `.zenodo.json` is prepended.
- Uploads **only files that changed** since the last published release
  (compared by md5); the initial release uploads everything. Files released
  before but no longer in the manifest are removed from the draft.
- Creates a **new version** of each existing record on subsequent releases; the
  first release of a collection creates a new deposition.
- Retries transient failures (connection errors, timeouts, HTTP 408/5xx proxy
  errors) with exponential backoff on every API call, including uploads.
  Uploads reopen the file for each attempt so a dropped connection on a
  multi-GB file can be resumed. If a publish request times out after Zenodo
  already processed it, the script checks the deposition state and treats an
  already-submitted record as success.
- Refuses to run when any release file differs from git HEAD; pass
  `--allow-dirty` to override.

## CLI reference

| Flag | Purpose |
| --- | --- |
| `--manifest` | Path to the core collection manifest (default `data/manifest.json`). Other collections read `<folder>/manifest.json`. |
| `--dotenv-path` | Optional dotenv file supplying the API keys (default `.env`). |
| `--production` / `--use-production` | Use production Zenodo instead of the sandbox. |
| `--publish` | Publish the deposition (irreversible in production). Default is draft-only. |
| `--save-state` | Write `.zenodo.json` even when only creating a draft. Always written on publish. |
| `--deposition-id` | Resume a specific draft deposition id (e.g. after a run died mid-upload before state was saved). |
| `--collection` | Only release this collection; repeatable. Choices: `core`, `profiles`, `existing_resource_groups`. |
| `--allow-dirty` | Publish even when release files differ from git HEAD. |
| `--sleep-seconds` | Delay between file uploads (default 1.0). |
| `--upload-retries` | Retries for a failed file upload (default 3). |
| `--upload-retry-delay` | Base delay in seconds between upload retries (default 5.0; doubles each retry). |
| `--dry-run` | Show the release plan without calling the Zenodo API. |

## Release state (`.zenodo.json`)

On the first publish the script writes `.zenodo.json` in the repo root with
shared Zenodo metadata plus a `releases` object keyed by collection section.
Each `releases.<section>` block tracks the environment (`sandbox` or
`production`), the published `data_version`, the `release_version` (the version
string actually recorded on Zenodo, suffix included), the `deposition_id`, the
resolved `doi`, the `versions` history (every release version string published
in this environment), and the per-file `md5`s that were released. The legacy
single-deposit `zenodo_release` block is migrated into `releases.core` on load.

The metadata block carries fields shared by every deposit — `creators`,
`access_right`, and the compilation `license` (e.g. `cc-zero`) — plus
per-deposit overrides under `metadata.sections.<section>` (e.g. `title` and
`creators`, so a collection can list additional authors). `.zenodo.json`
contains no secrets and is committed to the repo.

Release state is tracked **per environment**: sandbox and production
deposition ids, DOIs, and version suffix counters are independent, so you can
publish to both sites from the same `.zenodo.json` without one release
interfering with the other.

## Requirements

- A plain Python environment with the `requests` package (e.g.
  `uv run python publish_zenodo.py` or a `.venv` with `requests` installed).
- A Zenodo personal access token with `deposit:write` and `deposit:actions`
  scopes: `ZENODO_SANDBOX_API_KEY` for the sandbox and `ZENODO_API_KEY` for
  production. Tokens may be loaded from a local `.env` file via
  `--dotenv-path`; `.env` is git-ignored and never committed.
- Optional: the `markdown` package to render collection `README.md` files into
  the Zenodo description (falls back to escaped plain text).

## Fallback: generic helper for ad-hoc uploads

For uploading arbitrary files that are not part of a collection manifest, a
generic helper lives at `scripts/zenodo_upload.py` (in this skill folder). It
uses `ZENODO_TOKEN` and supports `create-draft`, `new-version`, and `publish`
subcommands. See `references/api-reference.md` for the manual `curl` workflow
and `references/metadata-template.json` for a metadata template. Prefer
`publish_zenodo.py` for project data releases.

## Agent Execution Notes

- Prefer `publish_zenodo.py` for project data; use the generic helper only for
  ad-hoc uploads outside the manifest workflow.
- Run `--dry-run` first, then the draft run, inspect the JSON output, then ask
  the user for confirmation before running `--publish`.
- If the script exits with a manifest md5 mismatch, run
  `update_data_manifest.py` for that collection before retrying.
- If the script exits because release files differ from git HEAD, commit the
  changes first (or ask the user before using `--allow-dirty`).
- If Zenodo returns `400`, inspect the metadata first. Missing `title`,
  `upload_type`, `description`, or creator names are the common causes.
- If Zenodo returns `401` or `403`, verify the token scopes `deposit:write`
  and `deposit:actions` and that the correct environment token is set.
- If `requests` is unavailable, install it in the current environment or fall
  back to a manual `curl` workflow using `references/api-reference.md`.

## Expected Outputs

After draft creation, report:

- Deposition ID(s) and draft URL(s).
- Pre-reserved DOI(s) if present.
- Uploaded file list with verified checksums.

After publish, report:

- Final DOI(s) and record URL(s).
- Whether the operation used sandbox or production.