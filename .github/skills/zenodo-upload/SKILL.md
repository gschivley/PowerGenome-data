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

Use this skill to package files from the current workspace and create, update, or publish a Zenodo deposition through the Zenodo REST API.

## Defaults

- Default to Zenodo sandbox unless the user explicitly opts into production.
- Never hardcode tokens or write them into files.
- Prefer the helper script at `scripts/zenodo_upload.py` and invoke it from bash.
- Stop before publish unless the user explicitly confirms publish.
- Reuse metadata from a workspace `.zenodo.json` when present.

## Ask First

Before running the upload workflow, collect the minimum required choices from the user:

1. Target environment: sandbox or production.
2. Workflow mode: new deposit, new version, or draft only.
3. Files to include: explicit file paths or shell globs.
4. Metadata: title, upload type, description, creators.
5. Publish behavior: confirm before publish or leave as draft.
6. Token source: `ZENODO_TOKEN` environment variable is preferred; a local `.env` file is acceptable if it is already ignored and not committed.

Recommended defaults if the user does not specify:

- Sandbox.
- Support `.zenodo.json` in the project root.
- Support creating new versions.
- Require explicit publish confirmation.
- Use explicit file paths or shell globs provided at run time rather than a hardcoded include pattern.

## Safety Rules

- Use `https://sandbox.zenodo.org/api` by default.
- Only use `https://zenodo.org/api` when the user explicitly requests production or sets `USE_PRODUCTION=true`.
- Treat publish as irreversible in production.
- Refuse to proceed if `ZENODO_TOKEN` is missing.
- Warn if the total upload exceeds 100 files or 50 GB.
- Validate uploaded file checksums against the local MD5.
- Add a short delay between multiple uploads.

## Workflow

### Mode A: New Deposit

1. Confirm the target files and metadata.
2. If `.zenodo.json` exists in the project root, use it as a starting point and layer user-supplied metadata on top.
3. Run the helper to create a draft deposition, upload files, and apply metadata.
4. Show the returned deposition ID, draft URL, and pre-reserved DOI.
5. Ask the user whether to publish.
6. If confirmed, run the publish command and return the final DOI and record URL.

### Mode B: New Version

1. Ask for the existing deposition ID or DOI.
2. Create a new draft version.
3. Upload replacement or additional files.
4. Update metadata, especially `version` and related identifiers.
5. Ask for publish confirmation.
6. Publish only after confirmation.

### Mode C: Draft Only

1. Create the deposition draft.
2. Upload files.
3. Apply metadata.
4. Return the draft link `https://zenodo.org/deposit/<id>` or the sandbox equivalent.

## Helper Script

The helper script lives at `scripts/zenodo_upload.py`.

### Expected environment variables

- `ZENODO_TOKEN`: required personal access token.
- `USE_PRODUCTION=true`: optional opt-in for production.
- `ZENODO_BASE`: optional explicit API base URL override.

### Supported commands

Create a draft deposition with uploaded files and metadata:

```bash
python3 zenodo-upload/scripts/zenodo_upload.py create-draft \
  --file data/archive.zip \
  --metadata-file zenodo-upload/references/metadata-template.json
```

Create a new version draft from an existing record:

```bash
python3 zenodo-upload/scripts/zenodo_upload.py new-version \
  --record 1234567 \
  --file dist/release.tar.gz \
  --metadata-file metadata.json
```

Publish a draft only after explicit user confirmation:

```bash
python3 zenodo-upload/scripts/zenodo_upload.py publish \
  --deposition-id 1234567 \
  --confirm-publish
```

Persist merged metadata back into the project root `.zenodo.json` when the user wants the repo metadata updated:

```bash
python3 zenodo-upload/scripts/zenodo_upload.py create-draft \
  --file data/results.csv \
  --metadata-json '{"title": "Example Dataset", "upload_type": "dataset", "description": "<p>Example</p>", "creators": [{"name": "Doe, Jane"}]}' \
  --write-project-zenodo-json
```

## Agent Execution Notes

- Prefer passing file paths directly and let the shell expand globs.
- If `.zenodo.json` exists, the helper loads it automatically unless `--skip-project-zenodo-json` is set.
- For publish flows, run `create-draft` or `new-version` first, inspect the JSON output, then ask the user for confirmation before running `publish`.
- If Zenodo returns `400`, inspect the metadata first. Missing `title`, `upload_type`, `description`, or creator names are the common causes.
- If Zenodo returns `401` or `403`, verify token scopes `deposit:write` and `deposit:actions`.
- If `requests` is unavailable, install it in the current environment or fall back to a manual `curl` workflow using `references/api-reference.md`.

## Expected Outputs

After draft creation, report:

- Deposition ID.
- Draft URL.
- Pre-reserved DOI if present.
- Uploaded file list with verified checksums.

After publish, report:

- Final DOI.
- Record URL.
- Whether the operation used sandbox or production.