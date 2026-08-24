#!/usr/bin/env python3
"""Publish PowerGenome-data data files listed in data/manifest.json to Zenodo.

The release is driven entirely by data/manifest.json:
  * The Zenodo release version is the manifest's top-level "data_version"
    (calendar version, e.g. "2026.08.14").
  * The Zenodo description is generated from the manifest: one
    file/version/last-updated/sources table per data element, using each
    file's own "version" key (the date that data element was last updated).
  * Only files whose md5 changed since the last release are uploaded. The
    first release (no prior state) uploads all manifest files. Files
    previously released but no longer in the manifest are removed from the
    draft so Zenodo mirrors the manifest.

Release state (last published data_version, deposition id, per-file md5s,
publish status) is stored in .zenodo.json in the project root. That file
contains no secrets and is safe to commit. If a draft was created but not
yet published, the draft is resumed on the next run instead of creating a
duplicate.

Defaults to the Zenodo sandbox. Production requires --production or
USE_PRODUCTION=true. Publishing requires --publish; without it a draft is
created/updated and the draft URL is printed.

The API token is read from the ZENODO_TOKEN environment variable, which may
be loaded from a local dotenv file (default .env) via --dotenv-path.
"""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

try:
    import requests
except ImportError:  # pragma: no cover - depends on environment
    sys.exit("This script requires the 'requests' package: pip install requests")

SANDBOX_BASE = "https://sandbox.zenodo.org/api"
PRODUCTION_BASE = "https://zenodo.org/api"
MAX_FILES = 100
MAX_TOTAL_BYTES = 50 * 1024 * 1024 * 1024
PROJECT_ROOT = None
STATE_PATH = None


def log(msg: str) -> None:
    print(f"[zenodo] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Environment / helpers
# ---------------------------------------------------------------------------

def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_base(use_production: bool) -> str:
    override = os.getenv("ZENODO_BASE")
    if override:
        return override.rstrip("/")
    return PRODUCTION_BASE if use_production else SANDBOX_BASE


def load_token() -> str:
    token = os.getenv("ZENODO_TOKEN") or os.getenv("ZENODO_SANDBOX_API_KEY")
    if not token:
        sys.exit(
            "A token is required. Set ZENODO_TOKEN (or ZENODO_SANDBOX_API_KEY for "
            "sandbox) in the environment or a local .env file (see --dotenv-path)."
        )
    return token


def md5_for_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_user_name() -> str:
    try:
        return subprocess.check_output(
            ["git", "config", "user.name"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Manifest + state
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "data_version" not in manifest or not isinstance(manifest.get("files"), dict):
        sys.exit(f"Manifest {manifest_path} must have 'data_version' and a 'files' object.")
    return manifest


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def default_creators() -> list[dict]:
    name = git_user_name()
    if not name:
        sys.exit(
            "No creators configured. Add a 'creators' list to .zenodo.json "
            "(see .zenodo.json metadata format) or set git user.name."
        )
    parts = name.split()
    last = parts[-1]
    first = " ".join(parts[:-1])
    return [{"name": f"{last}, {first}"}]


# ---------------------------------------------------------------------------
# Description generation from the manifest
# ---------------------------------------------------------------------------

def describe_file(filename: str, info: dict) -> str:
    esc = html.escape
    sources = info.get("sources") or []
    source_items = []
    for source in sources:
        text = esc(source.get("source") or "")
        url = source.get("source_url")
        if url:
            source_items.append(f'<li><strong>Source:</strong> <a href="{esc(url)}">{esc(url)}</a> &mdash; {text}</li>')
        else:
            source_items.append(f"<li><strong>Source:</strong> {text}</li>")
    sources_html = "<ul>" + "".join(source_items) + "</ul>" if source_items else "<em>No source documented.</em>"
    return (
        f"<h3><code>{esc(filename)}</code></h3>"
        f"<table>"
        f"<tr><th>Data element version</th><td>{esc(info.get('version') or 'Unknown')}</td></tr>"
        f"<tr><th>Last updated</th><td>{esc(info.get('last_updated') or 'Unknown')}</td></tr>"
        f"<tr><th>md5</th><td><code>{esc(info.get('md5') or 'Unknown')}</code></td></tr>"
        f"</table>"
        f"{sources_html}"
    )


def build_description(manifest: dict, updated_files: list[str], initial: bool) -> str:
    data_version = manifest["data_version"]
    files = manifest["files"]
    esc = html.escape

    if initial:
        change_note = (
            f"<p>This is the <strong>initial release</strong> of this dataset. "
            f"All {len(files)} files are new in this version.</p>"
        )
    else:
        names = "".join(f"<li><code>{esc(name)}</code></li>" for name in updated_files)
        change_note = (
            f"<p><strong>Files updated in this release:</strong></p>"
            f"<ul>{names if names else '<li>None</li>'}</ul>"
        )

    file_sections = "\n".join(
        describe_file(filename, info)
        for filename, info in sorted(files.items())
    )

    return (
        "<p>PowerGenome input data assembled from public sources. This release "
        f"corresponds to <code>data/manifest.json</code> data version "
        f"<code>{esc(data_version)}</code>.</p>"
        f"{change_note}"
        "<p>Each data element's own version key records when that element was "
        "last updated; the per-file sources below document where it came from.</p>"
        f"{file_sections}"
    )


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def make_session(token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    return session


def raise_for_status(response: requests.Response, action: str) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text.strip()
        raise SystemExit(f"{action} failed with HTTP {response.status_code}: {detail}") from exc


def create_deposition(session: requests.Session, base_url: str) -> dict:
    response = session.post(
        f"{base_url}/deposit/depositions", json={},
        headers={"Content-Type": "application/json"}, timeout=120,
    )
    raise_for_status(response, "Creating deposition")
    return response.json()


def get_deposition(session: requests.Session, base_url: str, deposition_id: str) -> dict:
    response = session.get(f"{base_url}/deposit/depositions/{deposition_id}", timeout=120)
    raise_for_status(response, f"Fetching deposition {deposition_id}")
    return response.json()


def create_new_version(session: requests.Session, base_url: str, record: str) -> dict:
    response = session.post(
        f"{base_url}/deposit/depositions/{record}/actions/newversion", timeout=120,
    )
    raise_for_status(response, "Creating new version draft")
    payload = response.json()
    draft_url = payload.get("links", {}).get("latest_draft")
    if draft_url:
        draft_response = session.get(draft_url, timeout=120)
        raise_for_status(draft_response, "Fetching new draft deposition")
        return draft_response.json()
    return payload


def upload_file(session: requests.Session, bucket_url: str, path: Path) -> dict:
    checksum = md5_for_file(path)
    with path.open("rb") as handle:
        response = session.put(f"{bucket_url}/{path.name}", data=handle, timeout=3600)
    raise_for_status(response, f"Uploading {path.name}")

    payload = response.json()
    remote_checksum = payload.get("checksum", "")
    if remote_checksum.startswith("md5:"):
        remote_checksum = remote_checksum.split(":", 1)[1]
    if remote_checksum and remote_checksum != checksum:
        raise SystemExit(
            f"Checksum mismatch for {path.name}: local {checksum}, remote {payload.get('checksum')}"
        )
    return {
        "filename": path.name,
        "local_md5": checksum,
        "remote_checksum": payload.get("checksum"),
        "response": payload,
    }


def delete_file(session: requests.Session, file_url: str, filename: str) -> None:
    response = session.delete(file_url, timeout=120)
    raise_for_status(response, f"Deleting {filename} from draft")


def set_metadata(session: requests.Session, base_url: str, deposition_id: str, metadata: dict) -> dict:
    response = session.put(
        f"{base_url}/deposit/depositions/{deposition_id}",
        json={"metadata": metadata}, headers={"Content-Type": "application/json"}, timeout=120,
    )
    raise_for_status(response, "Updating metadata")
    return response.json()


def publish_deposition(session: requests.Session, base_url: str, deposition_id: str) -> dict:
    response = session.post(
        f"{base_url}/deposit/depositions/{deposition_id}/actions/publish", timeout=120,
    )
    raise_for_status(response, "Publishing deposition")
    return response.json()


def pre_reserved_doi(payload: dict) -> str | None:
    return payload.get("metadata", {}).get("prereserve_doi", {}).get("doi")


def draft_page_url(base_url: str, deposition_id: str) -> str:
    host = "https://zenodo.org" if base_url == PRODUCTION_BASE else "https://sandbox.zenodo.org"
    return f"{host}/deposit/{deposition_id}"


def existing_draft_files(session: requests.Session, deposition: dict) -> dict[str, tuple[str, str]]:
    """Map of filename -> (file id, md5) for files already present in a draft.

    The current Zenodo deposit-files API removes a file by its internal ``id``
    (a UUID present in each file object); deleting by filename returns 500.
    A filename fallback is kept for API variants without an explicit ``id``.
    """
    response = session.get(deposition["links"]["files"], timeout=120)
    raise_for_status(response, "Listing draft files")
    mapping: dict[str, tuple[str, str]] = {}
    for item in response.json():
        name = item.get("key") or item.get("filename")
        file_id = item.get("id") or name
        checksum = item.get("checksum", "")
        if checksum.startswith("md5:"):
            checksum = checksum.split(":", 1)[1]
        mapping[name] = (file_id, checksum)
    return mapping


# ---------------------------------------------------------------------------
# Release logic
# ---------------------------------------------------------------------------

def validate_constraints(files_by_name: dict[str, Path]) -> None:
    total_bytes = sum(path.stat().st_size for path in files_by_name.values())
    if len(files_by_name) > MAX_FILES:
        sys.exit(f"Zenodo allows at most {MAX_FILES} files per record; manifest has {len(files_by_name)}.")
    if total_bytes > MAX_TOTAL_BYTES:
        sys.exit(f"Zenodo allows at most 50 GB per record; total is {total_bytes / 1e9:.1f} GB.")


def run_release(args) -> None:
    global PROJECT_ROOT, STATE_PATH
    PROJECT_ROOT = Path.cwd()
    STATE_PATH = PROJECT_ROOT / ".zenodo.json"

    load_dotenv(Path(args.dotenv_path).expanduser())
    token = load_token()
    session = make_session(token)
    base_url = resolve_base(args.use_production or truthy_env("USE_PRODUCTION"))

    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    manifest = load_manifest(manifest_path)
    data_version = manifest["data_version"]
    manifest_files = manifest["files"]
    data_dir = manifest_path.parent

    state = load_state()
    released = state.get("zenodo_release", {})
    previously_published = bool(released.get("published"))
    # Only diff against md5s of a previous *published* release. An unpublished
    # draft is resumed by re-uploading everything (idempotent bucket overwrite).
    released_files = released.get("files", {}) if previously_published else {}
    deposition_id = released.get("deposition_id")

    # Local file existence + manifest md5 verification.
    local_paths: dict[str, Path] = {}
    local_md5s: dict[str, str] = {}
    for filename in manifest_files:
        local_path = data_dir / filename
        if not local_path.exists():
            sys.exit(f"Manifest lists {filename} but {local_path} does not exist on disk.")
        actual = md5_for_file(local_path)
        local_md5s[filename] = actual
        expected = manifest_files[filename].get("md5")
        if expected and actual != expected:
            sys.exit(
                f"Manifest md5 mismatch for {filename}: manifest says {expected}, "
                f"file has {actual}. Run update_data_manifest.py before releasing."
            )
        local_paths[filename] = local_path
    validate_constraints(local_paths)

    # Determine which files were updated in this release.
    changed = [
        name
        for name, info in manifest_files.items()
        if released_files.get(name) != info.get("md5")
    ]
    removed = [name for name in released_files if name not in manifest_files]
    initial = not released_files

    log(f"data version: {data_version}")
    log(f"updated files ({len(changed)}): {', '.join(changed) if changed else 'none'}")
    if removed:
        log(f"files dropped from manifest: {', '.join(removed)}")
    if not changed and not removed:
        log("no file changes detected since last release; the description/version may still be updated")

    # Nothing to do if this exact version was already published unchanged.
    if previously_published and data_version == released.get("data_version") and not changed and not removed:
        log(f"version {data_version} is already published on Zenodo (deposition {deposition_id}); nothing to do.")
        return

    # Resolve the target draft.
    if deposition_id and not previously_published:
        log(f"resuming unpublished draft {deposition_id}")
        deposition = get_deposition(session, base_url, str(deposition_id))
    elif deposition_id:
        log(f"creating new version from deposition {deposition_id}")
        deposition = create_new_version(session, base_url, str(deposition_id))
    else:
        log("creating a new deposition")
        deposition = create_deposition(session, base_url)

    deposition_id = str(deposition["id"])
    bucket_url = deposition["links"]["bucket"]

    # Remove files no longer part of the manifest (covers released-then-dropped
    # files and any leftovers in a resumed draft / previous version).
    present = existing_draft_files(session, deposition)
    for filename, (file_id, _checksum) in present.items():
        if filename in local_paths:
            continue
        file_url = deposition["links"]["files"] + "/" + file_id
        log(f"removing {filename} (no longer in manifest)")
        delete_file(session, file_url, filename)

    # Upload files whose local checksum differs from what the draft already
    # holds (skips files unchanged since a prior draft copy / new-version).
    uploads = []
    for index, (filename, local_path) in enumerate(local_paths.items()):
        local_md5 = local_md5s[filename]
        _, draft_checksum = present.get(filename, (None, None))
        if draft_checksum == local_md5:
            log(f"already up to date in draft, skipping: {filename}")
            continue
        log(f"uploading {filename} ({local_path.stat().st_size / 1e6:.1f} MB)")
        result = upload_file(session, bucket_url, local_path)
        result["filename"] = filename
        uploads.append(result)
        if index < len(local_paths) - 1 and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    # Build + update metadata.
    description = build_description(manifest, changed, initial)
    project_metadata = state.get("metadata", {}) if isinstance(state, dict) else {}
    creators = project_metadata.get("creators") or default_creators()
    metadata = dict(project_metadata)
    metadata.update(
        {
            "title": project_metadata.get("title") or "PowerGenome Input Data",
            "upload_type": project_metadata.get("upload_type") or "dataset",
            "access_right": project_metadata.get("access_right") or "open",
            "creators": creators,
            "version": data_version,
            "description": description,
        }
    )

    updated = set_metadata(session, base_url, deposition_id, metadata)

    # Publish (only with --publish) or leave as draft.
    published = False
    publish_payload = {}
    if args.publish:
        log(f"publishing deposition {deposition_id}")
        publish_payload = publish_deposition(session, base_url, deposition_id)
        published = True
    else:
        log(f"draft only (no --publish); draft: {draft_page_url(base_url, deposition_id)}")

    summary = {
        "mode": "publish" if published else "draft",
        "environment": "production" if base_url == PRODUCTION_BASE else "sandbox",
        "data_version": data_version,
        "deposition_id": deposition_id,
        "draft_url": draft_page_url(base_url, deposition_id),
        "pre_reserved_doi": pre_reserved_doi(updated) or pre_reserved_doi(deposition),
        "doi": publish_payload.get("doi") if published else None,
        "record_url": (publish_payload.get("links", {}).get("record_html") or publish_payload.get("links", {}).get("html")) if published else None,
        "updated_files": [u["filename"] for u in uploads],
        "removed_files": removed,
        "check_uploads": uploads,
    }
    print(json.dumps(summary, indent=2))

    # Persist release state (after a successful publish, or on resume so the
    # draft can be continued; changed-file diff only applies when published).
    save_state = args.publish or args.save_state
    if save_state:
        new_release = {
            "data_version": data_version,
            "deposition_id": deposition_id,
            "published": published,
            "doi": publish_payload.get("doi") if published else None,
            "files": {name: info["md5"] for name, info in manifest_files.items()},
        }
        merged = {"metadata": metadata, "zenodo_release": new_release}
        STATE_PATH.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log(f"state written to {STATE_PATH}")

    if not published:
        log("Not published. Run again with --publish to publish this draft, or download from the draft URL above.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="publish_zenodo.py",
        description="Publish/update data/manifest.json files to Zenodo (sandbox by default).",
    )
    parser.add_argument("--manifest", default="data/manifest.json", help="Path to the release manifest (default: data/manifest.json).")
    parser.add_argument("--dotenv-path", default=".env", help="Optional dotenv file supplying ZENODO_TOKEN.")
    parser.add_argument("--use-production", "--production", dest="use_production", action="store_true", help="Use production Zenodo instead of the sandbox.")
    parser.add_argument("--publish", action="store_true", help="Publish the deposition (irreversible in production). Default is draft-only.")
    parser.add_argument("--save-state", action="store_true", help="Write .zenodo.json even when only creating a draft. Always written on publish.")
    parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Delay between file uploads (default 1.0).")
    parser.add_argument("--dry-run", action="store_true", help="Show the release plan without calling the Zenodo API.")
    return parser


def dry_run(manifest_path: Path) -> None:
    manifest = load_manifest(manifest_path)
    data_dir = manifest_path.parent
    files = manifest["files"]
    print(f"data version: {manifest['data_version']}")
    print(f"manifest files: {len(files)}")
    for name, info in sorted(files.items()):
        missing = not (data_dir / name).exists()
        print(f"  {'MISSING ' if missing else 'ok      '}{name}  (version {info.get('version')})")
    total = sum((data_dir / n).stat().st_size for n in files if (data_dir / n).exists())
    print(f"total bytes: {total / 1e6:.1f} MB")


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        manifest_path = Path(args.manifest).expanduser()
        dry_run(manifest_path)
        return
    run_release(args)


if __name__ == "__main__":
    main()
