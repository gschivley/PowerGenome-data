#!/usr/bin/env python3
"""Publish PowerGenome-data data files listed in data/manifest.json to Zenodo.

Each data collection (section) in the manifest is published to its own Zenodo
deposit:

  * ``core``                     -- files under ``data/`` ("PowerGenome Input Data")
  * ``profiles``                 -- files under ``resource_profiles/`` (hourly
                                    generation profiles for new-build renewable
                                    resources; PowerGenome ``RESOURCE_GROUP_PROFILES``)
  * ``existing_resource_groups`` -- files under ``existing_resource_groups/``
                                    (resource group files for existing renewables;
                                    PowerGenome ``RESOURCE_GROUPS``)

The flat top-level ``files`` object in the core manifest is the core section;
each other collection keeps its own ``manifest.json`` at the top of its data
folder (``resource_profiles/manifest.json``,
``existing_resource_groups/manifest.json``). For backward compatibility, a
collection without a local manifest falls back to a top-level ``sections``
object in the core manifest. Sections with no files are skipped, so a deposit
is only created once a collection actually has data.

Each release is driven by its collection's manifest:
  * The Zenodo release version is that manifest's "data_version" with a
    sequential per-day suffix, e.g. "2026.08.24.v1", "2026.08.24.v2",
    so multiple releases on the same date get distinct version numbers.
    Every collection versions independently, so updating profiles does not
    bump the core deposit's version (and vice versa).
  * The Zenodo description is generated from the manifest: a change note
    listing files added, updated, and removed in this release, plus one
    file/version/last-updated/sources table per data element, using each
    file's own "version" key (the date that data element was last updated).
  * Only files whose md5 changed since the last release are uploaded. The
    first release (no prior state) uploads all manifest files. Files
    previously released but no longer in the manifest are removed from the
    draft so Zenodo mirrors the manifest.

Release state (per-section data_version, deposition id, per-file md5s, publish
status) is stored in .zenodo.json in the project root, under a "releases"
object keyed by section. The legacy single-deposit "zenodo_release" object is
migrated into ``releases.core`` on load. .zenodo.json contains no secrets and
is safe to commit. If a draft was created but not yet published, the draft is
resumed on the next run instead of creating a duplicate.

Defaults to the Zenodo sandbox. Production requires --production or
USE_PRODUCTION=true. Publishing requires --publish; without it a draft is
created/updated and the draft URL is printed.

The API token is read from the environment: ZENODO_SANDBOX_API_KEY for the
sandbox and ZENODO_API_KEY for production. Tokens may be loaded from a local
dotenv file (default .env) via --dotenv-path.

By default all collections with files are released on every run; pass one or
more --collection <name> flags (core, profiles, existing_resource_groups) to
limit the run to specific deposits.
"""

import argparse
import hashlib
import html
import json
import os
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover - depends on environment
    sys.exit("This script requires the 'requests' package: pip install requests")

try:
    import markdown
except ImportError:  # pragma: no cover - depends on environment
    markdown = None

SANDBOX_BASE = "https://sandbox.zenodo.org/api"
PRODUCTION_BASE = "https://zenodo.org/api"
MAX_FILES = 100
MAX_TOTAL_BYTES = 50 * 1024 * 1024 * 1024
PROJECT_ROOT = None
STATE_PATH = None

# One Zenodo deposit per data collection (manifest section). ``data_dir`` is
# relative to the repo root; ``key_prefix`` prefixes file keys inside the
# deposit so filenames can never collide across collections and downloads
# self-organize by collection.
COLLECTIONS = {
    "core": {
        "data_dir": "data",
        "key_prefix": "",
        "title": "PowerGenome Input Data",
    },
    "profiles": {
        "data_dir": "resource_profiles",
        "key_prefix": "profiles/",
        "title": "PowerGenome Renewable Resource Profiles",
    },
    "existing_resource_groups": {
        "data_dir": "existing_resource_groups",
        "key_prefix": "existing_resource_groups/",
        "title": "PowerGenome Existing Renewable Resource Groups",
    },
}
CORE_SECTION = "core"


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


def load_token(use_production: bool) -> str:
    if use_production:
        token = os.getenv("ZENODO_API_KEY")
        if not token:
            sys.exit(
                "A production token is required. Set ZENODO_API_KEY in the "
                "environment or a local .env file (see --dotenv-path)."
            )
        return token
    token = os.getenv("ZENODO_SANDBOX_API_KEY")
    if not token:
        sys.exit(
            "A sandbox token is required. Set ZENODO_SANDBOX_API_KEY in the "
            "environment or a local .env file (see --dotenv-path)."
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


def _validate_manifest(manifest: dict, manifest_path: Path) -> None:
    if "data_version" not in manifest or not isinstance(manifest.get("files"), dict):
        sys.exit(
            f"Manifest {manifest_path} must have 'data_version' and a 'files' object."
        )
    sections = manifest.get("sections")
    if sections is not None:
        if not isinstance(sections, dict):
            sys.exit(f"Manifest {manifest_path}: 'sections' must be an object.")
        for name, section in sections.items():
            if name not in COLLECTIONS:
                sys.exit(
                    f"Manifest {manifest_path}: unknown section {name!r}; "
                    f"known sections: {', '.join(sorted(COLLECTIONS))}."
                )
            if not isinstance(section, dict) or not isinstance(
                section.get("files"), dict
            ):
                sys.exit(
                    f"Manifest {manifest_path}: section {name!r} must be an object "
                    "with a 'files' object."
                )


def load_manifest(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(manifest, manifest_path)
    return manifest


def collection_manifest_path(
    project_root: Path, section: str, flag_manifest: Path
) -> Path:
    """Manifest path for one collection.

    The core collection uses the ``--manifest`` flag (default
    ``data/manifest.json``). Every other collection has its own
    ``manifest.json`` at the top of its data folder.
    """
    if section == CORE_SECTION:
        return flag_manifest
    return project_root / COLLECTIONS[section]["data_dir"] / "manifest.json"


def load_manifests(args, project_root: Path) -> dict[str, dict]:
    """Per-collection manifests: ``{section: {"data_version", "files"}}``.

    Reads ``<folder>/manifest.json`` for each collection. As a fallback for
    the older single-file layout, a collection without a local manifest is
    populated from ``sections.<name>`` in the core manifest.
    """
    flag_manifest = Path(args.manifest).expanduser()
    if not flag_manifest.is_absolute():
        flag_manifest = project_root / flag_manifest
    core = load_manifest(flag_manifest)

    manifests = {CORE_SECTION: core}
    for name in COLLECTIONS:
        if name == CORE_SECTION:
            continue
        local = collection_manifest_path(project_root, name, flag_manifest)
        if local.exists():
            manifests[name] = load_manifest(local)
        else:
            section = (core.get("sections") or {}).get(name) or {}
            manifests[name] = {
                "data_version": core["data_version"],
                "files": section.get("files") or {},
            }
    return manifests


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def section_release(state: dict, section: str) -> dict:
    """Release state for one collection, migrating the legacy single-deposit
    ``zenodo_release`` object into ``releases.core``."""
    releases = state.get("releases")
    if isinstance(releases, dict) and isinstance(releases.get(section), dict):
        return releases[section]
    if section == CORE_SECTION:
        legacy = state.get("zenodo_release")
        if isinstance(legacy, dict) and legacy:
            return legacy
    return {}


def section_metadata(state: dict, section: str) -> dict:
    """Zenodo metadata for one collection: shared top-level defaults with
    per-section overrides from ``metadata.sections.<section>``."""
    project_metadata = state.get("metadata", {}) if isinstance(state, dict) else {}
    if not isinstance(project_metadata, dict):
        project_metadata = {}
    overrides = {}
    sections = project_metadata.get("sections")
    if isinstance(sections, dict) and isinstance(sections.get(section), dict):
        overrides = sections[section]
    merged = {k: v for k, v in project_metadata.items() if k != "sections"}
    merged.update(overrides)
    # Title/description/version are per-deposit and regenerated every release;
    # a stray value in shared metadata (legacy state, hand edits) must not leak
    # into another deposit. An explicit per-section override still wins.
    for per_release_field in ("title", "description", "version"):
        if per_release_field not in overrides:
            merged.pop(per_release_field, None)
    return merged


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

LICENSE_LABELS = {
    "cc-zero": "CC0 (public domain dedication)",
    "cc-by-4.0": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
    "public-domain": "Public domain (U.S. government)",
}


def describe_file(filename: str, info: dict) -> str:
    esc = html.escape
    sources = info.get("sources") or []
    source_items = []
    for source in sources:
        text = esc(source.get("source") or "")
        url = source.get("source_url")
        if url:
            source_items.append(
                f'<li><strong>Source:</strong> <a href="{esc(url)}">{esc(url)}</a> &mdash; {text}</li>'
            )
        else:
            source_items.append(f"<li><strong>Source:</strong> {text}</li>")
    sources_html = (
        "<ul>" + "".join(source_items) + "</ul>"
        if source_items
        else "<em>No source documented.</em>"
    )
    license_key = info.get("license") or ""
    license_label = (
        LICENSE_LABELS.get(license_key) or esc(license_key) or "Not specified"
    )
    return (
        f"<h3><code>{esc(filename)}</code></h3>"
        f"<table>"
        f"<tr><th>Data element version</th><td>{esc(info.get('version') or 'Unknown')}</td></tr>"
        f"<tr><th>Last updated</th><td>{esc(info.get('last_updated') or 'Unknown')}</td></tr>"
        f"<tr><th>md5</th><td><code>{esc(info.get('md5') or 'Unknown')}</code></td></tr>"
        f"<tr><th>License</th><td>{esc(license_label)}</td></tr>"
        f"</table>"
        f"{sources_html}"
    )


def readme_to_html(data_dir: Path) -> str:
    """Render a collection's ``README.md`` as HTML for the Zenodo description.

    Returns ``""`` when the collection has no README. The rendered HTML is the
    descriptive body of the record, shown below the file-change note.
    """
    readme_path = data_dir / "README.md"
    if not readme_path.is_file():
        return ""
    if markdown is None:
        return f"<pre>{html.escape(readme_path.read_text(encoding='utf-8'))}</pre>"
    return markdown.markdown(
        readme_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists"],
    )


def build_description(
    manifest: dict,
    section: str,
    files: dict,
    added: list[str],
    updated: list[str],
    removed: list[str],
    initial: bool,
    readme_html: str = "",
) -> str:
    data_version = manifest["data_version"]
    title = COLLECTIONS[section]["title"]
    esc = html.escape

    def names_html(names: list[str]) -> str:
        return "".join(f"<li><code>{esc(name)}</code></li>" for name in names)

    if initial:
        change_note = (
            f"<p>This is the <strong>initial release</strong> of this dataset. "
            f"All {len(files)} files are new in this version.</p>"
        )
    else:
        changes_html = []
        if added:
            changes_html.append(
                f"<p><strong>Files added in this release:</strong></p><ul>{names_html(added)}</ul>"
            )
        if updated:
            changes_html.append(
                f"<p><strong>Files updated in this release:</strong></p><ul>{names_html(updated)}</ul>"
            )
        if removed:
            changes_html.append(
                f"<p><strong>Files removed in this release:</strong></p><ul>{names_html(removed)}</ul>"
            )
        change_note = (
            "".join(changes_html) or "<p><em>No file changes in this release.</em></p>"
        )

    file_sections = "\n".join(
        describe_file(filename, info) for filename, info in sorted(files.items())
    )

    return (
        f"<p>{esc(title)}: PowerGenome input data assembled from public sources. "
        "This release corresponds to a PowerGenome-data manifest at data version "
        f"<code>{esc(data_version)}</code>.</p>"
        f"{change_note}"
        f"{readme_html}"
        "<p><strong>Licensing:</strong> this compilation as a whole is released under "
        "Creative Commons Zero (CC0, public domain dedication). Because the files assemble "
        "public data from a variety of sources, each file retains the license of its "
        "underlying source: U.S. government works (e.g. EIA, BEA, FRED) are in the public "
        "domain, data derived from ReEDS / NREL (including the NREL ATB and PUDL) is "
        "Creative Commons Attribution 4.0 International (CC BY 4.0), and NERC LTRA data is "
        "CC BY 4.0. See each file's License row below.</p>"
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
        raise SystemExit(
            f"{action} failed with HTTP {response.status_code}: {detail}"
        ) from exc


def next_release_version(
    session: requests.Session,
    base_url: str,
    data_version: str,
    conceptrecid,
    local_versions: list[str] | None = None,
) -> str:
    """Return the Zenodo version string for a release of ``data_version``.

    The base is the manifest data_version (e.g. "2026.08.14") with a
    sequential per-day suffix so multiple releases on the same date never
    share a version number (e.g. "2026.08.14.v1", "2026.08.14.v2", ...).
    The suffix is one more than the larger of two counts, each counting the
    releases that already carry this data_version (bare or ``.vN``-suffixed):
    the versions published on Zenodo for this concept (via the
    eventually-consistent records search) and the release versions recorded
    locally in the saved state (``zenodo_release.versions``). The local
    history closes the window where a fast follow-up release would not yet be
    indexed by Zenodo's search, preventing two releases from reusing a
    suffix. A brand-new deposition (no concept yet) gets ".v1".
    """

    def matches(version: str) -> bool:
        return version == data_version or version.startswith(f"{data_version}.")

    local_count = sum(1 for v in (local_versions or []) if matches(v))
    remote_count = 0
    if conceptrecid:
        query = urllib.parse.quote(f'conceptrecid:"{conceptrecid}"')
        response = session.get(
            f"{base_url}/records/?q={query}&all_versions=true&size=100",
            timeout=120,
        )
        raise_for_status(response, "Listing published versions")
        for hit in response.json().get("hits", {}).get("hits", []):
            version = (hit.get("metadata") or {}).get("version")
            if version and matches(version):
                remote_count += 1
    return f"{data_version}.v{max(local_count, remote_count) + 1}"


def create_deposition(session: requests.Session, base_url: str) -> dict:
    response = session.post(
        f"{base_url}/deposit/depositions",
        json={},
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    raise_for_status(response, "Creating deposition")
    return response.json()


def get_deposition(
    session: requests.Session, base_url: str, deposition_id: str
) -> dict:
    response = session.get(
        f"{base_url}/deposit/depositions/{deposition_id}", timeout=120
    )
    raise_for_status(response, f"Fetching deposition {deposition_id}")
    return response.json()


def create_new_version(session: requests.Session, base_url: str, record: str) -> dict:
    response = session.post(
        f"{base_url}/deposit/depositions/{record}/actions/newversion",
        timeout=120,
    )
    raise_for_status(response, "Creating new version draft")
    payload = response.json()
    draft_url = payload.get("links", {}).get("latest_draft")
    if draft_url:
        draft_response = session.get(draft_url, timeout=120)
        raise_for_status(draft_response, "Fetching new draft deposition")
        return draft_response.json()
    return payload


def upload_file(
    session: requests.Session, bucket_url: str, path: Path, key: str | None = None
) -> dict:
    checksum = md5_for_file(path)
    key = key or path.name
    with path.open("rb") as handle:
        response = session.put(f"{bucket_url}/{key}", data=handle, timeout=3600)
    raise_for_status(response, f"Uploading {key}")

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


def set_metadata(
    session: requests.Session, base_url: str, deposition_id: str, metadata: dict
) -> dict:
    response = session.put(
        f"{base_url}/deposit/depositions/{deposition_id}",
        json={"metadata": metadata},
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    raise_for_status(response, "Updating metadata")
    return response.json()


def publish_deposition(
    session: requests.Session, base_url: str, deposition_id: str
) -> dict:
    response = session.post(
        f"{base_url}/deposit/depositions/{deposition_id}/actions/publish",
        timeout=120,
    )
    raise_for_status(response, "Publishing deposition")
    return response.json()


def pre_reserved_doi(payload: dict) -> str | None:
    return payload.get("metadata", {}).get("prereserve_doi", {}).get("doi")


def draft_page_url(base_url: str, deposition_id: str) -> str:
    host = (
        "https://zenodo.org"
        if base_url == PRODUCTION_BASE
        else "https://sandbox.zenodo.org"
    )
    return f"{host}/deposit/{deposition_id}"


def existing_draft_files(
    session: requests.Session, deposition: dict
) -> dict[str, tuple[str, str]]:
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
        sys.exit(
            f"Zenodo allows at most {MAX_FILES} files per record; manifest has {len(files_by_name)}."
        )
    if total_bytes > MAX_TOTAL_BYTES:
        sys.exit(
            f"Zenodo allows at most 50 GB per record; total is {total_bytes / 1e9:.1f} GB."
        )


def uncommitted_release_files(manifest_files: dict, data_dir: Path) -> list[str]:
    """Names of files in a manifest section that differ from git HEAD.

    Publishing a release whose files are not in git history means the Zenodo
    record has no reproducible commit behind it. Returns ``[]`` when git is
    unavailable or the tree is clean for every file.
    """
    paths = [data_dir / name for name in manifest_files]
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--"] + [str(p) for p in paths],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    dirty = []
    for line in out.splitlines():
        if not line.strip():
            continue
        # porcelain format: XY <path>; X = staged, Y = worktree. '', M, D, A,
        # or '?' (untracked) all mean the file differs from HEAD.
        if line[0] in "MAD?" or line[1:2] in "MD":
            dirty.append(line)
    return dirty


def build_release_description(
    manifest: dict,
    section: str,
    files: dict,
    added: list[str],
    updated: list[str],
    removed: list[str],
    initial: bool,
    base_metadata: dict,
    readme_html: str = "",
) -> str:
    """Full Zenodo description for a deposit.

    Layout: generated title + file-change note, then the collection's
    ``README.md`` rendered as HTML (the descriptive body, if present), then
    the licensing paragraph and per-file tables. Any custom prose supplied via
    ``metadata.sections.<section>.description`` is prepended as a leading
    paragraph.
    """
    description = build_description(
        manifest,
        section,
        files,
        added,
        updated,
        removed,
        initial,
        readme_html,
    )
    custom_description = base_metadata.get("description")
    if custom_description:
        description = f"{custom_description}\n{description}"
    return description


def release_section(
    args,
    session: requests.Session,
    base_url: str,
    environment: str,
    manifest: dict,
    section: str,
    manifest_files: dict,
    data_dir: Path,
    state: dict,
) -> dict:
    """Create/update/publish the Zenodo deposit for one manifest section.

    Returns ``{"section": ..., "summary": ..., "release": ..., "metadata": ...}``
    where ``summary`` is the per-section run report, ``release`` is the new
    state to persist under ``releases.<section>``, and ``metadata`` is the
    Zenodo metadata that was set (or would be set) for the deposit.
    """
    data_version = manifest["data_version"]
    config = COLLECTIONS[section]
    key_prefix = config["key_prefix"]

    released = section_release(state, section)
    # Release state is environment-specific: deposition ids, DOIs, and the
    # version suffix counter differ between sandbox and production. Treat a
    # stored release from the *other* environment as absent so each environment
    # starts its own first release, while keeping shared metadata like creators.
    stored_environment = released.get("environment", "sandbox")
    if released and stored_environment != environment:
        log(
            f"[{section}] stored release state is for {stored_environment}, "
            f"target is {environment}; ignoring it"
        )
        released = {}
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
            sys.exit(
                f"[{section}] manifest lists {filename} but {local_path} does not exist on disk."
            )
        actual = md5_for_file(local_path)
        local_md5s[filename] = actual
        expected = manifest_files[filename].get("md5")
        if expected and actual != expected:
            sys.exit(
                f"[{section}] manifest md5 mismatch for {filename}: manifest says {expected}, "
                f"file has {actual}. Run update_data_manifest.py before releasing."
            )
        local_paths[filename] = local_path
    validate_constraints(local_paths)

    # Keys used inside the deposit bucket. Prefixed per section so filenames
    # cannot collide across collections.
    keys = {name: f"{key_prefix}{name}" for name in manifest_files}

    # Determine which files were updated in this release.
    added = [name for name in manifest_files if keys[name] not in released_files]
    updated = [
        name
        for name, info in manifest_files.items()
        if keys[name] in released_files
        and released_files[keys[name]] != info.get("md5")
    ]
    changed = added + updated
    removed = [key for key in released_files if key not in set(keys.values())]
    initial = not released_files

    log(f"[{section}] data version: {data_version}")
    if added:
        log(f"[{section}] files added ({len(added)}): {', '.join(added)}")
    if updated:
        log(f"[{section}] files updated ({len(updated)}): {', '.join(updated)}")
    if removed:
        log(
            f"[{section}] files dropped from manifest ({len(removed)}): {', '.join(removed)}"
        )
    if not changed and not removed:
        log(
            f"[{section}] no file changes detected since last release; "
            "the description/version may still be updated"
        )

    # Nothing to do if this exact version was already published unchanged.
    if (
        previously_published
        and data_version == released.get("data_version")
        and not changed
        and not removed
    ):
        log(
            f"[{section}] version {data_version} is already published on Zenodo "
            f"(deposition {deposition_id}); nothing to do."
        )
        return {
            "section": section,
            "summary": {
                "section": section,
                "mode": "published",
                "environment": environment,
                "data_version": data_version,
                "release_version": released.get("release_version"),
                "deposition_id": deposition_id,
                "doi": released.get("doi"),
                "skipped": True,
            },
            "release": released,
            "metadata": section_metadata(state, section),
        }

    # Publishing files that aren't in git history leaves a record with no
    # reproducible commit behind it. Refuse unless --allow-dirty is passed.
    dirty = uncommitted_release_files(manifest_files, data_dir)
    if dirty and not getattr(args, "allow_dirty", False):
        log(f"[{section}] uncommitted changes vs git HEAD in release files:")
        for line in dirty:
            log(f"  {line}")
        sys.exit(
            f"[{section}] release files differ from git HEAD. Commit the changes first, or "
            "re-run with --allow-dirty. Publishing an uncommitted release leaves "
            "no reproducible commit for the Zenodo record."
        )

    # Resolve the target draft.
    if deposition_id and not previously_published:
        log(f"[{section}] resuming unpublished draft {deposition_id}")
        deposition = get_deposition(session, base_url, str(deposition_id))
    elif deposition_id:
        log(f"[{section}] creating new version from deposition {deposition_id}")
        deposition = create_new_version(session, base_url, str(deposition_id))
    else:
        log(f"[{section}] creating a new deposition")
        deposition = create_deposition(session, base_url)

    deposition_id = str(deposition["id"])
    bucket_url = deposition["links"]["bucket"]
    release_version = next_release_version(
        session,
        base_url,
        data_version,
        deposition.get("conceptrecid"),
        released.get("versions") if isinstance(released, dict) else None,
    )
    log(f"[{section}] release version: {release_version}")

    # Remove files no longer part of the manifest (covers released-then-dropped
    # files and any leftovers in a resumed draft / previous version).
    wanted_keys = set(keys.values())
    present = existing_draft_files(session, deposition)
    for key, (file_id, _checksum) in present.items():
        if key in wanted_keys:
            continue
        file_url = deposition["links"]["files"] + "/" + file_id
        log(f"[{section}] removing {key} (no longer in manifest)")
        delete_file(session, file_url, key)

    # Upload files whose local checksum differs from what the draft already
    # holds (skips files unchanged since a prior draft copy / new-version).
    uploads = []
    for index, (filename, local_path) in enumerate(local_paths.items()):
        key = keys[filename]
        local_md5 = local_md5s[filename]
        _, draft_checksum = present.get(key, (None, None))
        if draft_checksum == local_md5:
            log(f"[{section}] already up to date in draft, skipping: {key}")
            continue
        log(f"[{section}] uploading {key} ({local_path.stat().st_size / 1e6:.1f} MB)")
        result = upload_file(session, bucket_url, local_path, key=key)
        result["filename"] = key
        uploads.append(result)
        if index < len(local_paths) - 1 and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    # Build + update metadata.
    base_metadata = section_metadata(state, section)
    description = build_release_description(
        manifest,
        section,
        manifest_files,
        added,
        updated,
        removed,
        initial,
        base_metadata,
        readme_html=readme_to_html(data_dir),
    )
    creators = base_metadata.get("creators") or default_creators()
    metadata = dict(base_metadata)
    # Core keeps its long-standing title unless explicitly overridden.
    metadata.update(
        {
            "title": base_metadata.get("title") or COLLECTIONS[section]["title"],
            "upload_type": base_metadata.get("upload_type") or "dataset",
            "access_right": base_metadata.get("access_right") or "open",
            "creators": creators,
            "version": release_version,
            "description": description,
        }
    )

    updated = set_metadata(session, base_url, deposition_id, metadata)

    # Publish (only with --publish) or leave as draft.
    published = False
    publish_payload = {}
    if args.publish:
        log(f"[{section}] publishing deposition {deposition_id}")
        publish_payload = publish_deposition(session, base_url, deposition_id)
        published = True
    else:
        log(
            f"[{section}] draft only (no --publish); draft: {draft_page_url(base_url, deposition_id)}"
        )

    summary = {
        "section": section,
        "mode": "publish" if published else "draft",
        "environment": environment,
        "data_version": data_version,
        "release_version": release_version,
        "deposition_id": deposition_id,
        "draft_url": draft_page_url(base_url, deposition_id),
        "pre_reserved_doi": pre_reserved_doi(updated) or pre_reserved_doi(deposition),
        "doi": publish_payload.get("doi") if published else None,
        "record_url": (
            (
                publish_payload.get("links", {}).get("record_html")
                or publish_payload.get("links", {}).get("html")
            )
            if published
            else None
        ),
        "updated_files": [u["filename"] for u in uploads],
        "removed_files": removed,
        "check_uploads": uploads,
    }

    # Track released version strings per environment so a fast follow-up
    # release never reuses a suffix even while Zenodo's search index lags.
    prior_versions = released.get("versions", []) if isinstance(released, dict) else []
    versions = list(prior_versions)
    if published and release_version not in versions:
        versions.append(release_version)
    new_release = {
        "environment": environment,
        "data_version": data_version,
        "release_version": release_version,
        "deposition_id": deposition_id,
        "published": published,
        "doi": publish_payload.get("doi") if published else None,
        "versions": versions,
        "files": {keys[name]: info["md5"] for name, info in manifest_files.items()},
    }

    return {
        "section": section,
        "summary": summary,
        "release": new_release,
        "metadata": metadata,
    }


def requested_sections(args) -> list[str]:
    """Manifest sections to process, honoring repeated ``--collection`` flags.

    With no ``--collection`` flags every collection is processed (in registry
    order). Names are validated by argparse ``choices``.
    """
    if not getattr(args, "collection", None):
        return list(COLLECTIONS)
    return list(args.collection)


def run_release(args) -> None:
    global PROJECT_ROOT, STATE_PATH
    PROJECT_ROOT = Path.cwd()
    STATE_PATH = PROJECT_ROOT / ".zenodo.json"

    load_dotenv(Path(args.dotenv_path).expanduser())
    use_production = args.use_production or truthy_env("USE_PRODUCTION")
    environment = "production" if use_production else "sandbox"
    token = load_token(use_production)
    session = make_session(token)
    base_url = resolve_base(use_production)

    manifests = load_manifests(args, PROJECT_ROOT)
    sections = {
        section: section_manifest["files"]
        for section, section_manifest in manifests.items()
        if section in requested_sections(args)
    }
    for section in requested_sections(args):
        if section not in sections:
            log(f"[{section}] not present in manifest; skipping")

    state = load_state()
    save_state = args.publish or args.save_state

    results = []
    for section, manifest_files in sections.items():
        if not manifest_files:
            log(f"[{section}] no files in manifest section; skipping")
            continue
        data_dir = PROJECT_ROOT / COLLECTIONS[section]["data_dir"]
        if not data_dir.is_dir():
            sys.exit(
                f"[{section}] data directory {data_dir} does not exist but the "
                "manifest section lists files."
            )
        results.append(
            release_section(
                args,
                session,
                base_url,
                environment,
                manifests[section],
                section,
                manifest_files,
                data_dir,
                state,
            )
        )

    if not results:
        log("no manifest sections with files; nothing to do.")
        return

    # Persist release state (after a successful publish, or on resume so the
    # draft can be continued; changed-file diff only applies when published).
    if save_state:
        project_metadata = state.get("metadata", {}) if isinstance(state, dict) else {}
        if not isinstance(project_metadata, dict):
            project_metadata = {}
        project_metadata = dict(project_metadata)
        # Description/version in stored metadata are per-deposit; keep the
        # shared fields (creators, license, access_right, ...) only.
        for per_release_field in ("title", "description", "version"):
            project_metadata.pop(per_release_field, None)
        new_state = {
            "metadata": project_metadata,
            "releases": {result["section"]: result["release"] for result in results},
        }
        STATE_PATH.write_text(
            json.dumps(new_state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        log(f"state written to {STATE_PATH}")

    summaries = [result["summary"] for result in results]
    if any(not s.get("skipped") and s["mode"] == "draft" for s in summaries):
        log(
            "Not all deposits published. Run again with --publish to publish the drafts, "
            "or download from the draft URLs in the summary."
        )
    print(json.dumps({"collections": summaries}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="publish_zenodo.py",
        description="Publish/update data/manifest.json collections to Zenodo, one deposit per section (sandbox by default).",
    )
    parser.add_argument(
        "--manifest",
        default="data/manifest.json",
        help=(
            "Path to the core collection manifest (default: data/manifest.json). "
            "The profiles and existing_resource_groups collections are read from "
            "<folder>/manifest.json."
        ),
    )
    parser.add_argument(
        "--dotenv-path",
        default=".env",
        help="Optional dotenv file supplying ZENODO_SANDBOX_API_KEY / ZENODO_API_KEY.",
    )
    parser.add_argument(
        "--use-production",
        "--production",
        dest="use_production",
        action="store_true",
        help="Use production Zenodo instead of the sandbox.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish the deposition (irreversible in production). Default is draft-only.",
    )
    parser.add_argument(
        "--save-state",
        action="store_true",
        help="Write .zenodo.json even when only creating a draft. Always written on publish.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Delay between file uploads (default 1.0).",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Publish even when release files differ from git HEAD.",
    )
    parser.add_argument(
        "--collection",
        action="append",
        choices=list(COLLECTIONS),
        metavar="COLLECTION",
        help=(
            "Only release this collection; repeatable (e.g. --collection profiles). "
            "Default: release every collection in the manifest. Choices: "
            + ", ".join(sorted(COLLECTIONS))
            + "."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the release plan without calling the Zenodo API.",
    )
    return parser


def dry_run(args) -> None:
    project_root = Path.cwd()
    manifests = load_manifests(args, project_root)
    sections = requested_sections(args)
    print(f"data version: {manifests[CORE_SECTION]['data_version']}")
    for section in sections:
        if section not in manifests:
            print(f"\n[{section}] not present in manifest; skipping")
            continue
        section_manifest = manifests[section]
        files = section_manifest["files"]
        config = COLLECTIONS[section]
        data_dir = project_root / config["data_dir"]
        print(
            f"\n[{section}] {config['title']} ({config['data_dir']}/) "
            f"[data version {section_manifest['data_version']}]"
        )
        if not files:
            print("  (no files in this manifest section; deposit skipped)")
            continue
        if not data_dir.is_dir():
            print(f"  ERROR: data directory {data_dir} does not exist")
            continue
        print(f"  manifest files: {len(files)}")
        for name, info in sorted(files.items()):
            missing = not (data_dir / name).exists()
            print(
                f"  {'MISSING ' if missing else 'ok      '}{config['key_prefix']}{name}  (version {info.get('version')})"
            )
        total = sum(
            (data_dir / n).stat().st_size for n in files if (data_dir / n).exists()
        )
        print(f"  total bytes: {total / 1e6:.1f} MB")


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        dry_run(args)
        return
    run_release(args)


if __name__ == "__main__":
    main()
