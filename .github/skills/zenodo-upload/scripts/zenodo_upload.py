import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError as exc:
    raise SystemExit(
        "The helper script requires the 'requests' package. Install it or follow the curl fallback in zenodo-upload/references/api-reference.md."
    ) from exc


SANDBOX_BASE = "https://sandbox.zenodo.org/api"
PRODUCTION_BASE = "https://zenodo.org/api"
MAX_FILES = 100
MAX_TOTAL_BYTES = 50 * 1024 * 1024 * 1024


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        os.environ.setdefault(key, value)


def resolve_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists() or (candidate / ".zenodo.json").exists():
            return candidate
    return current.resolve()


def truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_base(use_production: bool) -> str:
    override = os.getenv("ZENODO_BASE")
    if override:
        return override.rstrip("/")
    return PRODUCTION_BASE if use_production else SANDBOX_BASE


def load_token() -> str:
    token = os.getenv("ZENODO_TOKEN")
    if not token:
        raise SystemExit("ZENODO_TOKEN is required and must be provided through the environment or a local .env file.")
    return token


def md5_for_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def total_size(paths: list[Path]) -> int:
    return sum(path.stat().st_size for path in paths)


def validate_file_constraints(paths: list[Path]) -> None:
    if len(paths) > MAX_FILES:
        raise SystemExit(f"Zenodo allows at most {MAX_FILES} files per record; received {len(paths)} files.")

    size_bytes = total_size(paths)
    if size_bytes > MAX_TOTAL_BYTES:
        gib = size_bytes / 1024 / 1024 / 1024
        raise SystemExit(f"Zenodo allows at most 50 GB per record; received {gib:.2f} GB.")


def validate_paths(file_args: list[str]) -> list[Path]:
    paths = [Path(item).expanduser().resolve() for item in file_args]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"These file paths do not exist: {missing}")

    directories = [str(path) for path in paths if path.is_dir()]
    if directories:
        raise SystemExit(f"Directories are not supported; provide concrete files instead: {directories}")

    validate_file_constraints(paths)
    return paths


def metadata_from_file(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "metadata" in payload and isinstance(payload["metadata"], dict):
        return payload["metadata"]
    if not isinstance(payload, dict):
        raise SystemExit(f"Metadata file must contain a JSON object: {path}")
    return payload


def metadata_from_json(raw_json: str | None) -> dict:
    if not raw_json:
        return {}
    payload = json.loads(raw_json)
    if "metadata" in payload and isinstance(payload["metadata"], dict):
        return payload["metadata"]
    if not isinstance(payload, dict):
        raise SystemExit("Inline metadata must be a JSON object.")
    return payload


def merge_metadata(*parts: dict) -> dict:
    merged: dict = {}
    for part in parts:
        for key, value in part.items():
            if value is not None:
                merged[key] = value
    return merged


def validate_metadata(metadata: dict) -> None:
    required = ["title", "upload_type", "description", "creators"]
    missing = [field for field in required if not metadata.get(field)]
    if missing:
        raise SystemExit(f"Metadata is missing required fields: {missing}")

    creators = metadata.get("creators")
    if not isinstance(creators, list) or not creators:
        raise SystemExit("Metadata 'creators' must be a non-empty list.")

    invalid_creators = [creator for creator in creators if not isinstance(creator, dict) or not creator.get("name")]
    if invalid_creators:
        raise SystemExit("Each creator entry must be an object with a 'name' field.")

    if metadata.get("access_right") == "embargoed" and not metadata.get("embargo_date"):
        raise SystemExit("Embargoed records require 'embargo_date' in YYYY-MM-DD format.")


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
        f"{base_url}/deposit/depositions",
        json={},
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    raise_for_status(response, "Creating deposition")
    return response.json()


def resolve_record_id(session: requests.Session, base_url: str, record: str) -> str:
    if record.isdigit():
        return record

    response = session.get(
        f"{base_url}/records",
        params={"q": f'doi:"{record}"'},
        timeout=120,
    )
    raise_for_status(response, "Resolving DOI")
    payload = response.json()
    hits = payload.get("hits", {}).get("hits", [])
    if not hits:
        raise SystemExit(f"No Zenodo record matched DOI '{record}'.")
    return str(hits[0]["id"])


def create_new_version(session: requests.Session, base_url: str, record: str) -> dict:
    record_id = resolve_record_id(session, base_url, record)
    response = session.post(
        f"{base_url}/deposit/depositions/{record_id}/actions/newversion",
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
        "path": str(path),
        "size": path.stat().st_size,
        "local_md5": checksum,
        "remote_checksum": payload.get("checksum"),
        "response": payload,
    }


def set_metadata(session: requests.Session, base_url: str, deposition_id: str, metadata: dict) -> dict:
    response = session.put(
        f"{base_url}/deposit/depositions/{deposition_id}",
        json={"metadata": metadata},
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    raise_for_status(response, "Updating metadata")
    return response.json()


def publish_deposition(session: requests.Session, base_url: str, deposition_id: str) -> dict:
    response = session.post(
        f"{base_url}/deposit/depositions/{deposition_id}/actions/publish",
        timeout=120,
    )
    raise_for_status(response, "Publishing deposition")
    return response.json()


def draft_page_url(base_url: str, deposition_id: str) -> str:
    host = "https://zenodo.org" if base_url == PRODUCTION_BASE else "https://sandbox.zenodo.org"
    return f"{host}/deposit/{deposition_id}"


def pre_reserved_doi(payload: dict) -> str | None:
    return payload.get("metadata", {}).get("prereserve_doi", {}).get("doi")


def load_project_metadata(project_root: Path, skip_project_zenodo_json: bool) -> dict:
    if skip_project_zenodo_json:
        return {}

    project_metadata_path = project_root / ".zenodo.json"
    if not project_metadata_path.exists():
        return {}
    return metadata_from_file(project_metadata_path)


def maybe_write_project_metadata(project_root: Path, metadata: dict, enabled: bool) -> Path | None:
    if not enabled:
        return None

    project_metadata_path = project_root / ".zenodo.json"
    project_metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return project_metadata_path


def draft_workflow(args: argparse.Namespace, mode: str) -> None:
    project_root = resolve_project_root(Path.cwd())
    load_dotenv(Path(args.dotenv_path).expanduser())
    token = load_token()
    base_url = resolve_base(args.use_production or truthy_env("USE_PRODUCTION"))
    metadata_parts = [load_project_metadata(project_root, args.skip_project_zenodo_json)]
    if args.metadata_file:
        metadata_parts.append(metadata_from_file(Path(args.metadata_file).expanduser().resolve()))
    metadata_parts.append(metadata_from_json(args.metadata_json))
    metadata = merge_metadata(*metadata_parts)
    validate_metadata(metadata)
    file_paths = validate_paths(args.file)

    session = make_session(token)
    deposition = create_deposition(session, base_url) if mode == "create-draft" else create_new_version(session, base_url, args.record)
    deposition_id = str(deposition["id"])
    bucket_url = deposition["links"]["bucket"]
    uploads = []
    for index, path in enumerate(file_paths):
        uploads.append(upload_file(session, bucket_url, path))
        if index < len(file_paths) - 1 and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    updated = set_metadata(session, base_url, deposition_id, metadata)
    written_path = maybe_write_project_metadata(project_root, metadata, args.write_project_zenodo_json)
    summary = {
        "mode": mode,
        "base_url": base_url,
        "deposition_id": deposition_id,
        "draft_url": draft_page_url(base_url, deposition_id),
        "bucket_url": bucket_url,
        "pre_reserved_doi": pre_reserved_doi(updated) or pre_reserved_doi(deposition),
        "uploaded_files": uploads,
        "metadata": metadata,
        "project_metadata_path": str(written_path) if written_path else None,
        "publish_required": True,
    }
    print(json.dumps(summary, indent=2))


def publish_workflow(args: argparse.Namespace) -> None:
    if not args.confirm_publish:
        raise SystemExit("Publishing requires --confirm-publish to enforce explicit confirmation.")

    load_dotenv(Path(args.dotenv_path).expanduser())
    token = load_token()
    base_url = resolve_base(args.use_production or truthy_env("USE_PRODUCTION"))
    session = make_session(token)
    payload = publish_deposition(session, base_url, str(args.deposition_id))
    summary = {
        "mode": "publish",
        "base_url": base_url,
        "deposition_id": str(payload.get("id", args.deposition_id)),
        "doi": payload.get("doi") or payload.get("metadata", {}).get("doi"),
        "record_url": payload.get("links", {}).get("record_html") or payload.get("links", {}).get("html"),
        "response": payload,
    }
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create, update, and publish Zenodo depositions.")
    parser.add_argument("--dotenv-path", default=".env", help="Optional dotenv file to read before resolving environment variables.")
    parser.add_argument("--use-production", action="store_true", help="Use production Zenodo instead of the sandbox.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    draft_parent = argparse.ArgumentParser(add_help=False)
    draft_parent.add_argument("--file", action="append", required=True, help="File to upload. Repeat for multiple files.")
    draft_parent.add_argument("--metadata-file", help="Path to a JSON file containing Zenodo metadata.")
    draft_parent.add_argument("--metadata-json", help="Inline JSON metadata object.")
    draft_parent.add_argument("--skip-project-zenodo-json", action="store_true", help="Ignore a project root .zenodo.json file if present.")
    draft_parent.add_argument("--write-project-zenodo-json", action="store_true", help="Write merged metadata back to the project root .zenodo.json file.")
    draft_parent.add_argument("--sleep-seconds", type=float, default=1.0, help="Delay between file uploads.")

    create_draft = subparsers.add_parser("create-draft", parents=[draft_parent], help="Create a new deposition draft, upload files, and apply metadata.")
    create_draft.set_defaults(handler=lambda args: draft_workflow(args, "create-draft"))

    new_version = subparsers.add_parser("new-version", parents=[draft_parent], help="Create a draft for a new version of an existing deposition.")
    new_version.add_argument("--record", required=True, help="Existing deposition ID or DOI.")
    new_version.set_defaults(handler=lambda args: draft_workflow(args, "new-version"))

    publish = subparsers.add_parser("publish", help="Publish an existing draft deposition.")
    publish.add_argument("--deposition-id", required=True, help="Draft deposition ID to publish.")
    publish.add_argument("--confirm-publish", action="store_true", help="Required confirmation flag for publish operations.")
    publish.set_defaults(handler=publish_workflow)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()