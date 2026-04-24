# Zenodo API Reference

Use this reference when the helper script is unavailable, when debugging API failures, or when you need a manual `curl` fallback.

## Base URLs

- Sandbox: `https://sandbox.zenodo.org/api`
- Production: `https://zenodo.org/api`

Default to sandbox. Only switch to production after the user explicitly confirms.

## Authentication

All requests require a personal access token with these scopes:

- `deposit:write`
- `deposit:actions`

Pass the token as an HTTP header:

```text
Authorization: Bearer <ACCESS_TOKEN>
```

## Core Workflow

### 1. Create an empty deposition

```bash
curl -sS -X POST "${ZENODO_BASE:-https://sandbox.zenodo.org/api}/deposit/depositions" \
  -H "Authorization: Bearer ${ZENODO_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{}'
```

Success response: `201 Created`

Important fields in the response:

- `id`
- `links.bucket`
- `links.publish`
- `metadata.prereserve_doi.doi`

### 2. Upload files to the bucket

```bash
curl -sS -X PUT "${BUCKET_URL}/results.zip" \
  -H "Authorization: Bearer ${ZENODO_TOKEN}" \
  --upload-file results.zip
```

Success response: `200 OK`

Constraints:

- Up to 100 files
- Up to 50 GB total per record

### 3. Set deposition metadata

```bash
curl -sS -X PUT "${ZENODO_BASE:-https://sandbox.zenodo.org/api}/deposit/depositions/${DEPOSITION_ID}" \
  -H "Authorization: Bearer ${ZENODO_TOKEN}" \
  -H "Content-Type: application/json" \
  -d @metadata.json
```

Success response: `200 OK`

Payload format:

```json
{
  "metadata": {
    "title": "Required title",
    "upload_type": "dataset",
    "description": "<p>Required description</p>",
    "creators": [{"name": "Lastname, Firstname"}]
  }
}
```

### 4. Publish the deposition

```bash
curl -sS -X POST "${ZENODO_BASE:-https://sandbox.zenodo.org/api}/deposit/depositions/${DEPOSITION_ID}/actions/publish" \
  -H "Authorization: Bearer ${ZENODO_TOKEN}"
```

Success response: `202 Accepted`

Warning: publish is irreversible in production.

## High-Value Optional Metadata

```json
{
  "metadata": {
    "access_right": "open",
    "license": "cc-by-4.0",
    "keywords": ["power-systems", "dataset"],
    "version": "1.0.0",
    "language": "eng",
    "communities": [{"identifier": "community-slug"}],
    "related_identifiers": [
      {
        "identifier": "10.1234/example",
        "relation": "isSupplementTo",
        "scheme": "doi"
      }
    ]
  }
}
```

## Additional Endpoints

| Action | Method | Endpoint |
| --- | --- | --- |
| List depositions | `GET` | `/api/deposit/depositions` |
| Get deposition | `GET` | `/api/deposit/depositions/<id>` |
| Update metadata | `PUT` | `/api/deposit/depositions/<id>` |
| Delete draft | `DELETE` | `/api/deposit/depositions/<id>` |
| Unlock published record for edit | `POST` | `/api/deposit/depositions/<id>/actions/edit` |
| Discard edits | `POST` | `/api/deposit/depositions/<id>/actions/discard` |
| Create new version | `POST` | `/api/deposit/depositions/<id>/actions/newversion` |
| List files | `GET` | `/api/deposit/depositions/<id>/files` |
| Delete file from bucket | `DELETE` | `/api/files/<bucket_id>/<filename>` |

## Status Codes

- `200` or `201`: success
- `202`: accepted, typically for publish
- `400`: invalid request or invalid metadata
- `401`: missing or invalid token
- `403`: insufficient scope or forbidden action
- `404`: missing deposition or file
- `415`: wrong content type
- `500`: Zenodo error, retry after checking service health

## Common Failure Checks

- `400`: verify `title`, `upload_type`, `description`, and `creators[].name`
- `401`: verify `ZENODO_TOKEN` is set
- `403`: verify token scopes and that you are editing a draft
- `415`: set `Content-Type: application/json` for metadata requests
- Bucket upload checksum mismatch: recompute local MD5 and retry