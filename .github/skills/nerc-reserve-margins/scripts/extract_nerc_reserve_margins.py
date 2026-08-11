#!/usr/bin/env python3
"""Extract NERC LTRA reference margin levels into a tidy CSV."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from pypdf import PdfReader

DEFAULT_ASSESSMENTS_URL = (
    "https://www.nerc.com/our-work/assessments/long-term-reliability-assessments"
)
DEFAULT_PDF_URL = (
    "https://www.nerc.com/globalassets/our-work/assessments/nerc_ltra_2025.pdf"
)
DEFAULT_OUTPUT = Path("data/nerc_reserve_margins.csv")
DEFAULT_VINTAGE = Path("data/nerc_reserve_margins_vintage.json")
DEFAULT_CACHE_DIR = Path("cache_data/nerc_ltra")
REQUEST_TIMEOUT_SECONDS = 60
TARGET_YEAR = 2050
VALUE_DECIMAL_PLACES = 3

REFERENCE_ROW = re.compile(
    r"^Reference\s+Margin\s+Level\s*\(%\)\s*(?P<values>.*)$", re.IGNORECASE
)
YEAR_TOKEN = re.compile(r"20\d{2}(?:\s*[\u2013-]\s*20\d{2})?")
PERCENT_TOKEN = re.compile(r"[-+]?\d+(?:\.\d+)?\s*%")
REPORT_HEADER = re.compile(
    r"20\d{2}\s+Long\s*-?\s*Term\s+Reliability\s+Assessment\s+\d+"
)


class ExtractionError(ValueError):
    """Raised when a report page cannot be parsed unambiguously."""


def normalize_text(text: str) -> str:
    """Normalize PDF typography and whitespace while preserving line boundaries."""
    text = text.replace("\u00a0", " ").replace("\u2011", "-")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[ \t]+", " ", text)
    return "\n".join(line.strip() for line in text.splitlines())


def parse_years(line: str) -> list[int]:
    """Return the first year from each annual or year-range header token."""
    years = []
    for token in YEAR_TOKEN.findall(line):
        years.append(int(token[:4]))
    return years


def parse_reference_row(line: str) -> list[float]:
    """Parse percentage values from a Reference Margin Level row."""
    match = REFERENCE_ROW.match(line.strip())
    if not match:
        raise ExtractionError(f"Not a reference margin row: {line!r}")
    values = [
        round(float(token[:-1].strip()) / 100, VALUE_DECIMAL_PLACES)
        for token in PERCENT_TOKEN.findall(match.group("values"))
    ]
    if not values:
        raise ExtractionError(
            "Reference Margin Level row contains no percentage values"
        )
    if any(value < 0 or value > 1 for value in values):
        raise ExtractionError(f"Reference margin values must be percentages: {values}")
    return values


def infer_region(lines: list[str], page_number: int) -> str:
    """Find the assessment-area heading at the start of a detailed report page."""
    for index, line in enumerate(lines):
        if REPORT_HEADER.search(line):
            for candidate in lines[index + 1 : index + 8]:
                if candidate and not candidate.isdigit():
                    return candidate
    raise ExtractionError(
        f"Could not identify an assessment area on PDF page {page_number}"
    )


def parse_page(text: str, page_number: int) -> pd.DataFrame | None:
    """Extract one assessment-area table from a PDF page, if present."""
    lines = normalize_text(text).splitlines()
    row_indexes = [
        index for index, line in enumerate(lines) if REFERENCE_ROW.match(line)
    ]
    if not row_indexes:
        return None
    if len(row_indexes) != 1:
        raise ExtractionError(
            f"PDF page {page_number} has {len(row_indexes)} reference rows; expected one"
        )

    row_index = row_indexes[0]
    has_detail_table = any(
        "Demand, Resources, and Reserve Margins" in line for line in lines
    )
    if not has_detail_table:
        return None

    row_text = lines[row_index]
    continuation_index = row_index + 1
    while not PERCENT_TOKEN.search(row_text) and continuation_index < len(lines):
        row_text = f"{row_text} {lines[continuation_index]}"
        continuation_index += 1

    years = []
    for line in reversed(lines[max(0, row_index - 35) : row_index]):
        candidate_years = parse_years(line)
        if len(candidate_years) >= 2:
            years = candidate_years
            break
    values = parse_reference_row(row_text)
    if len(years) != len(values):
        raise ExtractionError(
            f"PDF page {page_number} has {len(years)} years and {len(values)} reference values"
        )

    region = infer_region(lines, page_number)
    return pd.DataFrame({"region": region, "planning_year": years, "value": values})


def parse_pdf(pdf_path: Path) -> pd.DataFrame:
    """Extract all detailed assessment-area reference margin tables."""
    reader = PdfReader(str(pdf_path))
    frames = []
    for page_number, page in enumerate(reader.pages, start=1):
        frame = parse_page(page.extract_text() or "", page_number)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise ExtractionError(
            "No detailed Reference Margin Level (%) tables were found"
        )
    result = pd.concat(frames, ignore_index=True)
    validate_output(result, target_year=None)
    return result.sort_values(["region", "planning_year"], ignore_index=True)


def extend_to_target(df: pd.DataFrame, target_year: int = TARGET_YEAR) -> pd.DataFrame:
    """Carry each region's last published value forward through target_year."""
    if target_year < int(df["planning_year"].max()):
        raise ValueError(
            "target_year cannot precede the latest published planning year"
        )
    additions = []
    for region, group in df.groupby("region", sort=False):
        last_year = int(group["planning_year"].max())
        last_value = float(group.loc[group["planning_year"].idxmax(), "value"])
        additions.extend(
            {"region": region, "planning_year": year, "value": last_value}
            for year in range(last_year + 1, target_year + 1)
        )
    if additions:
        df = pd.concat([df, pd.DataFrame(additions)], ignore_index=True)
    validate_output(df, target_year=target_year)
    return df.sort_values(["region", "planning_year"], ignore_index=True)


def validate_output(df: pd.DataFrame, target_year: int | None = TARGET_YEAR) -> None:
    """Validate the tidy output and optionally require a terminal year."""
    required = ["region", "planning_year", "value"]
    if list(df.columns) != required:
        raise ValueError(f"Expected columns {required}, found {list(df.columns)}")
    if df[required].isna().any().any():
        raise ValueError("Output contains missing region, planning_year, or value")
    if df.duplicated(["region", "planning_year"]).any():
        raise ValueError("Output contains duplicate (region, planning_year) pairs")
    if not pd.api.types.is_integer_dtype(df["planning_year"]):
        raise ValueError("planning_year must contain integers")
    if not df["value"].between(0, 1, inclusive="both").all():
        raise ValueError("value must contain decimal fractions between 0 and 1")
    if target_year is not None:
        terminal_years = df.groupby("region")["planning_year"].max()
        if not (terminal_years == target_year).all():
            raise ValueError(f"Every region must extend through {target_year}")


def resolve_pdf_url(assessment_url: str = DEFAULT_ASSESSMENTS_URL) -> tuple[str, int]:
    """Find the newest NERC LTRA PDF linked from the assessment index."""
    response = requests.get(assessment_url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    candidates = []
    for href in re.findall(
        r"(?:href|src)=[\"']([^\"']+\.pdf(?:\?[^\"']*)?)[\"']", response.text, re.I
    ):
        url = urljoin(response.url, href)
        years = [int(year) for year in re.findall(r"20\d{2}", url)]
        if years and "ltra" in url.lower():
            candidates.append((max(years), url))
    if not candidates:
        raise ExtractionError(f"No LTRA PDF links found at {assessment_url}")
    return max(candidates, key=lambda item: item[0])


def download_pdf(url: str, cache_dir: Path) -> Path:
    """Download a PDF once and return its cached path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = re.sub(r"[^A-Za-z0-9_.-]", "_", url.rsplit("/", 1)[-1].split("?", 1)[0])
    path = cache_dir / filename
    if not path.exists():
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def write_outputs(
    df: pd.DataFrame,
    output_path: Path,
    vintage_path: Path,
    source_url: str,
    report_year: int,
) -> None:
    """Write the tidy CSV and source-vintage metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vintage_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    vintage = {
        "report_year": report_year,
        "source_url": source_url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "value_units": "decimal_fraction",
        "source_metric": "Reference Margin Level (%)",
        "assessment_area_count": int(df["region"].nunique()),
        "planning_year_start": int(df["planning_year"].min()),
        "planning_year_end": int(df["planning_year"].max()),
    }
    vintage_path.write_text(json.dumps(vintage, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-url", help="Explicit report PDF URL")
    parser.add_argument(
        "--report-year", type=int, help="Report year when using --pdf-url"
    )
    parser.add_argument("--assessment-url", default=DEFAULT_ASSESSMENTS_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vintage-output", type=Path, default=DEFAULT_VINTAGE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--target-year", type=int, default=TARGET_YEAR)
    args = parser.parse_args()

    if args.pdf_url:
        source_url = args.pdf_url
        report_year = args.report_year or max(
            int(year) for year in re.findall(r"20\d{2}", source_url)
        )
    else:
        source_url, report_year = resolve_pdf_url(args.assessment_url)
    pdf_path = download_pdf(source_url, args.cache_dir)
    extracted = extend_to_target(parse_pdf(pdf_path), args.target_year)
    write_outputs(extracted, args.output, args.vintage_output, source_url, report_year)
    print(
        f"Saved {len(extracted)} rows for {extracted['region'].nunique()} regions to {args.output}"
    )


if __name__ == "__main__":
    main()
