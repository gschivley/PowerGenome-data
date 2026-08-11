---
name: nerc-reserve-margins
description: "Extract NERC Long-Term Reliability Assessment Reference Margin Level (%) values by assessment area from the latest PDF and write nerc_reserve_margins.csv through 2050. Use when updating NERC reserve margins, reading LTRA reports, or creating a tidy reserve-margin reference dataset."
argument-hint: "Optional report PDF URL or report year; defaults to the newest NERC LTRA"
---

# NERC Reserve Margins

## What This Skill Does

Downloads the newest NERC Long-Term Reliability Assessment (LTRA), extracts the exact
`Reference Margin Level (%)` row from each detailed assessment-area table, converts percentages
to decimal fractions, and writes a tidy CSV with `region`, `planning_year`, and `value`.
Each region's last published value is carried forward through 2050. The exact report URL and
report year are written to a JSON vintage sidecar.

This skill is independent of the ReEDS BA-level `data/reserve_margins.csv` workflow. It keeps
NERC assessment-area names and does not map them to balancing areas or model regions.

## Procedure

1. Run the bundled extractor from the repository root:

   ```bash
   uv run python .github/skills/nerc-reserve-margins/scripts/extract_nerc_reserve_margins.py
   ```

   By default it discovers the newest PDF linked from the NERC long-term assessments page,
   caches it under `cache_data/nerc_ltra/`, and writes:

   - `data/nerc_reserve_margins.csv`
   - `data/nerc_reserve_margins_vintage.json`

2. For a reproducible historical extraction, provide the exact PDF URL:

   ```bash
   uv run python .github/skills/nerc-reserve-margins/scripts/extract_nerc_reserve_margins.py \
     --pdf-url https://www.nerc.com/globalassets/our-work/assessments/nerc_ltra_2025.pdf \
     --report-year 2025
   ```

3. Use `--output`, `--vintage-output`, `--cache-dir`, and `--target-year` to override paths or
   the terminal year. The report's annual table headers are used as planning years. For a
   winter table with headers such as `2026-2027`, the first year (`2026`) is used as the
   planning year for that value.

## Output Contract

The CSV has exactly these columns:

```text
region,planning_year,value
```

`value` is a decimal fraction: a report value of `17.7%` is written as `0.177`. Values are
sorted by region and planning year, have unique `(region, planning_year)` keys, and extend from
each region's last published year through 2050 using that region's final published value.

The vintage JSON records `report_year`, `source_url`, retrieval time, units, source metric,
assessment-area count, and output year bounds.

## Verification

Check the generated file with pandas:

```bash
uv run python -c "import pandas as pd; d=pd.read_csv('data/nerc_reserve_margins.csv'); print(d.head()); print(d.groupby('region').planning_year.agg(['min','max','count']))"
```

Confirm that:

- the columns are exactly `region`, `planning_year`, and `value`;
- all values are numeric decimal fractions in `[0, 1]`;
- `(region, planning_year)` is unique;
- every region reaches the requested target year;
- the vintage JSON identifies the exact PDF used; and
- the MISO 2030 value agrees with the corresponding report table.

The extractor fails rather than producing partial data when it cannot identify a region,
year header, or unambiguous reference row. It does not substitute anticipated reserve margin,
prospective reserve margin, planning reserve margin, or summary-table ranges.

## Troubleshooting

**No LTRA PDF links found** — use `--pdf-url` with the direct PDF URL. The default 2025 source is
`https://www.nerc.com/globalassets/our-work/assessments/nerc_ltra_2025.pdf`.

**Year/value count mismatch** — inspect the affected PDF page. NERC may have changed the table
layout or introduced a non-annual/multi-season table that needs explicit parser support.

**Cached report is stale** — remove the relevant file under `cache_data/nerc_ltra/` and rerun.

**Output validation fails** — inspect the page text and confirm that the exact
`Reference Margin Level (%)` row, rather than another reserve-margin metric, was extracted.
