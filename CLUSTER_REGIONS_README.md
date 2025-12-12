# cluster_regions.py - ReEDS Region Clustering Tool

Cluster ReEDS regions based on transmission capacity using spectral and hierarchical clustering while respecting regional group constraints.

## Overview

This script uses graph-based clustering to group ReEDS balancing authorities (BAs) based on transmission network capacity, creating balanced clusters that maintain regional groupings from a hierarchy file.

## Installation

### Requirements

- Python 3.8+
- numpy
- pandas
- networkx
- scipy
- scikit-learn
- pyyaml

### Setup

Install dependencies using uv:
```bash
uv pip install networkx scipy scikit-learn pandas pyyaml numpy
```

Or using pip directly:
```bash
pip install networkx scipy scikit-learn pandas pyyaml numpy
```

## Usage

### Basic Syntax

```bash
./cluster_regions.py <hierarchy_file> <transmission_file> <target_regions> <grouping_column> [options]
```

### Required Arguments

- **hierarchy_file**: Path to hierarchy.csv (typically `cache_data/hierarchy.csv`)
- **transmission_file**: Path to transmission capacity file (typically `data/transmission_capacity_reeds.csv`)
- **target_regions**: Target number of clusters (integer)
- **grouping_column**: Hierarchy column for regional grouping (see Available Columns below)

### Optional Arguments

- `--include_groups GROUP1 GROUP2 ...`: Whitelist filter—only include BAs from these regional groups. If not specified, all groups are included.
- `--no_cluster GROUP1 GROUP2 ...`: Regional groups that should NOT be clustered together. Each group in this list will remain as separate clusters. Only applies to groups within `--include_groups` if specified.
- `--output FILE`: Output file path (default: `region_aggregations.yml`)
- `--validate`: Run hierarchical clustering for validation/comparison
- `--verbose`: Show detailed progress information

## Available Grouping Columns

The hierarchy file includes the following columns that can be used for `grouping_column`:

| Column | Description | Typical Use |
|--------|-------------|------------|
| **ba** | Balancing Authority identifier | The BA code |
| **nercr** | NERC Region | 6 main reliability regions (WECC_NW, WECC_SW, MRO, SERC, RFC, TRE) |
| **transreg** | Transmission Region | Larger transmission operator groups |
| **transgrp** | Transmission Group | Mid-level transmission groupings |
| **cendiv** | Census Division | US Census divisions (9 divisions) |
| **st** | State | US state abbreviations |
| **interconnect** | Interconnection | eastern, western, ercot |
| **st_interconnect** | State-Interconnect | State + Interconnect combination |
| **country** | Country | USA |
| **usda_region** | USDA Region | Agricultural/climate regions |
| **h2ptcreg** | Hydrogen PTC Region | Hydrogen production tax credit regions |
| **hurdlereg** | Hurdle Region | Hurdle rate groupings |
| **aggreg** | Aggregation | Standard aggregation identifier |

## Examples

### Cluster all regions to 10 clusters by NERC Region

```bash
./cluster_regions.py cache_data/hierarchy.csv data/transmission_capacity_reeds.csv 10 nercr
```

### Cluster only WECC regions to 4 clusters

```bash
./cluster_regions.py cache_data/hierarchy.csv data/transmission_capacity_reeds.csv 4 nercr \
  --include_groups WECC_NW WECC_CA WECC_SW
```

### Cluster by transmission group with validation

```bash
./cluster_regions.py cache_data/hierarchy.csv data/transmission_capacity_reeds.csv 20 transgrp \
  --validate --output my_clusters.yml
```

### Cluster by state but keep certain NERC regions unclustered

```bash
./cluster_regions.py cache_data/hierarchy.csv data/transmission_capacity_reeds.csv 13 st \
  --no_cluster WECC_NW ERCOT --verbose
```

### Cluster Eastern regions only, preventing certain groups from merging

```bash
./cluster_regions.py cache_data/hierarchy.csv data/transmission_capacity_reeds.csv 8 transreg \
  --include_groups PJM NYISO ISONE SERTP \
  --no_cluster NYISO ISONE
```

## Output Format

The script outputs a YAML file with clustering results:

```yaml
region_aggregations:
  WECC_NW:
    - p1
    - p2
    - p3
  WECC_CA+WECC_SW:
    - p8
    - p9
    - p27
  # ... additional clusters
```

Cluster names are generated from regional groups:
- If a cluster contains BAs from a single group: `GROUP_NAME`
- If a cluster contains BAs from multiple groups: `GROUP1+GROUP2+...` (alphabetically sorted)
- If multiple clusters share the same group composition: `GROUP_1`, `GROUP_2`, etc.

If `--validate` is used and hierarchical clustering differs from spectral clustering, the output includes both:

```yaml
region_aggregations:
  # Spectral clustering results (primary)
  ...
hierarchical_clustering:
  # Hierarchical clustering results (for comparison)
  ...
```

## How It Works

### --include_groups (Whitelist Filter)

The `--include_groups` argument acts as a **whitelist**. Only BAs from the specified groups are included in clustering. All other BAs are excluded.

**Example:**
```bash
# Cluster only WECC regions
./cluster_regions.py ... nercr --include_groups WECC_NW WECC_CA WECC_SW
```

### --no_cluster (Prevent Merging)

The `--no_cluster` argument prevents specific regional groups from being merged together. Each group marked with `--no_cluster` will remain as a separate cluster.

**Example:**
```bash
# Cluster all regions by state, but keep NYISO and ISONE as single clusters
./cluster_regions.py ... st --no_cluster NYISO ISONE
```

### Interaction of --include_groups and --no_cluster

1. **--include_groups filters first** (whitelist): Only BAs from specified groups are kept
2. **--no_cluster prevents merging** (within the filtered groups): Specified groups won't be combined with others

**Example:**
```bash
./cluster_regions.py ... nercr \
  --include_groups WECC_NW WECC_CA WECC_SW PJM MISO \
  --no_cluster WECC_CA
```
Result: 
- Only WECC and PJM/MISO BAs are included
- WECC_CA stays as its own cluster
- WECC_NW and WECC_SW can be merged
- PJM and MISO can be merged

### Merging Groups When target < num_groups

If `target_regions` is less than the number of regional groups, the script automatically merges groups with the **weakest inter-group transmission connections** first. This preserves strong transmission ties within clusters.

**Example:**
```bash
# 15 unique NERC regions, but target only 8 clusters
./cluster_regions.py ... nercr 8
# The 7 weakest inter-group connections will be severed
```

## Output Interpretation

The clustering summary shows:

```
Clustering Summary:
----------------------------------------------------------------------
  Cluster_Name                  : NNN BAs  Internal: XXXXXXXXX MW  Cut: XXXXXXXXX MW
  ...
----------------------------------------------------------------------
Average cluster size: XX.X ± XX.X BAs
Total internal capacity: XXXXXXXXX MW
Total cut capacity: XXXXXXXXX MW
```

- **Cluster Name**: Generated from regional groups (e.g., `WECC_NW`, `PJM+MISO`)
- **BAs**: Number of balancing authorities in the cluster
- **Internal**: Total transmission capacity within the cluster (preserved)
- **Cut**: Transmission capacity between clusters (ties severed)
- **Average size ± std dev**: Cluster balance metric
- **Total internal capacity**: Sum across all clusters
- **Total cut capacity**: Cross-cluster transmission requirements

Lower cut capacity is better (fewer ties severed).

## Clustering Algorithms

### Spectral Clustering (Primary)
- Uses normalized Laplacian matrix from transmission network
- Applies KMeans to cluster graph Laplacian eigenvectors
- Effective for balancing cluster sizes and minimizing cuts
- Output: Primary `region_aggregations` in YAML

### Hierarchical Clustering (Validation)
- Uses Ward's linkage method on transmission distance matrix
- Distance metric: inverse of transmission capacity
- Useful for validation and comparison
- Output: `hierarchical_clustering` in YAML if different from spectral result

Use `--validate` to run both algorithms and compare results.

## Common Issues

### Column Not Found Error
Ensure `grouping_column` matches exactly with a column in hierarchy.csv. Check available columns with `--verbose` mode.

**Example error:**
```
Error: Column 'nercr_region' not found in hierarchy. Available columns: ba, nercr, transreg, ...
```

**Fix:** Use `nercr` instead of `nercr_region`

### Missing BAs in Hierarchy
Some BAs in transmission file may not exist in hierarchy. This is typically a warning and can usually be ignored.

### Cluster Size Imbalance
For highly interconnected networks, consider:
- Using different `grouping_column` values
- Adjusting `target_regions` to better match network structure
- Using `--validate` to compare spectral and hierarchical results

### Isolated BAs
The script automatically removes BAs with no transmission connections. These BAs can be manually connected or excluded from analysis.

## Integration with PowerGenome

To use the output in PowerGenome:

1. Generate clusters:
```bash
./cluster_regions.py cache_data/hierarchy.csv data/transmission_capacity_reeds.csv 15 transgrp \
  --output my_regions.yml
```

2. Reference in PowerGenome settings YAML:
```yaml
# In your EI_PJM/settings/model_definition.yml or similar

region_aggregations: !include my_regions.yml

# Or copy the aggregations directly
region_aggregations:
  WECC_NW:
    - p1
    - p2
    # ...
```

## Performance Notes

- Processing time depends on number of BAs and interconnections
- Typical execution: < 5 seconds for ReEDS dataset (134 BAs, ~300 edges)
- Use `--validate` for confidence in results (adds ~1-2 seconds)
- Memory usage: ~100 MB for full ReEDS dataset

## Troubleshooting

### Graph is not fully connected
Some BAs may be isolated (no transmission connections). The script automatically removes them. Check if these should be manually connected.

### Clustering takes too long
- Reduce `target_regions` or use simpler `grouping_column` like `nercr` instead of `ba`
- The script should complete in <10 seconds even for full dataset

### Results differ from expected
Use `--validate --verbose` to:
1. Compare spectral and hierarchical clustering
2. See detailed merging decisions
3. Understand group composition

### Output file format issues
Ensure PyYAML is installed correctly:
```bash
uv pip install pyyaml --force-reinstall
```

## Advanced Usage

### Generate multiple clustering schemes

```bash
# By NERC region
./cluster_regions.py ... nercr --output nerc_clusters.yml

# By transmission region
./cluster_regions.py ... transreg --output transreg_clusters.yml

# By state
./cluster_regions.py ... st --output state_clusters.yml
```

### Selective clustering

```bash
# Cluster Western regions only, keeping state-level separation
./cluster_regions.py ... st \
  --include_groups WA OR CA NV ID MT WY UT AZ NM CO \
  --output western_state_clusters.yml

# Cluster Eastern PJM with specific rules
./cluster_regions.py ... transgrp \
  --include_groups PJM_East \
  --no_cluster NYISO \
  --output pjm_east_clusters.yml
```

## References

- **Spectral Clustering**: Uses normalized Laplacian; see Shi & Malik (2000), Von Luxburg (2007)
- **Ward's Linkage**: Hierarchical clustering minimizing within-cluster variance
- **Normalized Cut**: Minimizes cut capacity while balancing cluster sizes
