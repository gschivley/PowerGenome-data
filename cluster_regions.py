#!/usr/bin/env python3
"""
Cluster ReEDS regions based on transmission capacity.

This script uses spectral and hierarchical clustering to group regions,
respecting regional group constraints from a hierarchy file.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import yaml
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering


def load_and_validate_data(
    hierarchy_file: str, transmission_file: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate hierarchy and transmission files."""
    try:
        hierarchy = pd.read_csv(hierarchy_file)
        transmission = pd.read_csv(transmission_file)
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"Error: Failed to parse CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    return hierarchy, transmission


def filter_transmission_to_hierarchy(
    transmission: pd.DataFrame, hierarchy_bas: Set[str]
) -> pd.DataFrame:
    """Keep only transmission rows where both endpoints are in the hierarchy list."""
    mask = transmission["region_from"].isin(hierarchy_bas) & transmission["region_to"].isin(hierarchy_bas)
    filtered = transmission.loc[mask].copy()
    return filtered


def validate_columns(
    hierarchy: pd.DataFrame,
    transmission: pd.DataFrame,
    grouping_column: str,
) -> None:
    """Validate that required columns exist."""
    if grouping_column not in hierarchy.columns:
        print(
            f"Error: Column '{grouping_column}' not found in hierarchy. "
            f"Available columns: {', '.join(hierarchy.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    required_transmission_cols = {"region_from", "region_to", "firm_ttc_mw"}
    missing_cols = required_transmission_cols - set(transmission.columns)
    if missing_cols:
        print(
            f"Error: Missing columns in transmission file: {missing_cols}",
            file=sys.stderr,
        )
        sys.exit(1)


def validate_bas(hierarchy: pd.DataFrame, transmission: pd.DataFrame) -> None:
    """Validate that all BAs in transmission file exist in hierarchy."""
    hierarchy_bas = set(hierarchy["ba"].unique())
    transmission_bas = set(
        pd.concat([transmission["region_from"], transmission["region_to"]]).unique()
    )

    missing_bas = transmission_bas - hierarchy_bas
    if missing_bas:
        print(
            f"Warning: {len(missing_bas)} BAs in transmission file "
            f"not found in hierarchy: {sorted(missing_bas)[:10]}...",
            file=sys.stderr,
        )


def build_transmission_graph(transmission: pd.DataFrame) -> nx.Graph:
    """Build undirected weighted graph from transmission data."""
    G = nx.Graph()

    for _, row in transmission.iterrows():
        p1, p2 = row["region_from"], row["region_to"]
        capacity = row["firm_ttc_mw"]

        if G.has_edge(p1, p2):
            G[p1][p2]["weight"] += capacity
        else:
            G.add_edge(p1, p2, weight=capacity)

    return G


def get_regional_groups(
    hierarchy: pd.DataFrame, grouping_column: str, graph: nx.Graph
) -> Dict[str, Set[str]]:
    """Map regional groups to their BAs from the graph."""
    groups = {}
    ba_to_group = dict(zip(hierarchy["ba"], hierarchy[grouping_column]))

    for ba in graph.nodes():
        if ba in ba_to_group:
            group = ba_to_group[ba]
            if group not in groups:
                groups[group] = set()
            groups[group].add(ba)

    return groups


def filter_to_included_groups(
    graph: nx.Graph,
    hierarchy: pd.DataFrame,
    grouping_column: str,
    include_groups: List[str],
    verbose: bool = False,
) -> Tuple[nx.Graph, Dict[str, Set[str]]]:
    """Filter the graph to only include BAs from specified regional groups.
    
    Parameters
    ----------
    graph : nx.Graph
        Original BA connectivity graph
    hierarchy : pd.DataFrame
        Hierarchy data containing BA and group information
    grouping_column : str
        Column name for regional groups
    include_groups : List[str]
        List of group names to include
    verbose : bool
        Whether to print verbose output
        
    Returns
    -------
    tuple
        Filtered graph and regional groups mapping
    """
    ba_to_group = dict(zip(hierarchy["ba"], hierarchy[grouping_column]))
    
    # Find BAs in included groups
    included_bas = set()
    for ba, group in ba_to_group.items():
        if group in include_groups:
            included_bas.add(ba)
    
    # Find excluded BAs
    all_bas = set(graph.nodes())
    excluded_bas = all_bas - included_bas
    
    if excluded_bas:
        print(
            f"Filtering to {len(include_groups)} regional group(s): {', '.join(sorted(include_groups))}",
            file=sys.stderr,
        )
        print(
            f"Excluding {len(excluded_bas)} BA(s) from other groups",
            file=sys.stderr,
        )
        if verbose:
            excluded_by_group = {}
            for ba in excluded_bas:
                group = ba_to_group.get(ba, "unknown")
                if group not in excluded_by_group:
                    excluded_by_group[group] = []
                excluded_by_group[group].append(ba)
            for group in sorted(excluded_by_group.keys()):
                print(
                    f"  {group}: {', '.join(sorted(excluded_by_group[group]))}",
                    file=sys.stderr,
                )
    
    # Create filtered graph
    filtered_graph = graph.subgraph(included_bas).copy()
    
    # Get groups from filtered graph
    filtered_groups = get_regional_groups(hierarchy, grouping_column, filtered_graph)
    
    if verbose:
        print(
            f"Filtered graph: {len(filtered_graph.nodes())} nodes, "
            f"{len(filtered_graph.edges())} edges",
            file=sys.stderr,
        )
    
    return filtered_graph, filtered_groups


def merge_weak_groups(
    groups: Dict[str, Set[str]],
    graph: nx.Graph,
    num_groups_target: int,
) -> Dict[str, Set[str]]:
    """Merge regional groups with weakest inter-group connections."""
    if len(groups) <= num_groups_target:
        return groups

    groups_merged = {name: bas.copy() for name, bas in groups.items()}
    group_names_list = list(groups_merged.keys())

    while len(groups_merged) > num_groups_target:
        min_capacity = float("inf")
        merge_group1, merge_group2 = None, None

        for i, g1 in enumerate(group_names_list):
            for g2 in group_names_list[i + 1 :]:
                capacity = sum(
                    graph[n1][n2]["weight"]
                    for n1 in groups_merged[g1]
                    for n2 in groups_merged[g2]
                    if graph.has_edge(n1, n2)
                )
                if capacity < min_capacity:
                    min_capacity = capacity
                    merge_group1, merge_group2 = g1, g2

        if merge_group1 is None:
            break

        new_name = "+".join(sorted([merge_group1, merge_group2]))
        groups_merged[new_name] = (
            groups_merged[merge_group1] | groups_merged[merge_group2]
        )
        del groups_merged[merge_group1]
        del groups_merged[merge_group2]
        group_names_list = list(groups_merged.keys())

    return groups_merged


def spectral_cluster(
    graph: nx.Graph, n_clusters: int, verbose: bool = False
) -> Dict[int, Set[str]]:
    """Perform spectral clustering using normalized Laplacian."""
    nodes = sorted(graph.nodes())
    
    # Use the affinity matrix directly instead of precomputed Laplacian
    # sklearn's SpectralClustering will compute the Laplacian internally
    A = nx.to_numpy_array(graph, nodelist=nodes, weight="weight")
    
    # Check for isolated nodes or zero-degree nodes
    degrees = A.sum(axis=1)
    if np.any(degrees == 0):
        # Filter out isolated nodes
        connected_mask = degrees > 0
        connected_indices = np.where(connected_mask)[0]
        connected_nodes = [nodes[i] for i in connected_indices]
        A_connected = A[np.ix_(connected_indices, connected_indices)]
        
        if verbose:
            isolated_nodes = [nodes[i] for i in range(len(nodes)) if not connected_mask[i]]
            print(f"Warning: Found {len(isolated_nodes)} isolated nodes, clustering only connected nodes", file=sys.stderr)
        
        # Adjust number of clusters if needed
        effective_n_clusters = min(n_clusters, len(connected_nodes))
        
        clusterer = SpectralClustering(
            n_clusters=effective_n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
        )
        labels = clusterer.fit_predict(A_connected)
        
        # Map back to original nodes
        clusters = {}
        for label in set(labels):
            mask = labels == label
            cluster_nodes = [connected_nodes[i] for i in range(len(connected_nodes)) if mask[i]]
            clusters[label] = set(cluster_nodes)
        
        # Add isolated nodes as separate clusters
        next_label = max(clusters.keys()) + 1 if clusters else 0
        for node in isolated_nodes:
            clusters[next_label] = {node}
            next_label += 1
    else:
        clusterer = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
        )
        labels = clusterer.fit_predict(A)
        
        clusters = {}
        for label in set(labels):
            mask = labels == label
            cluster_nodes = [nodes[i] for i in range(len(nodes)) if mask[i]]
            clusters[label] = set(cluster_nodes)

    return clusters


def hierarchical_cluster(
    graph: nx.Graph, n_clusters: int, verbose: bool = False
) -> Dict[int, Set[str]]:
    """Perform hierarchical clustering using Ward's method."""
    nodes = sorted(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    distances = np.zeros((n, n))
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            if i < j:
                if graph.has_edge(n1, n2):
                    capacity = graph[n1][n2]["weight"]
                    distances[i, j] = 1.0 / (capacity + 1e-6)
                else:
                    distances[i, j] = 1.0 / 1e-6
                distances[j, i] = distances[i, j]

    dist_vector = squareform(distances)
    Z = linkage(dist_vector, method="ward")

    labels = fcluster(Z, n_clusters, criterion="maxclust") - 1

    clusters = {}
    for label in set(labels):
        mask = labels == label
        cluster_nodes = [nodes[i] for i in range(len(nodes)) if mask[i]]
        clusters[label] = set(cluster_nodes)

    return clusters


def compare_clusterings(
    spectral: Dict[int, Set[str]],
    hierarchical: Dict[int, Set[str]],
) -> bool:
    """Check if two clustering results are identical."""
    if len(spectral) != len(hierarchical):
        return False

    spectral_sets = sorted([frozenset(c) for c in spectral.values()])
    hier_sets = sorted([frozenset(c) for c in hierarchical.values()])

    return spectral_sets == hier_sets


def calculate_quality_metrics(
    clusters: Dict[int, Set[str]], graph: nx.Graph
) -> Dict[int, Dict[str, float]]:
    """Calculate quality metrics for each cluster."""
    metrics = {}

    for label, nodes in clusters.items():
        nodes_list = list(nodes)

        internal_capacity = sum(
            graph[n1][n2]["weight"]
            for n1 in nodes_list
            for n2 in nodes_list
            if n1 < n2 and graph.has_edge(n1, n2)
        )

        cut_capacity = sum(
            graph[n1][n2]["weight"]
            for n1 in nodes_list
            for n2 in graph.nodes()
            if n2 not in nodes and graph.has_edge(n1, n2)
        )

        total_degree = sum(
            graph.degree(n, weight="weight") for n in nodes_list
        )

        metrics[label] = {
            "internal_capacity": internal_capacity,
            "cut_capacity": cut_capacity,
            "size": len(nodes_list),
            "total_degree": total_degree,
        }

    return metrics


def generate_cluster_names(
    clusters: Dict[int, Set[str]],
    groups: Dict[str, Set[str]],
    graph: nx.Graph,
) -> Dict[int, str]:
    """Generate meaningful cluster names based on regional groups."""
    metrics = calculate_quality_metrics(clusters, graph)
    sorted_labels = sorted(metrics.keys(), key=lambda l: metrics[l]["total_degree"], reverse=True)

    cluster_names = {}
    group_counts = {}

    for label in sorted_labels:
        nodes = clusters[label]
        relevant_groups = []

        for group_name, group_nodes in groups.items():
            if nodes & group_nodes:
                relevant_groups.append(group_name)

        relevant_groups.sort()

        if len(relevant_groups) == 1:
            base_name = relevant_groups[0]
        else:
            base_name = "+".join(relevant_groups)

        if base_name not in group_counts:
            group_counts[base_name] = 0

        group_counts[base_name] += 1

        if group_counts[base_name] > 1:
            name = f"{base_name}_{group_counts[base_name]}"
        else:
            name = base_name

        cluster_names[label] = name

    return cluster_names


def clusters_to_dict(
    clusters: Dict[int, Set[str]], cluster_names: Dict[int, str]
) -> Dict[str, List[str]]:
    """Convert cluster labels to sorted BA lists with names."""
    result = {}
    for label, nodes in clusters.items():
        name = cluster_names[label]
        result[name] = sorted(list(nodes))
    return result


def filter_isolated_nodes(graph: nx.Graph) -> nx.Graph:
    """Remove isolated nodes from graph."""
    isolated = list(nx.isolates(graph))
    if isolated:
        print(f"Removing {len(isolated)} isolated nodes: {isolated[:5]}...", file=sys.stderr)
        graph.remove_nodes_from(isolated)
    return graph


def write_output(
    model_regions: List[str],
    spectral_result: Dict[str, List[str]],
    output_file: str,
) -> None:
    """Write clustering results to YAML file with model_regions and region_aggregations."""
    output = {
        "model_regions": model_regions,
        "region_aggregations": spectral_result,
    }

    with open(output_file, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)

    print(f"Results written to {output_file}")


def print_summary(
    clusters: Dict[int, Set[str]],
    cluster_names: Dict[int, str],
    graph: nx.Graph,
    verbose: bool = False,
) -> None:
    """Print summary of clustering results."""
    print("\nClustering Summary:")
    print("-" * 70)

    metrics = calculate_quality_metrics(clusters, graph)

    for label in sorted(metrics.keys(), key=lambda l: metrics[l]["total_degree"], reverse=True):
        name = cluster_names[label]
        size = metrics[label]["size"]
        internal = metrics[label]["internal_capacity"]
        cut = metrics[label]["cut_capacity"]
        degree = metrics[label]["total_degree"]

        print(f"  {name:35s}: {size:3d} BAs  "
              f"Internal: {internal:9.0f} MW  Cut: {cut:9.0f} MW")

    print("-" * 70)

    cluster_sizes = [metrics[l]["size"] for l in metrics.keys()]
    avg_size = np.mean(cluster_sizes)
    std_size = np.std(cluster_sizes)
    print(f"Average cluster size: {avg_size:.1f} ± {std_size:.1f} BAs")

    total_cut = sum(m["cut_capacity"] for m in metrics.values()) / 2
    total_capacity = sum(m["internal_capacity"] for m in metrics.values()) + total_cut
    print(f"Total internal capacity: {total_capacity - total_cut:.0f} MW")
    print(f"Total cut capacity: {total_cut:.0f} MW")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cluster ReEDS regions based on transmission capacity"
    )
    parser.add_argument("hierarchy_file", help="Path to hierarchy.csv")
    parser.add_argument(
        "transmission_file", help="Path to transmission_capacity_reeds.csv"
    )
    parser.add_argument(
        "target_regions", type=int, help="Target number of clusters"
    )
    parser.add_argument(
        "grouping_column",
        help="Column from hierarchy to use for regional groups "
        "(e.g., nercr, transreg, transgrp, cendiv, st)",
    )
    parser.add_argument(
        "--include_groups",
        nargs="+",
        default=None,
        help="Regional groups to include in clustering. "
        "If not specified, all groups are included. "
        "Only BAs from these groups will be clustered.",
    )
    parser.add_argument(
        "--no_cluster",
        nargs="+",
        default=[],
        help="Regional group values to keep unclustered (all BAs in these groups stay separate). "
        "Only applies to groups in --include_groups if specified.",
    )
    parser.add_argument(
        "--output",
        default="region_aggregations.yml",
        help="Output file path (default: region_aggregations.yml)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Also run hierarchical clustering for comparison",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show debug information"
    )

    args = parser.parse_args()

    if args.verbose:
        print("Loading data...")

    hierarchy, transmission = load_and_validate_data(
        args.hierarchy_file, args.transmission_file
    )

    if args.verbose:
        print(f"Hierarchy shape: {hierarchy.shape}")
        print(f"Transmission shape: {transmission.shape}")

    validate_columns(hierarchy, transmission, args.grouping_column)
    validate_bas(hierarchy, transmission)

    # Filter transmission to BAs present in hierarchy
    hierarchy_bas = set(hierarchy["ba"].unique())
    transmission = filter_transmission_to_hierarchy(transmission, hierarchy_bas)
    if args.verbose:
        print(f"Filtered transmission shape: {transmission.shape}")

    if args.verbose:
        print("Building transmission graph...")

    graph = build_transmission_graph(transmission)
    graph = filter_isolated_nodes(graph)

    if args.verbose:
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Filter to included groups if specified
    if args.include_groups:
        if args.verbose:
            print(f"Filtering to regional groups: {', '.join(args.include_groups)}")
        graph, groups = filter_to_included_groups(
            graph, hierarchy, args.grouping_column,
            args.include_groups, verbose=args.verbose
        )
        # Also trim hierarchy to included groups only
        hierarchy = hierarchy[hierarchy[args.grouping_column].isin(args.include_groups)].copy()
    else:
        groups = get_regional_groups(hierarchy, args.grouping_column, graph)

    # Identify BAs that must remain unclustered (no_cluster groups)
    bas_no_cluster: Set[str] = set()
    if args.no_cluster:
        group_set = set(args.no_cluster)
        bas_no_cluster = set(hierarchy.loc[hierarchy[args.grouping_column].isin(group_set), "ba"].tolist())
        if args.verbose:
            print(f"Keeping {len(bas_no_cluster)} BAs unclustered from groups: {', '.join(args.no_cluster)}")
        # Remove these nodes from the graph before clustering
        graph.remove_nodes_from(bas_no_cluster)

    # Recompute groups from the filtered graph (excluding no_cluster nodes)
    groups = get_regional_groups(hierarchy, args.grouping_column, graph)

    if args.verbose:
        print(f"Regional groups used for clustering: {len(groups)}")
        for name, bas in sorted(groups.items()):
            print(f"  {name}: {len(bas)} BAs")

    # Merge groups if target is smaller than group count
    if args.target_regions < len(groups):
        if args.verbose:
            print(
                f"Merging regional groups ({len(groups)} -> {args.target_regions})..."
            )
        groups = merge_weak_groups(groups, graph, args.target_regions)

    if args.verbose:
        print(f"After merging: {len(groups)} groups")

    # Check if graph is connected
    if not nx.is_connected(graph):
        n_components = nx.number_connected_components(graph)
        if args.verbose:
            print(f"Warning: Graph has {n_components} disconnected components", file=sys.stderr)
            for i, component in enumerate(nx.connected_components(graph), 1):
                print(f"  Component {i}: {len(component)} nodes", file=sys.stderr)
    
    # Determine required clusters after removing no_cluster BAs
    remaining_nodes = list(graph.nodes())
    needed_clusters = max(0, args.target_regions - len(bas_no_cluster))

    if len(bas_no_cluster) > args.target_regions:
        print(
            f"Warning: --no_cluster BAs ({len(bas_no_cluster)}) exceed target_regions ({args.target_regions}); truncating to first {args.target_regions}.",
            file=sys.stderr,
        )
        bas_no_cluster = set(sorted(bas_no_cluster)[: args.target_regions])
        needed_clusters = 0

    if needed_clusters == 0 or len(remaining_nodes) == 0:
        spectral_result = {}
        spectral_clusters = {}
        spectral_names = {}
    else:
        effective_clusters = min(needed_clusters, len(remaining_nodes))
        if effective_clusters < needed_clusters:
            print(
                f"Warning: Only {len(remaining_nodes)} nodes available; using {effective_clusters} clusters instead of requested {needed_clusters}.",
                file=sys.stderr,
            )
        if args.verbose:
            print(f"Running spectral clustering with {effective_clusters} clusters on {len(remaining_nodes)} nodes...")

        spectral_clusters = spectral_cluster(graph, effective_clusters, args.verbose)
        spectral_names = generate_cluster_names(spectral_clusters, groups, graph)
        spectral_result = clusters_to_dict(spectral_clusters, spectral_names)

    # Build model_regions: unclustered BAs + clustered region names
    model_regions_list = sorted(bas_no_cluster) + sorted(spectral_result.keys())
    if len(model_regions_list) != args.target_regions:
        print(
            f"Warning: model_regions count {len(model_regions_list)} differs from target_regions {args.target_regions}.",
            file=sys.stderr,
        )

    write_output(model_regions_list, spectral_result, args.output)
    print_summary(spectral_clusters, spectral_names, graph, args.verbose)

    if args.verbose:
        print(f"\nOutput written to {args.output}")


if __name__ == "__main__":
    main()
