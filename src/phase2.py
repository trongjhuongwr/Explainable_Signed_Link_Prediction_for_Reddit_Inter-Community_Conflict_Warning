"""Phase 2 graph construction and feature engineering helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


EDGE_COLUMNS = [
    "source_subreddit",
    "target_subreddit",
    "post_id",
    "timestamp",
    "link_sentiment",
    "properties",
]

OPTIONAL_EDGE_COLUMNS = ["dataset_source"]
PROPERTY_FEATURE_COUNT = 86
TEXT_FEATURE_PREFIX = "text_property_"


def load_phase1_filtered(path: str | Path) -> pd.DataFrame:
    """Load the cleaned phase 1 dataset used for graph construction."""
    header = pd.read_csv(path, nrows=0)
    available_columns = [column for column in EDGE_COLUMNS + OPTIONAL_EDGE_COLUMNS if column in header.columns]
    frame = pd.read_csv(path, usecols=available_columns)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["link_sentiment"] = pd.to_numeric(frame["link_sentiment"], errors="coerce").astype("Int64")
    frame["source_subreddit"] = frame["source_subreddit"].astype("string").str.strip().str.lower()
    frame["target_subreddit"] = frame["target_subreddit"].astype("string").str.strip().str.lower()
    frame["post_id"] = frame["post_id"].astype("string").str.strip()
    frame["properties"] = frame["properties"].astype("string")
    if "dataset_source" in frame.columns:
        frame["dataset_source"] = frame["dataset_source"].astype("string").str.strip().str.lower()
    return frame.dropna(subset=["source_subreddit", "target_subreddit", "timestamp", "link_sentiment"]).reset_index(drop=True)


def build_signed_multidigraph(frame: pd.DataFrame) -> nx.MultiDiGraph:
    """Build a signed directed multigraph from the cleaned hyperlink frame."""
    graph = nx.MultiDiGraph()
    for row in frame.itertuples(index=False):
        graph.add_edge(
            row.source_subreddit,
            row.target_subreddit,
            post_id=row.post_id,
            timestamp=row.timestamp,
            sentiment=int(row.link_sentiment),
            properties=row.properties,
        )
    return graph


def aggregate_edge_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated interactions into one row per ordered pair."""
    grouped = frame.groupby(["source_subreddit", "target_subreddit"], dropna=False)
    table = grouped.agg(
        interaction_count=("link_sentiment", "size"),
        positive_count=("link_sentiment", lambda values: int((values == 1).sum())),
        negative_count=("link_sentiment", lambda values: int((values == -1).sum())),
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
    ).reset_index()
    table["sentiment_balance"] = table["positive_count"] - table["negative_count"]
    table["negative_ratio"] = table["negative_count"] / table["interaction_count"].clip(lower=1)
    return table


def property_feature_names(prefix: str = TEXT_FEATURE_PREFIX) -> list[str]:
    """Return stable column names for the 86 SNAP text-property features."""
    return [f"{prefix}{index:02d}" for index in range(PROPERTY_FEATURE_COUNT)]


def parse_property_matrix(properties: pd.Series, prefix: str = TEXT_FEATURE_PREFIX) -> pd.DataFrame:
    """Parse the comma-separated SNAP text properties into numeric columns.

    The raw dataset stores 86 precomputed text features as a comma-separated
    string. Keeping these as numeric columns lets the report compare text-only,
    graph-only, and hybrid models without reprocessing raw post text.
    """
    names = property_feature_names(prefix)
    if properties.empty:
        return pd.DataFrame(columns=names, index=properties.index)

    rows = []
    for value in properties.fillna("").astype(str):
        parsed = np.fromstring(value, sep=",", dtype=np.float64)
        if parsed.size != PROPERTY_FEATURE_COUNT:
            fixed = np.full(PROPERTY_FEATURE_COUNT, np.nan, dtype=np.float64)
            fixed[: min(parsed.size, PROPERTY_FEATURE_COUNT)] = parsed[:PROPERTY_FEATURE_COUNT]
            parsed = fixed
        rows.append(parsed)

    return pd.DataFrame(np.vstack(rows), columns=names, index=properties.index)


def build_text_feature_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate historical post text properties into pair-level features."""
    base_columns = ["source_subreddit", "target_subreddit"]
    if frame.empty:
        return pd.DataFrame(columns=base_columns + ["text_feature_count"] + property_feature_names())

    text_features = parse_property_matrix(frame["properties"])
    feature_frame = pd.concat([frame[base_columns].reset_index(drop=True), text_features.reset_index(drop=True)], axis=1)
    grouped = feature_frame.groupby(base_columns, dropna=False)
    table = grouped[property_feature_names()].mean().reset_index()
    table["text_feature_count"] = grouped.size().to_numpy()

    if "dataset_source" in frame.columns:
        source_dummies = pd.get_dummies(frame["dataset_source"].fillna("unknown"), prefix="link_location")
        source_frame = pd.concat([frame[base_columns].reset_index(drop=True), source_dummies.reset_index(drop=True)], axis=1)
        source_table = source_frame.groupby(base_columns, dropna=False).mean(numeric_only=True).reset_index()
        table = table.merge(source_table, on=base_columns, how="left")

    return table


def build_node_feature_table(graph: nx.MultiDiGraph) -> pd.DataFrame:
    """Compute node-level structural features for the signed network."""
    simple_graph = nx.DiGraph()
    for source, target, data in graph.edges(data=True):
        sentiment = int(data.get("sentiment", 1))
        sign_weight = 1 if sentiment > 0 else -1
        if simple_graph.has_edge(source, target):
            simple_graph[source][target]["weight"] += 1
            simple_graph[source][target]["signed_weight"] += sign_weight
        else:
            simple_graph.add_edge(source, target, weight=1, signed_weight=sign_weight)

    pagerank_scores = nx.pagerank(simple_graph, alpha=0.85, weight="weight") if simple_graph.number_of_edges() else {}
    betweenness_scores = (
        nx.betweenness_centrality(simple_graph, normalized=True, endpoints=False, k=min(256, simple_graph.number_of_nodes()), seed=42)
        if simple_graph.number_of_nodes() > 1
        else {}
    )
    reciprocity_scores = nx.reciprocity(simple_graph, nodes=list(simple_graph.nodes())) if simple_graph.number_of_edges() else {}
    if isinstance(reciprocity_scores, float):
        reciprocity_scores = {node: reciprocity_scores for node in simple_graph.nodes()}

    undirected_graph = simple_graph.to_undirected()
    clustering_scores = nx.clustering(undirected_graph, weight="weight") if undirected_graph.number_of_edges() else {}
    community_map = _detect_communities(undirected_graph)
    community_sizes = Counter(community_map.values())
    community_stats = _build_community_sentiment_stats(graph, community_map)

    rows = []
    for node in simple_graph.nodes():
        in_edges = list(simple_graph.in_edges(node, data=True))
        out_edges = list(simple_graph.out_edges(node, data=True))
        in_positive = sum(1 for _, _, data in in_edges if data.get("signed_weight", 1) > 0)
        in_negative = sum(1 for _, _, data in in_edges if data.get("signed_weight", 1) < 0)
        out_positive = sum(1 for _, _, data in out_edges if data.get("signed_weight", 1) > 0)
        out_negative = sum(1 for _, _, data in out_edges if data.get("signed_weight", 1) < 0)
        rows.append(
            {
                "node": node,
                "community_id": community_map.get(node, -1),
                "community_size": community_sizes.get(community_map.get(node, -1), 0),
                "clustering_coefficient": clustering_scores.get(node, 0.0),
                "in_degree": simple_graph.in_degree(node),
                "out_degree": simple_graph.out_degree(node),
                "total_degree": simple_graph.degree(node),
                "in_positive_degree": in_positive,
                "in_negative_degree": in_negative,
                "out_positive_degree": out_positive,
                "out_negative_degree": out_negative,
                "pagerank": pagerank_scores.get(node, 0.0),
                "betweenness": betweenness_scores.get(node, 0.0),
                "reciprocity": reciprocity_scores.get(node, 0.0),
                "community_negative_ratio": community_stats.get(community_map.get(node, -1), {}).get("negative_ratio", 0.0),
                "community_out_negative_ratio": community_stats.get(community_map.get(node, -1), {}).get("out_negative_ratio", 0.0),
                "community_in_negative_ratio": community_stats.get(community_map.get(node, -1), {}).get("in_negative_ratio", 0.0),
            }
        )
    return pd.DataFrame(rows)


def _detect_communities(graph: nx.Graph) -> dict[str, int]:
    """Detect communities on the undirected projection for SNA features."""
    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_edges() == 0:
        return {node: index for index, node in enumerate(graph.nodes())}

    try:
        communities = nx.community.louvain_communities(graph, weight="weight", seed=42)
    except Exception:
        communities = nx.community.greedy_modularity_communities(graph, weight="weight")

    community_map: dict[str, int] = {}
    for community_id, community_nodes in enumerate(communities):
        for node in community_nodes:
            community_map[node] = community_id
    return community_map


def _build_community_sentiment_stats(graph: nx.MultiDiGraph, community_map: dict[str, int]) -> dict[int, dict[str, float]]:
    """Aggregate positive/negative interaction ratios at community level."""
    stats: dict[int, Counter] = defaultdict(Counter)
    for source, target, data in graph.edges(data=True):
        sentiment = int(data.get("sentiment", 1))
        source_community = community_map.get(source, -1)
        target_community = community_map.get(target, -1)

        stats[source_community]["out_total"] += 1
        stats[target_community]["in_total"] += 1
        stats[source_community]["total"] += 1
        if target_community != source_community:
            stats[target_community]["total"] += 1

        if sentiment == -1:
            stats[source_community]["out_negative"] += 1
            stats[target_community]["in_negative"] += 1
            stats[source_community]["negative"] += 1
            if target_community != source_community:
                stats[target_community]["negative"] += 1

    ratios: dict[int, dict[str, float]] = {}
    for community_id, counts in stats.items():
        ratios[community_id] = {
            "negative_ratio": counts["negative"] / counts["total"] if counts["total"] else 0.0,
            "out_negative_ratio": counts["out_negative"] / counts["out_total"] if counts["out_total"] else 0.0,
            "in_negative_ratio": counts["in_negative"] / counts["in_total"] if counts["in_total"] else 0.0,
        }
    return ratios


def build_edge_feature_table(frame: pd.DataFrame, graph: nx.MultiDiGraph) -> pd.DataFrame:
    """Compute edge-level frequency and reciprocity features."""
    table = aggregate_edge_table(frame)
    unique_pairs = set(zip(table["source_subreddit"], table["target_subreddit"]))
    reciprocity_flags = []
    for source, target in unique_pairs:
        reciprocity_flags.append((source, target, int((target, source) in unique_pairs)))
    reciprocity_map = {(source, target): reciprocal for source, target, reciprocal in reciprocity_flags}
    table["reciprocal_edge"] = [reciprocity_map.get((source, target), 0) for source, target in zip(table["source_subreddit"], table["target_subreddit"])]
    return table


def _signed_undirected_neighbors(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Build signed neighbor counts for a coarse structural-balance approximation."""
    adjacency: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    pair_signs: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in frame.itertuples(index=False):
        source = row.source_subreddit
        target = row.target_subreddit
        sign = int(row.link_sentiment)
        key = (source, target)
        pair_signs[key][sign] += 1
        pair_signs[(target, source)][sign] += 1

    for (source, target), counts in pair_signs.items():
        if source == target:
            continue
        majority_sign = 1 if counts[1] >= counts[-1] else -1
        adjacency[source][target] = majority_sign
    return adjacency


def build_triadic_feature_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Approximate structural balance counts using local signed neighborhoods."""
    adjacency = _signed_undirected_neighbors(frame)
    unique_pairs = frame[["source_subreddit", "target_subreddit"]].drop_duplicates()
    rows = []
    for source, target in unique_pairs.itertuples(index=False, name=None):
        source_neighbors = set(adjacency.get(source, {})) - {target}
        target_neighbors = set(adjacency.get(target, {})) - {source}
        common_neighbors = source_neighbors & target_neighbors
        balance_plusplus = balance_plusminus = balance_minusminus = balance_minusminus2 = 0
        for middle in common_neighbors:
            sign_st = adjacency[source].get(target, 1)
            sign_sm = adjacency[source].get(middle, 1)
            sign_mt = adjacency[middle].get(target, 1)
            pattern = (sign_st, sign_sm, sign_mt)
            if pattern == (1, 1, 1):
                balance_plusplus += 1
            elif pattern in {(1, 1, -1), (1, -1, 1), (-1, 1, 1)}:
                balance_plusminus += 1
            elif pattern in {(1, -1, -1), (-1, 1, -1), (-1, -1, 1)}:
                balance_minusminus += 1
            else:
                balance_minusminus2 += 1
        rows.append(
            {
                "source_subreddit": source,
                "target_subreddit": target,
                "common_neighbors": len(common_neighbors),
                "balance_+++": balance_plusplus,
                "balance_++-": balance_plusminus,
                "balance_+--": balance_minusminus,
                "balance_---": balance_minusminus2,
            }
        )
    return pd.DataFrame(rows)


def build_feature_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Merge node, edge, and triadic features into one modeling table."""
    graph, node_features, edge_features, triadic_features, text_features = build_feature_components(frame)
    return assemble_feature_dataset(node_features, edge_features, triadic_features, text_features)


def build_feature_components(frame: pd.DataFrame) -> tuple[nx.MultiDiGraph, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the reusable graph and feature tables for phase 2."""
    graph = build_signed_multidigraph(frame)
    node_features = build_node_feature_table(graph)
    edge_features = build_edge_feature_table(frame, graph)
    triadic_features = build_triadic_feature_table(frame)
    text_features = build_text_feature_table(frame)
    return graph, node_features, edge_features, triadic_features, text_features


def assemble_feature_dataset(
    node_features: pd.DataFrame,
    edge_features: pd.DataFrame,
    triadic_features: pd.DataFrame,
    text_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge precomputed feature tables into the final modeling table."""

    source_features = node_features.rename(columns={column: f"source_{column}" for column in node_features.columns if column != "node"})
    source_features = source_features.rename(columns={"node": "source_subreddit"})
    target_features = node_features.rename(columns={column: f"target_{column}" for column in node_features.columns if column != "node"})
    target_features = target_features.rename(columns={"node": "target_subreddit"})

    edge_features = edge_features.merge(
        source_features,
        left_on="source_subreddit",
        right_on="source_subreddit",
        how="left",
    )
    edge_features = edge_features.merge(
        target_features,
        left_on="target_subreddit",
        right_on="target_subreddit",
        how="left",
    )
    merged = edge_features.merge(triadic_features, on=["source_subreddit", "target_subreddit"], how="left")
    if text_features is not None and not text_features.empty:
        merged = merged.merge(text_features, on=["source_subreddit", "target_subreddit"], how="left")
    if {"source_community_id", "target_community_id"}.issubset(merged.columns):
        merged["same_community"] = (merged["source_community_id"] == merged["target_community_id"]).astype(int)
    if {"source_community_size", "target_community_size"}.issubset(merged.columns):
        smaller = merged[["source_community_size", "target_community_size"]].min(axis=1)
        larger = merged[["source_community_size", "target_community_size"]].max(axis=1).clip(lower=1)
        merged["community_size_ratio"] = smaller / larger
    if {"source_community_negative_ratio", "target_community_negative_ratio"}.issubset(merged.columns):
        merged["community_negative_ratio_gap"] = (
            merged["source_community_negative_ratio"] - merged["target_community_negative_ratio"]
        ).abs()
    merged["negative_label"] = (merged["negative_count"] > merged["positive_count"]).astype(int)
    merged = merged.fillna(0)
    return merged


def export_phase2_tables(frame: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Export the phase 2 tables to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    graph, node_features, edge_features, triadic_features, text_features = build_feature_components(frame)
    modeling_table = assemble_feature_dataset(node_features, edge_features, triadic_features, text_features)

    paths = {
        "node_features": output_path / "phase2_node_features.csv",
        "edge_features": output_path / "phase2_edge_features.csv",
        "triadic_features": output_path / "phase2_triadic_features.csv",
        "text_features": output_path / "phase2_text_features.csv",
        "modeling_table": output_path / "phase2_modeling_table.csv",
    }
    node_features.to_csv(paths["node_features"], index=False)
    edge_features.to_csv(paths["edge_features"], index=False)
    triadic_features.to_csv(paths["triadic_features"], index=False)
    text_features.to_csv(paths["text_features"], index=False)
    modeling_table.to_csv(paths["modeling_table"], index=False)
    return paths
