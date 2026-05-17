"""Reusable plotting helpers for notebook and report-ready figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve


FIGURE_DPI = 180
PALETTE = {
    "positive": "#4C78A8",
    "negative": "#E45756",
    "neutral": "#72B7B2",
    "text": "#333333",
}


def _prepare_output_dir(output_dir: str | Path | None) -> Path | None:
    if output_dir is None:
        return None
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _display_figure(fig) -> None:
    """Display a matplotlib figure when running inside a notebook."""
    try:
        from IPython.display import display

        display(fig)
    except Exception:
        fig.show()


def _finalize_figure(fig, path: Path | None = None, *, show: bool = False) -> Path | None:
    """Optionally display, save, and close a figure.

    The figure is displayed before saving so notebook readers see the plot at
    the exact point where it is created, while the same figure object is still
    saved for the report.
    """
    fig.tight_layout()
    if show:
        _display_figure(fig)
    if path is not None:
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _output_path(output_dir: str | Path | None, filename: str) -> Path | None:
    output_path = _prepare_output_dir(output_dir)
    return output_path / filename if output_path is not None else None


def plot_label_distribution(frame: pd.DataFrame, output_dir: str | Path | None = None, *, show: bool = False) -> Path | None:
    """Plot positive/neutral versus negative hyperlink counts."""
    label_counts = frame["link_sentiment"].map({1: "positive/neutral", -1: "negative"}).value_counts()
    colors = [PALETTE["negative"] if label == "negative" else PALETTE["positive"] for label in label_counts.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=label_counts.index, y=label_counts.values, hue=label_counts.index, palette=colors, legend=False, ax=ax)
    ax.set_title("Distribution of Reddit hyperlink sentiment labels")
    ax.set_xlabel("Label")
    ax.set_ylabel("Number of hyperlinks")
    for index, value in enumerate(label_counts.values):
        ax.text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=9)

    return _finalize_figure(fig, _output_path(output_dir, "label_distribution.png"), show=show)


def plot_monthly_negative_ratio(frame: pd.DataFrame, output_dir: str | Path | None = None, *, show: bool = False) -> Path | None:
    """Plot the monthly share of negative cross-community hyperlinks."""
    dated = frame.copy()
    dated["timestamp"] = pd.to_datetime(dated["timestamp"], errors="coerce")
    dated = dated.dropna(subset=["timestamp"])
    dated["month"] = dated["timestamp"].dt.to_period("M").astype(str)
    monthly = dated.groupby("month")["link_sentiment"].agg(
        total="size",
        negative=lambda values: int((values == -1).sum()),
    ).reset_index()
    monthly["negative_ratio"] = monthly["negative"] / monthly["total"].clip(lower=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=monthly, x="month", y="negative_ratio", marker="o", color=PALETTE["negative"], ax=ax)
    ax.set_title("Monthly negative hyperlink ratio")
    ax.set_xlabel("Month")
    ax.set_ylabel("Negative ratio")
    ax.tick_params(axis="x", rotation=60, labelsize=7)

    return _finalize_figure(fig, _output_path(output_dir, "monthly_negative_ratio.png"), show=show)


def plot_top_negative_subreddits(
    frame: pd.DataFrame,
    output_dir: str | Path | None = None,
    *,
    top_n: int = 15,
    show: bool = False,
) -> list[Path | None]:
    """Plot top source and target subreddits by negative hyperlink count."""
    negative_frame = frame[frame["link_sentiment"] == -1]
    paths: list[Path | None] = []
    for column, label, filename in [
        ("source_subreddit", "Source subreddit", "top_negative_sources.png"),
        ("target_subreddit", "Target subreddit", "top_negative_targets.png"),
    ]:
        counts = negative_frame[column].value_counts().head(top_n).sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=counts.values, y=counts.index, color=PALETTE["negative"], ax=ax)
        ax.set_title(f"Top {top_n} {label.lower()}s by negative links")
        ax.set_xlabel("Negative hyperlink count")
        ax.set_ylabel(label)
        paths.append(_finalize_figure(fig, _output_path(output_dir, filename), show=show))
    return paths


def plot_degree_distribution(frame: pd.DataFrame, output_dir: str | Path | None = None, *, show: bool = False) -> Path | None:
    """Plot source and target degree distributions on a log scale."""
    out_degree = frame.groupby("source_subreddit")["target_subreddit"].nunique()
    in_degree = frame.groupby("target_subreddit")["source_subreddit"].nunique()
    degree_frame = pd.concat(
        [
            pd.DataFrame({"degree": out_degree, "type": "out-degree"}),
            pd.DataFrame({"degree": in_degree, "type": "in-degree"}),
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=degree_frame, x="degree", hue="type", bins=60, log_scale=(True, True), element="step", ax=ax)
    ax.set_title("Directed subreddit degree distribution")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Number of subreddits")

    return _finalize_figure(fig, _output_path(output_dir, "degree_distribution.png"), show=show)


def plot_model_comparison(metrics_frame: pd.DataFrame, output_dir: str | Path | None = None, *, show: bool = False) -> Path | None:
    """Plot test PR-AUC for the strongest model and feature-set combinations."""
    plot_frame = metrics_frame.copy()
    plot_frame["model_label"] = plot_frame["feature_set"] + " | " + plot_frame["model"]
    plot_frame = plot_frame.sort_values("test_pr_auc", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(plot_frame))))
    sns.barplot(data=plot_frame, x="test_pr_auc", y="model_label", color=PALETTE["positive"], ax=ax)
    ax.set_title("Model comparison on strict temporal test set")
    ax.set_xlabel("Test PR-AUC")
    ax.set_ylabel("Feature set | model")

    return _finalize_figure(fig, _output_path(output_dir, "model_comparison_pr_auc.png"), show=show)


def plot_best_confusion_matrix(metrics_frame: pd.DataFrame, output_dir: str | Path | None = None, *, show: bool = False) -> Path | None:
    """Plot the test confusion matrix for the best PR-AUC model."""
    best = metrics_frame.sort_values(["test_pr_auc", "test_f1"], ascending=False).iloc[0]
    matrix = [
        [int(best["test_tn"]), int(best["test_fp"])],
        [int(best["test_fn"]), int(best["test_tp"])],
    ]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"], ax=ax)
    ax.set_title(f"Best model confusion matrix: {best['feature_set']} | {best['model']}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    return _finalize_figure(fig, _output_path(output_dir, "best_confusion_matrix.png"), show=show)


def plot_feature_importance(
    importance_frame: pd.DataFrame,
    metrics_frame: pd.DataFrame,
    output_dir: str | Path | None = None,
    *,
    top_n: int = 20,
    show: bool = False,
) -> Path | None:
    """Plot feature importances for the best non-baseline model."""
    ranked_models = metrics_frame[
        ~metrics_frame["model"].str.startswith("dummy")
        & (metrics_frame["model"] != "historical_negative_ratio")
    ].sort_values(["test_pr_auc", "test_f1"], ascending=False)
    if ranked_models.empty:
        raise ValueError("No fitted model with feature importances is available.")
    best = ranked_models.iloc[0]
    filtered = importance_frame[
        (importance_frame["feature_set"] == best["feature_set"])
        & (importance_frame["model"] == best["model"])
    ].sort_values("importance", ascending=False).head(top_n).sort_values("importance")

    fig, ax = plt.subplots(figsize=(8, max(5, 0.3 * len(filtered))))
    sns.barplot(data=filtered, x="importance", y="feature", color=PALETTE["neutral"], ax=ax)
    ax.set_title(f"Top {top_n} features: {best['feature_set']} | {best['model']}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    return _finalize_figure(fig, _output_path(output_dir, "feature_importance_top20.png"), show=show)


def _select_curve_models(metrics_frame: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Select a compact, informative set of models for curve plots."""
    ranked = metrics_frame[
        ~metrics_frame["model"].str.startswith("dummy")
    ].sort_values(["test_pr_auc", "test_f1"], ascending=False)
    selected = ranked.head(top_n).copy()
    historical = ranked[ranked["model"] == "historical_negative_ratio"].head(1)
    if not historical.empty:
        selected = pd.concat([selected, historical], ignore_index=True).drop_duplicates(["feature_set", "model"])
    return selected


def plot_precision_recall_curve(
    score_frame: pd.DataFrame,
    metrics_frame: pd.DataFrame,
    output_dir: str | Path | None = None,
    *,
    top_n: int = 5,
    show: bool = False,
) -> Path | None:
    """Plot test-set precision-recall curves for top models and a heuristic baseline."""
    test_scores = score_frame[score_frame["split"] == "test"].copy()
    selected = _select_curve_models(metrics_frame, top_n=top_n)

    fig, ax = plt.subplots(figsize=(7, 5))
    for row in selected.itertuples(index=False):
        subset = test_scores[(test_scores["feature_set"] == row.feature_set) & (test_scores["model"] == row.model)]
        if subset.empty or subset["y_true"].nunique() < 2:
            continue
        precision, recall, _ = precision_recall_curve(subset["y_true"], subset["score"])
        label = f"{row.feature_set} | {row.model} (AP={row.test_pr_auc:.3f})"
        ax.plot(recall, precision, linewidth=1.8, label=label)

    prevalence = test_scores["y_true"].mean() if not test_scores.empty else 0.0
    ax.axhline(prevalence, color="gray", linestyle="--", linewidth=1.2, label=f"prevalence={prevalence:.3f}")
    ax.set_title("Precision-recall curves on strict temporal test set")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=8)

    return _finalize_figure(fig, _output_path(output_dir, "precision_recall_curve.png"), show=show)


def plot_roc_curve(
    score_frame: pd.DataFrame,
    metrics_frame: pd.DataFrame,
    output_dir: str | Path | None = None,
    *,
    top_n: int = 5,
    show: bool = False,
) -> Path | None:
    """Plot test-set ROC curves for top models and a heuristic baseline."""
    test_scores = score_frame[score_frame["split"] == "test"].copy()
    selected = _select_curve_models(metrics_frame, top_n=top_n)

    fig, ax = plt.subplots(figsize=(7, 5))
    for row in selected.itertuples(index=False):
        subset = test_scores[(test_scores["feature_set"] == row.feature_set) & (test_scores["model"] == row.model)]
        if subset.empty or subset["y_true"].nunique() < 2:
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(subset["y_true"], subset["score"])
        roc_auc = auc(false_positive_rate, true_positive_rate)
        label = f"{row.feature_set} | {row.model} (AUC={roc_auc:.3f})"
        ax.plot(false_positive_rate, true_positive_rate, linewidth=1.8, label=label)

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.2, label="random")
    ax.set_title("ROC curves on strict temporal test set")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=8)

    return _finalize_figure(fig, _output_path(output_dir, "roc_curve.png"), show=show)


def plot_community_negative_ratio(
    node_features: pd.DataFrame,
    output_dir: str | Path | None = None,
    *,
    top_n: int = 15,
    min_size: int = 20,
    show: bool = False,
) -> Path | None:
    """Plot communities with the highest average negative-link ratio."""
    required = {"community_id", "community_negative_ratio", "node"}
    if not required.issubset(node_features.columns):
        raise ValueError(f"node_features must contain columns: {sorted(required)}")

    summary = (
        node_features.groupby("community_id", dropna=False)
        .agg(
            community_size=("node", "size"),
            negative_ratio=("community_negative_ratio", "mean"),
            avg_pagerank=("pagerank", "mean") if "pagerank" in node_features.columns else ("node", "size"),
        )
        .reset_index()
    )
    summary = summary[summary["community_size"] >= min_size]
    summary = summary.sort_values(["negative_ratio", "community_size"], ascending=False).head(top_n)
    summary = summary.sort_values("negative_ratio")
    summary["label"] = summary.apply(lambda row: f"C{int(row.community_id)} (n={int(row.community_size)})", axis=1)

    fig, ax = plt.subplots(figsize=(8, max(5, 0.35 * len(summary))))
    sns.barplot(data=summary, x="negative_ratio", y="label", color=PALETTE["negative"], ax=ax)
    ax.set_title(f"Top {len(summary)} communities by negative-link ratio")
    ax.set_xlabel("Average community negative ratio")
    ax.set_ylabel("Community")

    return _finalize_figure(fig, _output_path(output_dir, "community_negative_ratio.png"), show=show)


def _top_nodes_for_network(node_features: pd.DataFrame, max_nodes: int) -> pd.DataFrame:
    ranking_column = "pagerank" if "pagerank" in node_features.columns else "total_degree"
    columns = ["node", "community_id", "community_size", "total_degree", "pagerank"]
    available = [column for column in columns if column in node_features.columns]
    return node_features[available].sort_values(ranking_column, ascending=False).head(max_nodes).copy()


def plot_community_network_sample(
    interactions: pd.DataFrame,
    node_features: pd.DataFrame,
    output_dir: str | Path | None = None,
    *,
    max_nodes: int = 250,
    max_edges: int = 700,
    label_count: int = 20,
    show: bool = False,
) -> Path | None:
    """Plot a readable subreddit network sample colored by detected community."""
    node_sample = _top_nodes_for_network(node_features, max_nodes=max_nodes)
    sampled_nodes = set(node_sample["node"])
    edge_frame = interactions[
        interactions["source_subreddit"].isin(sampled_nodes)
        & interactions["target_subreddit"].isin(sampled_nodes)
    ].copy()
    edge_summary = (
        edge_frame.groupby(["source_subreddit", "target_subreddit"], dropna=False)["link_sentiment"]
        .agg(
            interaction_count="size",
            positive_count=lambda values: int((values == 1).sum()),
            negative_count=lambda values: int((values == -1).sum()),
        )
        .reset_index()
    )
    edge_summary["majority_sentiment"] = np.where(edge_summary["negative_count"] > edge_summary["positive_count"], -1, 1)
    edge_summary = edge_summary.sort_values("interaction_count", ascending=False).head(max_edges)

    graph = nx.DiGraph()
    for row in node_sample.itertuples(index=False):
        graph.add_node(
            row.node,
            community_id=int(getattr(row, "community_id", -1)),
            total_degree=float(getattr(row, "total_degree", 1.0)),
            pagerank=float(getattr(row, "pagerank", 0.0)),
        )
    for row in edge_summary.itertuples(index=False):
        graph.add_edge(
            row.source_subreddit,
            row.target_subreddit,
            weight=float(row.interaction_count),
            majority_sentiment=int(row.majority_sentiment),
        )

    nodes = list(graph.nodes())
    if not nodes:
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_title("Community-colored subreddit network sample")
        ax.text(0.5, 0.5, "No sampled nodes available", ha="center", va="center")
        ax.axis("off")
        return _finalize_figure(fig, _output_path(output_dir, "community_network_sample.png"), show=show)

    layout_graph = graph.to_undirected()
    pos = nx.spring_layout(layout_graph, seed=42, weight="weight", iterations=80)
    degree_values = np.array([graph.nodes[node].get("total_degree", 1.0) for node in nodes], dtype=float)
    if degree_values.max() > degree_values.min():
        node_sizes = 80 + 820 * (np.log1p(degree_values) - np.log1p(degree_values).min()) / (
            np.log1p(degree_values).max() - np.log1p(degree_values).min()
        )
    else:
        node_sizes = np.full(len(nodes), 180.0)

    community_values = [graph.nodes[node].get("community_id", -1) for node in nodes]
    unique_communities = {community: index for index, community in enumerate(sorted(set(community_values)))}
    node_colors = [unique_communities[value] for value in community_values]
    cmap = plt.get_cmap("tab20", max(1, len(unique_communities)))
    edge_colors = [
        PALETTE["negative"] if data.get("majority_sentiment", 1) == -1 else PALETTE["positive"]
        for _, _, data in graph.edges(data=True)
    ]
    edge_widths = [0.4 + min(2.2, np.log1p(data.get("weight", 1.0)) / 2) for _, _, data in graph.edges(data=True)]

    fig, ax = plt.subplots(figsize=(10, 8))
    if graph.number_of_edges():
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.25, arrows=False)
    node_artist = nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=cmap,
        alpha=0.9,
        linewidths=0.2,
        edgecolors="white",
    )
    label_nodes = node_sample.sort_values("pagerank" if "pagerank" in node_sample.columns else "total_degree", ascending=False).head(label_count)["node"]
    labels = {node: node for node in label_nodes if node in graph}
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=7, font_color=PALETTE["text"])
    colorbar = fig.colorbar(node_artist, ax=ax, fraction=0.035, pad=0.01)
    colorbar.set_label("Detected community index")
    ax.set_title(f"Community-colored subreddit network sample ({len(nodes)} nodes, {graph.number_of_edges()} edges)")
    ax.axis("off")

    return _finalize_figure(fig, _output_path(output_dir, "community_network_sample.png"), show=show)


def plot_community_pair_negative_heatmap(
    interactions: pd.DataFrame,
    node_features: pd.DataFrame,
    output_dir: str | Path | None = None,
    *,
    top_n: int = 15,
    show: bool = False,
) -> Path | None:
    """Plot negative-link ratios between the largest detected communities."""
    if not {"node", "community_id"}.issubset(node_features.columns):
        raise ValueError("node_features must contain 'node' and 'community_id'.")

    community_map = node_features.set_index("node")["community_id"].to_dict()
    working = interactions[["source_subreddit", "target_subreddit", "link_sentiment"]].copy()
    working["source_community"] = working["source_subreddit"].map(community_map)
    working["target_community"] = working["target_subreddit"].map(community_map)
    working = working.dropna(subset=["source_community", "target_community"])
    working["source_community"] = working["source_community"].astype(int)
    working["target_community"] = working["target_community"].astype(int)

    top_communities = node_features["community_id"].value_counts().head(top_n).index.astype(int).tolist()
    working = working[
        working["source_community"].isin(top_communities)
        & working["target_community"].isin(top_communities)
    ]
    pair_summary = (
        working.groupby(["source_community", "target_community"], dropna=False)["link_sentiment"]
        .agg(total="size", negative=lambda values: int((values == -1).sum()))
        .reset_index()
    )
    pair_summary["negative_ratio"] = pair_summary["negative"] / pair_summary["total"].clip(lower=1)
    pivot = pair_summary.pivot(index="source_community", columns="target_community", values="negative_ratio")
    pivot = pivot.reindex(index=top_communities, columns=top_communities)
    pivot = pivot.apply(pd.to_numeric, errors="coerce").astype(float)
    mask = pivot.isna()
    plot_data = pivot.fillna(0.0)
    labels = [f"C{community}" for community in top_communities]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        plot_data,
        mask=mask,
        cmap="Reds",
        vmin=0,
        vmax=max(0.15, float(pair_summary["negative_ratio"].quantile(0.95)) if not pair_summary.empty else 0.15),
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Negative hyperlink ratio"},
        ax=ax,
    )
    ax.set_title(f"Negative-link ratio between top {len(top_communities)} communities")
    ax.set_xlabel("Target community")
    ax.set_ylabel("Source community")

    return _finalize_figure(fig, _output_path(output_dir, "community_pair_negative_heatmap.png"), show=show)


def export_report_figures(
    interactions: pd.DataFrame,
    metrics_frame: pd.DataFrame,
    importance_frame: pd.DataFrame,
    output_dir: str | Path,
    score_frame: pd.DataFrame | None = None,
    node_features: pd.DataFrame | None = None,
    *,
    show: bool = False,
) -> dict[str, Path | list[Path | None] | None]:
    """Export the main figures expected in the final report."""
    figures: dict[str, Path | list[Path | None] | None] = {
        "label_distribution": plot_label_distribution(interactions, output_dir, show=show),
        "monthly_negative_ratio": plot_monthly_negative_ratio(interactions, output_dir, show=show),
        "top_negative_subreddits": plot_top_negative_subreddits(interactions, output_dir, show=show),
        "degree_distribution": plot_degree_distribution(interactions, output_dir, show=show),
        "model_comparison": plot_model_comparison(metrics_frame, output_dir, show=show),
        "best_confusion_matrix": plot_best_confusion_matrix(metrics_frame, output_dir, show=show),
        "feature_importance": plot_feature_importance(importance_frame, metrics_frame, output_dir, show=show),
    }
    if score_frame is not None and not score_frame.empty:
        figures["precision_recall_curve"] = plot_precision_recall_curve(score_frame, metrics_frame, output_dir, show=show)
        figures["roc_curve"] = plot_roc_curve(score_frame, metrics_frame, output_dir, show=show)
    if node_features is not None and not node_features.empty:
        figures["community_negative_ratio"] = plot_community_negative_ratio(node_features, output_dir, show=show)
        figures["community_network_sample"] = plot_community_network_sample(interactions, node_features, output_dir, show=show)
        figures["community_pair_negative_heatmap"] = plot_community_pair_negative_heatmap(interactions, node_features, output_dir, show=show)
    return figures
