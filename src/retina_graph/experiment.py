from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from .data import iter_all_samples
from .graph_features import perturbation_set, extract_graph_features

GRAPH_FEATURES = [
    "graph_node_count", "branch_count", "endpoint_count", "junction_count",
    "graph_component_count", "total_branch_length_px", "mean_branch_length_px",
    "mean_tortuosity", "mean_smoothed_tortuosity", "median_vessel_diameter_px",
    "pruned_branch_count", "pruned_total_length_px", "pruned_endpoint_count",
    "pruned_junction_count", "pruned_component_count", "skeleton_pixel_count",
]


def run_graph_experiment(hrf_root: Path | str, output_dir: Path | str, chase_root: Path | str | None = None, include_chase: bool = True, sample_limit: int | None = None, prune_length_px: float = 5.0):
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    graph_dir = output_dir / "graphs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    rows, branch_rows = [], []

    for index, sample in enumerate(iter_all_samples(hrf_root, chase_root, include_chase)):
        if sample_limit is not None and index >= sample_limit:
            break
        variants = perturbation_set(sample.vessel_mask, sample.fov_mask)
        baseline_metrics, baseline_graph, baseline_branches = extract_graph_features(variants["original"], sample.fov_mask, prune_length_px)
        baseline_graph.graph.update({"dataset": sample.dataset, "image_id": sample.image_id, "condition": "original"})
        nx.write_graphml(baseline_graph, graph_dir / f"{sample.dataset}_{sample.image_id}_original.graphml")
        for index_branch, branch in enumerate(baseline_branches):
            branch_rows.append({"dataset": sample.dataset, "image_id": sample.image_id, "condition": "original", "branch_index": index_branch, **branch})
        for condition, candidate in variants.items():
            measured, graph, branches = (baseline_metrics, baseline_graph, baseline_branches) if condition == "original" else extract_graph_features(candidate, sample.fov_mask, prune_length_px)
            row = {"dataset": sample.dataset, "image_id": sample.image_id, "condition": condition}
            row.update(measured)
            for feature in GRAPH_FEATURES:
                base = baseline_metrics.get(feature, np.nan)
                value = measured.get(feature, np.nan)
                row[f"{feature}_change_pct"] = 0.0 if condition == "original" else (100.0 * (value - base) / base if np.isfinite(base) and base != 0 and np.isfinite(value) else np.nan)
            rows.append(row)
            if condition != "original":
                for index_branch, branch in enumerate(branches):
                    branch_rows.append({"dataset": sample.dataset, "image_id": sample.image_id, "condition": condition, "branch_index": index_branch, **branch})

    per_image = pd.DataFrame(rows)
    if per_image.empty:
        raise RuntimeError("No retinal samples were processed.")
    branch_table = pd.DataFrame(branch_rows)
    per_image.to_csv(output_dir / "per_image_graph_features.csv", index=False)
    branch_table.to_csv(output_dir / "branch_features.csv", index=False)
    change_columns = [f"{feature}_change_pct" for feature in GRAPH_FEATURES]
    summary = per_image.groupby(["dataset", "condition"], sort=False).agg(
        images=("image_id", "nunique"),
        **{column: (column, "mean") for column in change_columns},
    ).reset_index()
    summary.to_csv(output_dir / "dataset_stability_summary.csv", index=False)
    _plot_summary(summary, figures_dir)
    return per_image, branch_table, summary


def _plot_summary(summary: pd.DataFrame, figures_dir: Path):
    feature_columns = ["branch_count_change_pct", "endpoint_count_change_pct", "junction_count_change_pct", "total_branch_length_px_change_pct", "mean_tortuosity_change_pct", "median_vessel_diameter_px_change_pct"]
    labels = ["Branches", "Endpoints", "Junctions", "Total length", "Tortuosity", "Median diameter"]
    conditions = ["erosion_r1", "erosion_r2", "dilation_r1"]
    datasets = list(summary["dataset"].drop_duplicates())
    for dataset in datasets:
        current = summary[summary["dataset"] == dataset].set_index("condition")
        fig, ax = plt.subplots(figsize=(10, 5.5))
        x = np.arange(len(feature_columns))
        width = 0.22
        for i, condition in enumerate(conditions):
            if condition in current.index:
                vals = current.loc[condition, feature_columns].to_numpy(dtype=float)
                ax.bar(x + (i - 1) * width, vals, width, label=condition)
        ax.axhline(0, linewidth=0.8)
        ax.set_xticks(x, labels, rotation=15, ha="right")
        ax.set_ylabel("Mean relative change from original (%)")
        ax.set_title(f"Vascular graph feature stability: {dataset}")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{dataset.lower()}_graph_stability.png", dpi=180)
        plt.close(fig)
