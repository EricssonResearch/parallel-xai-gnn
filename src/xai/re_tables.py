"""
This module contains the main function to execute explainability
experiments.
"""

# Standard libraries
import os
from typing import Any

# 3pps
import numpy as np
import pandas as pd
import torch
from scipy.sparse import lil_matrix
from tqdm.auto import tqdm

# Own modules
from src.utils import (
    DATASETS_NAME,
    MODEL_NAMES,
    set_torch_config,
)
from src.xai.constants import (
    CHECKPOINTS_PATH,
    METHODS,
    NUM_CLUSTERS,
    RATES,
    XAIType,
)
from src.xai.executions import compute_xai
from src.xai.methods import Explainer
from src.xai.utils import Data, load_artifacts

# Static variables
RESULTS_PATH: str = "results/drop"


@torch.no_grad()
def main() -> None:
    """
    This function is the main program for the explain module.

    Returns:
        None.
    """

    # Empty nohup file
    open("nohup.out", "w", encoding="utf-8").close()

    # Get device
    device: torch.device = set_torch_config()
    print(f"device: {device}")

    # Define progress bar
    progress_bar = tqdm(
        range(
            len(DATASETS_NAME)
            * len(MODEL_NAMES)
            * len(METHODS)
            * (1 + (len(NUM_CLUSTERS) + 1) * len(RATES))
        )
    )

    # Iter over dataset and model
    for dataset_name in DATASETS_NAME:
        for model_name in MODEL_NAMES:
            # Get dataset and device
            data: Data
            model: torch.nn.Module
            data, model = load_artifacts(dataset_name, model_name, device)

            # Iter over methods
            for method_name, method in METHODS.items():
                # Define explainer
                explainer: Explainer = method(model)

                # Init results
                results: list[Any] = []

                # Define folder path
                checkpoints_folder_path: str = (
                    f"{CHECKPOINTS_PATH}/{dataset_name}/{model_name}/{method_name}"
                )

                # Create dir if it doesn't exist
                if not os.path.isdir(checkpoints_folder_path):
                    # Create dirs
                    os.makedirs(checkpoints_folder_path)

                # Get original feature maps
                original_feature_maps = add_results(
                    results,
                    "original",
                    explainer,
                    data,
                    1,
                    1.0,
                    None,
                    checkpoints_folder_path,
                    device,
                    progress_bar,
                )

                # Iterate over cluster size and dropout rate
                for num_clusters in NUM_CLUSTERS:
                    for rate in RATES:
                        add_results(
                            results,
                            "cluster",
                            explainer,
                            data,
                            num_clusters,
                            rate,
                            original_feature_maps,
                            checkpoints_folder_path,
                            device,
                            progress_bar,
                        )

                # Add results of num_clusters=num_nodes
                for dropout_rate in RATES:
                    add_results(
                        results,
                        "cluster",
                        explainer,
                        data,
                        data.x.shape[0],
                        dropout_rate,
                        original_feature_maps,
                        checkpoints_folder_path,
                        device,
                        progress_bar,
                    )

                # Build original dataframe
                df = pd.DataFrame(
                    data=results,
                    columns=[
                        "Number of Clusters",
                        "Rate",
                        "Execution time",
                        "Increment in Number of Nodes",
                        "Increment in Number of Nodes (%)",
                        "Increment in Number of Edges",
                        "Increment in Number of Edges (%)",
                        "Affected neighbors threshold 0.2 (%)",
                        "Affected nodes threshold 0.2 (%)",
                        "Affected neighbors threshold 0.5 (%)",
                        "Affected nodes threshold 0.5 (%)",
                        "Affected neighbors threshold 0.7 (%)",
                        "Affected nodes threshold 0.7 (%)",
                    ],  # ty: ignore[invalid-argument-type]
                )

                # Get filtered df
                df = pd.DataFrame(
                    data=df[
                        [
                            "Number of Clusters",
                            "Rate",
                            "Execution time",
                            "Increment in Number of Nodes (%)",
                            "Increment in Number of Edges (%)",
                            "Affected neighbors threshold 0.2 (%)",
                            "Affected neighbors threshold 0.5 (%)",
                            "Affected neighbors threshold 0.7 (%)",
                            "Affected nodes threshold 0.2 (%)",
                            "Affected nodes threshold 0.5 (%)",
                            "Affected nodes threshold 0.7 (%)",
                        ]
                    ].to_numpy(),
                    columns=(
                        pd.MultiIndex.from_tuples(
                            [
                                ("Number of Clusters", ""),
                                ("Rate", ""),
                                ("Execution time (s)", ""),
                                ("Δnodes (%)", ""),
                                ("Δedges (%)", ""),
                                ("Affected explanations (%)", "0.2"),
                                ("Affected explanations (%)", "0.5"),
                                ("Affected explanations (%)", "0.7"),
                                ("Affected nodes (%)", "0.2"),
                                ("Affected nodes (%)", "0.5"),
                                ("Affected nodes (%)", "0.7"),
                            ]
                        )
                    ),
                )
                df = df.astype({("Number of Clusters", ""): "int16"})

                # Create results dir if it doesn't exist
                if not os.path.isdir(f"{RESULTS_PATH}"):
                    os.makedirs(f"{RESULTS_PATH}")

                # Save csv
                df.to_csv(
                    f"{RESULTS_PATH}/{dataset_name}_{model_name}_{method_name}.csv",
                    float_format="%.2f",
                )

                # Save latex
                df.to_latex(
                    f"{RESULTS_PATH}/{dataset_name}_{model_name}_{method_name}.tex",
                    index=False,
                    float_format="%.2f",
                )

    return None


def add_results(
    results: list[list[Any]],
    xai_type: XAIType,
    explainer: Explainer,
    data: Data,
    num_clusters: int | None,
    rate: float | None,
    original_feature_maps: lil_matrix | None,
    checkpoints_folder_path: str,
    device: torch.device,
    progress_bar: tqdm,
) -> lil_matrix:
    """
    This function adds the results to the list for an specific case.

    Args:
        results: List where results will be added.
        xai_type: Type of explainability.
        explainer: Explainer to use.
        data: Data containing all the dataset.
        num_clusters: Number of clusters.
        rate: Reconstruction rate for the dropout.
        original_feature_maps: Original feature maps to compare. If
            'xai_type' is 'original' it can be None since it will be
            reinitialized to the computed value.
        checkpoints_folder_path: Folder to save checkpoints.
        device: Device to use for tensor operations.
        progress_bar: Progress bar.

    Returns:
        Feature maps compute in this function.
    """

    # Compute XAI
    (
        parallel_feature_maps,
        num_extended_nodes,
        num_extended_edges,
        exec_time,
    ) = compute_xai(
        explainer,
        xai_type,
        num_clusters,
        rate,
        data,
        checkpoints_folder_path,
        device,
    )

    # Init original
    if xai_type == "original":
        original_feature_maps = parallel_feature_maps
    elif original_feature_maps is None:
        raise ValueError(
            "'original_feature_maps' can only be None when 'xai_type' is 'original'"
        )

    # Compute difference
    difference = np.abs(parallel_feature_maps - original_feature_maps)

    # Compute affected neighbors and nodes
    percentage_affected_neighbors: list[float] = []
    percentage_affected_nodes: list[float] = []
    thresholds: list[float] = [0.2, 0.5, 0.7]
    for threshold in thresholds:
        percentage_affected_neighbors.append(
            100
            * (difference > threshold).sum().item()
            / (original_feature_maps != 0).sum().item()
        )
        percentage_affected_nodes.append(
            100
            * ((difference > threshold).sum(axis=1) != 0).sum().item()
            / data.node_ids.shape[0]
        )

    # Append results
    x: torch.Tensor = data.x
    edge_index: torch.Tensor = data.edge_index
    increment_nodes: int = num_extended_nodes - x.shape[0]
    increment_nodes_percentage: float = 100 * increment_nodes / x.shape[0]
    increment_edges: int = num_extended_edges - edge_index.shape[1]
    increment_edges_percentage: float = 100 * increment_edges / edge_index.shape[1]
    results.append(
        [
            num_clusters,
            rate,
            exec_time,
            increment_nodes,
            increment_nodes_percentage,
            increment_edges,
            increment_edges_percentage,
            percentage_affected_neighbors[0],
            percentage_affected_nodes[0],
            percentage_affected_neighbors[1],
            percentage_affected_nodes[1],
            percentage_affected_neighbors[2],
            percentage_affected_nodes[2],
        ]
    )

    # Update progress bar
    progress_bar.update()

    return parallel_feature_maps


if __name__ == "__main__":
    main()
