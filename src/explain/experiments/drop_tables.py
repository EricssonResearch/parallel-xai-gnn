"""
This module contains the main function to execute explainability
experiments.
"""

# Standard libraries
import os

# 3pps
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from tqdm.auto import tqdm

# Own modules
from src.utils import (
    set_torch_config,
    DATASETS_NAME,
    MODEL_NAMES,
)
from src.explain.utils import (
    load_artifacts,
    METHODS,
    NUM_CLUSTERS,
    CHECKPOINTS_PATH,
    DROPOUT_RATES,
)
from src.explain.methods import Explainer
from src.explain.executions import compute_xai

# static variables
RESULTS_PATH: str = "results/drop"


@torch.no_grad()
def main() -> None:
    """
    This function is the main program for the explain module.

    Raises:
        RuntimeError: Original feature maps not computed at start.

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
            * len(NUM_CLUSTERS)
            * len(DROPOUT_RATES)
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
                global_results = []

                # Define folder path
                checkpoints_folder_path: str = (
                    f"{CHECKPOINTS_PATH}/{dataset_name}/{model_name}/{method_name}"
                )

                # Create dir if it doesnt´t exist
                if not os.path.isdir(checkpoints_folder_path):
                    # Create dirs
                    os.makedirs(checkpoints_folder_path)

                # Iterate over cluster size and dropout rate
                for num_clusters in NUM_CLUSTERS:
                    for dropout_rate in DROPOUT_RATES:
                        # Go to next iteration
                        if (num_clusters == 1 and dropout_rate > 0) or (
                            (
                                dataset_name == "PubMed"
                                and model_name == "gcn"
                                and num_clusters > 32
                            )
                            or (
                                dataset_name == "PubMed"
                                and model_name == "gat"
                                and num_clusters > 16
                            )
                        ):
                            progress_bar.update()
                            continue

                        # Compute XAI
                        (
                            parallel_feature_maps,
                            num_extended_nodes,
                            num_extended_edges,
                            exec_time,
                        ) = compute_xai(
                            explainer,
                            num_clusters,
                            dropout_rate,
                            data,
                            checkpoints_folder_path,
                            device,
                        )

                        # Get original feature maps
                        if num_clusters == 1:
                            original_feature_maps = parallel_feature_maps
                        if "original_feature_maps" not in locals():
                            raise RuntimeError(
                                "Original feature maps not computed at start."
                            )

                        # Compute difference
                        difference = np.abs(
                            parallel_feature_maps - original_feature_maps
                        )

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
                                * ((difference > threshold).sum(axis=1) != 0)
                                .sum()
                                .item()
                                / data.node_ids.shape[0]
                            )

                        # Append results
                        increment_nodes: int = num_extended_nodes - data.x.shape[0]
                        increment_nodes_percentage: float = (
                            100 * increment_nodes / data.x.shape[0]
                        )
                        increment_edges: int = (
                            num_extended_edges - data.edge_index.shape[1]
                        )
                        increment_edges_percentage: float = (
                            100 * increment_edges / data.edge_index.shape[1]
                        )
                        results = [
                            num_clusters,
                            dropout_rate,
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
                        global_results.append(results)

                        # Update progress bar
                        progress_bar.update()

                # Build dataframe
                df = pd.DataFrame(
                    data=global_results,
                    columns=[
                        "Number of Clusters",
                        "Dropout Rate",
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
                    ],
                )

                # Create results dir if it doesn't exist
                if not os.path.isdir(f"{RESULTS_PATH}"):
                    os.makedirs(f"{RESULTS_PATH}")

                # Save csv
                df.to_csv(
                    f"{RESULTS_PATH}/{dataset_name}_{model_name}_{method_name}.csv",
                    float_format="%.2f",
                )

                # Save latex
                df[
                    [
                        "Number of Clusters",
                        "Dropout Rate",
                        "Execution time",
                        "Increment in Number of Nodes (%)",
                        "Increment in Number of Edges (%)",
                        "Affected nodes threshold 0.2 (%)",
                        "Affected nodes threshold 0.5 (%)",
                        "Affected nodes threshold 0.7 (%)",
                        "Affected neighbors threshold 0.2 (%)",
                        "Affected neighbors threshold 0.5 (%)",
                        "Affected neighbors threshold 0.7 (%)",
                    ]
                ].to_latex(
                    f"{RESULTS_PATH}/{dataset_name}_{model_name}_{method_name}.tex",
                    index=False,
                    float_format="%.2f",
                )

    return None


if __name__ == "__main__":
    main()
