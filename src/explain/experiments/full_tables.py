"""
This module contains the code for the experiment of full reconstruction.
"""

# Standard libraries
import os

# 3pps
import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm.auto import tqdm

# Own modules
from src.utils import (
    set_torch_config,
    DATASETS_NAME,
    MODEL_NAMES,
)
from src.explain.utils import load_artifacts, METHODS, NUM_CLUSTERS, CHECKPOINTS_PATH
from src.explain.methods import Explainer
from src.explain.executions import compute_xai

# Static variables
RESULTS_PATH: str = "results/full"


def main() -> None:
    """
    This is the main function to execute full reconstruction experiment.

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
        range(len(DATASETS_NAME) * len(MODEL_NAMES) * len(METHODS) * len(NUM_CLUSTERS))
    )

    # Iter over dataset and model
    for dataset_name in DATASETS_NAME:
        for model_name in MODEL_NAMES:
            # Get dataset and device
            data: Data
            model: torch.nn.Module
            data, model = load_artifacts(dataset_name, model_name, device)

            # Init global results
            global_results = []

            # Iter over methods
            for method_name, method in METHODS.items():
                # Define explainer
                explainer: Explainer = method(model)

                # Init results
                results = []

                # Define folder path
                checkpoints_folder_path: str = (
                    f"{CHECKPOINTS_PATH}/{dataset_name}/{model_name}/{method_name}"
                )

                # Create dir if it doesnt´t exist
                if not os.path.isdir(checkpoints_folder_path):
                    # Create dirs
                    os.makedirs(checkpoints_folder_path)

                # Iterate over cluster size
                for num_clusters in NUM_CLUSTERS:
                    # Go to next iteration
                    if (
                        dataset_name == "PubMed"
                        and model_name == "gcn"
                        and num_clusters > 32
                    ) or (
                        dataset_name == "PubMed"
                        and model_name == "gat"
                        and num_clusters > 16
                    ):
                        progress_bar.update()
                        continue

                    # Compute XAI
                    _, num_extended_nodes, num_extended_edges, exec_time = compute_xai(
                        explainer,
                        num_clusters,
                        0.0,
                        data,
                        checkpoints_folder_path,
                        device,
                    )

                    # Append results
                    increment_nodes: int = num_extended_nodes - data.x.shape[0]
                    increment_nodes_percentage: float = (
                        100 * increment_nodes / data.x.shape[0]
                    )
                    increment_edges: int = num_extended_edges - data.edge_index.shape[1]
                    increment_edges_percentage: float = (
                        100 * increment_edges / data.edge_index.shape[1]
                    )
                    results.append(
                        [
                            num_clusters,
                            exec_time,
                            increment_nodes,
                            increment_nodes_percentage,
                            increment_edges,
                            increment_edges_percentage,
                        ]
                    )

                    # Update progress bar
                    progress_bar.update()

                # Build dataframe
                df = pd.DataFrame(
                    data=results,
                    columns=[
                        "Number of Clusters",
                        "Execution time (s)",
                        "Increment in Number of Nodes",
                        "Increment in Number of Nodes (%)",
                        "Increment in Number of Edges",
                        "Increment in Number of Edges (%)",
                    ],
                )

                # Create results dir if it doesn't exist
                if not os.path.isdir(f"{RESULTS_PATH}/tables"):
                    os.makedirs(f"{RESULTS_PATH}/tables")

                # Save csv
                df.to_csv(
                    f"{RESULTS_PATH}/tables/{dataset_name}_{model_name}_"
                    f"{method_name}.csv",
                    float_format="%.2f",
                )

                # Save latex
                df[
                    [
                        "Number of Clusters",
                        "Execution time (s)",
                        "Increment in Number of Nodes (%)",
                        "Increment in Number of Edges (%)",
                    ]
                ].to_latex(
                    f"{RESULTS_PATH}/tables/{dataset_name}_{model_name}_"
                    f"{method_name}.tex",
                    index=False,
                    float_format="%.2f",
                )

                # Append to global results
                global_results.append(results)

            # Get array from global results
            global_results_array: np.ndarray = np.array(global_results)

            # Create results dir if it doesn't exist
            if not os.path.isdir(f"{RESULTS_PATH}/charts"):
                os.makedirs(f"{RESULTS_PATH}/charts")

            # Draw figures
            draw_figures(
                global_results_array,
                list(df.columns),
                list(METHODS.keys()),
                f"{RESULTS_PATH}/charts",
                f"{dataset_name}_{model_name}",
            )

    return None


def draw_figures(
    global_results: np.ndarray,
    columns: list[str],
    methods: list[str],
    path: str,
    tag: str,
) -> None:
    """
    This function draws figures based on the global results.

    Args:
        global_results: Global results. Dimensions: [Number of methods,
            Number of Clusters, Number of KPIs].
        columns: Columns of the different KPIs.
        methods: List of methods that are used. This will be the legend
            of the visualizations.
        path: Path to save the figures.
        tag: Tag to save the figures.

    Returns:
        None.
    """

    # Prepare for figure
    x_label: str = columns[0]
    y_label: str = columns[1]
    x_data: np.ndarray = global_results[0, :, columns.index(x_label)]
    y_data: np.ndarray = np.transpose(global_results[:, :, columns.index(y_label)])
    max_x: float = 1.1 * global_results[:, :, columns.index(x_label)].max()
    max_y: float = 1.1 * y_data.max()

    # Create figure
    fig = plt.figure()
    plt.plot(x_data, y_data, marker="o")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim([0, max_x])
    plt.ylim([0, max_y])
    plt.grid()
    plt.legend(methods)
    plt.savefig(
        f"{path}/{tag}_time.pdf",
        bbox_inches="tight",
        pad_inches=0,
        format="pdf",
    )
    plt.close()

    # Prepare for figure
    y_label = columns[3]
    y_data = np.transpose(global_results[:, :, columns.index(y_label)])
    max_y = 1.1 * y_data.max()

    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(x_data, y_data, marker="o")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    plt.xlim([0, max_x])
    plt.ylim([0, max_y])
    plt.grid()
    plt.legend(methods)
    plt.savefig(
        f"{path}/{tag}_nodes.pdf",
        bbox_inches="tight",
        pad_inches=0,
        format="pdf",
    )
    plt.close()

    return None
