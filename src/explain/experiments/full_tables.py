"""
This module contains the code for the experiment of full reconstruction.
"""

# Standard libraries
import os
from typing import Any

# 3pps
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# Own modules
from src.explain.constants import (
    CHECKPOINTS_PATH, CHECKPOINTS_PATH,
    DROPOUT_RATES,
    METHODS,
    NUM_CLUSTERS,
    LIMITS,
    METHODS,
    NUM_CLUSTERS,
    XAIType,
)
from src.explain.executions import compute_xai
from src.explain.methods import Explainer
from src.explain.utils import load_artifacts
from src.utils import (
    DATASETS_NAME,
    MODEL_NAMES,
    Data,
    DatasetName,
    ModelName,
    set_torch_config,
)

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
    progress_bar: tqdm = tqdm(
        range(
            len(DATASETS_NAME)
            * len(MODEL_NAMES)
            * len(METHODS)
            * (1 + len(NUM_CLUSTERS) + 1)
        )
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
                
                # Get original XAI 
                add_results(
                    results,
                    "original",
                    explainer,
                    data,
                    dataset_name,
                    model_name,
                    None,
                    checkpoints_folder_path,
                    device,
                    progress_bar,
                )
                
                # Iterate over cluster size for cluster XAI
                for num_clusters in NUM_CLUSTERS:
                    add_results(
                        results,
                        "cluster",
                        explainer,
                        data,
                        dataset_name,
                        model_name,
                        num_clusters,
                        checkpoints_folder_path,
                        device,
                        progress_bar,
                    )
                
                # Get results with as many clusters as nodes
                add_results(
                    results,
                    "cluster",
                    explainer,
                    data,
                    dataset_name,
                    model_name,
                    data.x.shape[0],
                    checkpoints_folder_path,
                    device,
                    progress_bar,
                )

                # Build dataframe
                df = pd.DataFrame(
                    data=results,
                    columns=[
                        "XAI Type",
                        "Number of Clusters",
                        "Execution time (s)",
                        "Increment in Number of Nodes",
                        "Increment in Number of Nodes (%)",
                        "Increment in Number of Edges",
                        "Increment in Number of Edges (%)",
                    ],  # ty: ignore[invalid-argument-type]
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
                        "XAI Type",
                        "Number of Clusters",
                        "Execution time (s)",
                        "Increment in Number of Nodes",
                        "Increment in Number of Nodes (%)",
                        "Increment in Number of Edges",
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
                global_results_array[:, 1:, :],
                list(df.columns),
                list(METHODS.keys()),
                f"{RESULTS_PATH}/charts",
                f"{dataset_name}_{model_name}",
            )

    return None


def add_results(
    results: list[list[Any]],
    xai_type: XAIType,
    explainer: Explainer,
    data: Data,
    dataset_name: DatasetName,
    model_name: ModelName,
    num_clusters: int | None,
    checkpoints_folder_path: str,
    device: torch.device,
    progress_bar: tqdm,
) -> None:
    """
    This function adds the results to the list for an specific case.

    Args:
        results: List where results will be added.
        xai_type: Type of explainability.
        explainer: Explainer to use.
        data: Data containing all the dataset.
        dataset_name: Name of the dataset.
        model_name: Name of the model.
        num_clusters: Number of clusters.
        checkpoints_folder_path: Folder to save checkpoints.
        device: Device to use for tensor operations.
        progress_bar: Progress bar.

    Returns:
        None.
    """
    
    # Go to next iteration if it exceeds computational resources
    if num_clusters is not None:
        if (dataset_name, model_name) in LIMITS.keys():
            if num_clusters > LIMITS[(dataset_name, model_name)]:
                return None

    # Compute XAI
    _, num_extended_nodes, num_extended_edges, exec_time = (
        compute_xai(
            explainer,
            xai_type,
            num_clusters,
            0.0,
            data,
            checkpoints_folder_path,
            device,
        )
    )

    # Append results
    x: torch.Tensor = data.x
    edge_index: torch.Tensor = data.edge_index
    increment_nodes: int = num_extended_nodes - x.shape[0]
    increment_nodes_percentage: float = (
        100 * increment_nodes / x.shape[0]
    )
    increment_edges: int = num_extended_edges - edge_index.shape[1]
    increment_edges_percentage: float = (
        100 * increment_edges / edge_index.shape[1]
    )
    results.append(
        [
            xai_type,
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

    # Get global result
    global_results = global_results[:, :, 1:].astype(np.float64)

    # Prepare for figure
    x_label: str = "Number of Clusters"
    y_label: str = "Execution time (s)"
    x_data: np.ndarray = global_results[0, :, columns.index(x_label)]
    y_data: np.ndarray = np.transpose(global_results[:, :, columns.index(y_label)])
    max_x: float = 1.1 * global_results[:, :, columns.index(x_label)].max().item()
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
    y_label = "Increment in Number of Nodes (%)"
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
