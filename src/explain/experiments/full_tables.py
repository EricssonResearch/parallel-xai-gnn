"""
This module contains the code to generat
"""

# Standard libraries
import os
import time

# 3pps
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm.auto import tqdm

# Own modules
from src.utils import (
    set_seed,
    load_data,
    DATA_PATH,
    LOAD_PATH,
    DATASETS_NAME,
    MODEL_NAMES,
    NUM_CLUSTERS,
    ITERATIONS,
)
from src.explain.methods import Explainer, DeConvNet
from src.explain.executions import original_xai, parallel_xai

# Set seed and device
set_seed(42)
torch.set_num_threads(8)

# Static variables
RESULTS_PATH: str = "results/full_reconstruction"


def main() -> None:
    """
    _summary_
    """

    # Empty nohup file
    open("nohup.out", "w").close()

    # Define progress bar
    progress_bar = tqdm(
        range(len(DATASETS_NAME) * len(MODEL_NAMES) * len(NUM_CLUSTERS))
    )

    # Iter over dataset and model
    for dataset_name in DATASETS_NAME:
        for model_name in MODEL_NAMES:
            # Define dataset
            dataset: InMemoryDataset = load_data(
                dataset_name, f"{DATA_PATH}/{dataset_name}"
            )

            # Select device depending on the dataset
            if dataset_name == "PubMed":
                device = torch.device("cpu")
            else:
                device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )

            # Print used device
            print(f"Using: {device}")

            # Load model
            model: torch.nn.Module
            model = torch.load(
                f"{LOAD_PATH}/{dataset_name}_{model_name}.pt", weights_only=False
            ).to(device)
            model.eval()

            # Pass elements to correct device
            x: torch.Tensor = dataset[0].x.float()
            edge_index: torch.Tensor = dataset[0].edge_index.long()
            node_ids: torch.Tensor = torch.arange(x.shape[0])
            data: Data = Data(x=x, edge_index=edge_index, node_ids=node_ids)

            # define explainer
            explainer: Explainer = DeConvNet(model)

            # init results
            global_results = []

            # Iterate over cluster size and dropout rate
            for num_clusters in NUM_CLUSTERS:
                # Init exec time
                exec_time = 0.0

                # Iter over executions
                for _ in range(ITERATIONS):
                    # Start time
                    start = time.time()

                    # Compute xai
                    if num_clusters == 1:
                        _ = original_xai(explainer, data, device=device)
                        num_extended_nodes: int = x.shape[0]
                        num_extended_edges: int = edge_index.shape[1]
                    else:
                        (
                            _,
                            num_extended_nodes,
                            num_extended_edges,
                        ) = parallel_xai(
                            explainer,
                            data,
                            num_clusters,
                            3,
                            device=device,
                        )

                    # Compute execution time
                    exec_time += time.time() - start

                # Divide between executions
                exec_time /= ITERATIONS

                # append results
                increment_nodes: int = num_extended_nodes - x.shape[0]
                increment_nodes_percentage: float = 100 * increment_nodes / x.shape[0]
                increment_edges: int = num_extended_edges - edge_index.shape[1]
                increment_edges_percentage: float = (
                    100 * increment_edges / edge_index.shape[1]
                )
                results = [
                    num_clusters,
                    exec_time,
                    increment_nodes,
                    increment_nodes_percentage,
                    increment_edges,
                    increment_edges_percentage,
                ]
                global_results.append(results)

                # update progress bar
                progress_bar.update()

            # Build dataframe
            df = pd.DataFrame(
                data=global_results,
                columns=[
                    "Number of Clusters",
                    "Execution time",
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
                f"{RESULTS_PATH}/tables/{dataset_name}_{model_name}.csv",
                float_format="%.2f",
            )

            # Save latex
            df[
                [
                    "Number of Clusters",
                    "Execution time",
                    "Increment in Number of Nodes (%)",
                    "Increment in Number of Edges (%)",
                ]
            ].to_latex(
                f"{RESULTS_PATH}/tables/{dataset_name}_{model_name}.tex",
                index=False,
                float_format="%.2f",
            )

            # Create results dir if it doesn't exist
            if not os.path.isdir(f"{RESULTS_PATH}/visualizations"):
                os.makedirs(f"{RESULTS_PATH}/visualizations")

            # Create figure
            max_x: float = 1.1 * df["Number of Clusters"].max()
            max_y: float = 1.1 * df["Execution time"].max()
            fig = plt.figure()
            plt.plot(
                df["Number of Clusters"],
                df["Execution time"],
                marker="o",
            )
            plt.xlabel("Number of Clusters")
            plt.ylabel("Execution time (s)")
            plt.xlim([0, max_x])
            plt.ylim([0, max_y])
            plt.grid()
            plt.savefig(
                f"{RESULTS_PATH}/visualizations/{dataset_name}_{model_name}_time.pdf",
                bbox_inches="tight",
                pad_inches=0,
                format="pdf",
            )
            plt.close()

            # Create figure
            max_y = 1.1 * df["Increment in Number of Nodes (%)"].max()
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.plot(
                df["Number of Clusters"],
                df["Increment in Number of Nodes (%)"],
                marker="o",
            )
            plt.xlabel("Number of Clusters")
            plt.ylabel("Increment in Number of Nodes (%)")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
            plt.xlim([0, max_x])
            plt.ylim([0, max_y])
            plt.grid()
            plt.savefig(
                f"{RESULTS_PATH}/visualizations/{dataset_name}_{model_name}_nodes.pdf",
                bbox_inches="tight",
                pad_inches=0,
                format="pdf",
            )
            plt.close()

            break

    return None
