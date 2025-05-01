"""
This module contains the main function to execute explainability 
experiments.
"""

# standard libraries
import os
import time

# 3pps
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from scipy.sparse import lil_matrix
from tqdm.auto import tqdm
from typing import Literal, Type

# own modules
from src.utils import set_seed, load_data
from src.explain.methods import (
    Explainer,
    SaliencyMap,
    SmoothGrad,
    DeConvNet,
    GuidedBackprop,
    GNNExplainer,
)
from src.explain.executions import original_xai, parallel_xai


# set seed and device
set_seed(42)
torch.set_num_threads(8)
device = torch.device("cuda") if torch.cuda.is_available() else torch.cpu()

# static variables
DATA_PATH: str = "data"
LOAD_PATH: str = "models"
RESULTS_PATH: str = "results"
METHODS: dict[str, Type[Explainer]] = {
    "Saliency Map": SaliencyMap,
    # "Smoothgrad": SmoothGrad,
    # "Deconvnet": DeConvNet,
    # "Guided-Backprop": GuidedBackprop,
    # "GNNExplainer": GNNExplainer,
}
DATASETS_NAME: tuple[Literal["Cora", "CiteSeer", "PubMed"], ...] = (
    # "Cora",
    # "CiteSeer",
    "PubMed",
)
MODEL_NAMES: tuple[Literal["gcn", "gat"], ...] = ("gat",)
NUM_CLUSTERS: tuple[int, ...] = (8, 16, 32, 64, 128)
DROPOUT_RATES: tuple[float, ...] = (0.0, 0.2, 0.5, 0.7, 1.0)


@torch.no_grad
def main() -> None:
    """
    This function is the main program for the explain module.

    Returns:
        None.
    """

    # Empty nohup file
    open("nohup.out", "w").close()

    # Check device
    print(f"device: {device}")

    # Define progress bar
    progress_bar = tqdm(
        range(
            len(DATASETS_NAME)
            * len(MODEL_NAMES)
            # * len(METHODS)
            * len(NUM_CLUSTERS)
            * len(DROPOUT_RATES)
        )
    )

    # iter over dataset, model and method
    for dataset_name in DATASETS_NAME:
        for model_name in MODEL_NAMES:
            # define dataset
            dataset: InMemoryDataset = load_data(
                dataset_name, f"{DATA_PATH}/{dataset_name}"
            )

            # load model
            model: torch.nn.Module
            model = torch.load(f"{LOAD_PATH}/{dataset_name}_{model_name}.pt").to(device)
            model.eval()

            # pass elements to correct device
            x: torch.Tensor = dataset[0].x.float()
            edge_index: torch.Tensor = dataset[0].edge_index.long()
            node_ids: torch.Tensor = torch.arange(x.shape[0])
            data: Data = Data(x=x, edge_index=edge_index, node_ids=node_ids)

            # define explainer
            explainer: Explainer = SaliencyMap(model)

            # init results
            global_results = []

            # start time
            start = time.time()

            # compute feature maps
            original_feature_maps: lil_matrix = original_xai(
                explainer, data, device=device
            )

            # compute execution time
            exec_time = time.time() - start

            results = [
                1,
                0,
                exec_time,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
            global_results.append(results)

            # iterate over cluster size and dropout rate
            for num_clusters in NUM_CLUSTERS:
                for dropout_rate in DROPOUT_RATES:
                    # start time
                    start = time.time()

                    # compute xai
                    parallel_feature_maps: lil_matrix
                    (
                        parallel_feature_maps,
                        num_extended_nodes,
                        num_extended_edges,
                    ) = parallel_xai(
                        explainer,
                        data,
                        num_clusters,
                        3,
                        dropout_rate,
                        device=device,
                    )

                    # compute execution time
                    exec_time = time.time() - start

                    # compute difference
                    original_feature_maps_dense = original_feature_maps.todense()
                    parallel_feature_maps_dense = parallel_feature_maps.todense()
                    difference = np.abs(
                        parallel_feature_maps_dense - original_feature_maps_dense
                    )

                    # compute affected neighbors and nodes
                    percentage_affected_neighbors: list[float] = []
                    percentage_affected_nodes: list[float] = []
                    thresholds: list[float] = [0.2, 0.5, 0.7]
                    for threshold in thresholds:
                        percentage_affected_neighbors.append(
                            100
                            * (difference > threshold).sum()
                            / (original_feature_maps_dense != 0).sum()
                        )
                        percentage_affected_nodes.append(
                            100
                            * ((difference > threshold).sum(axis=1) != 0).sum()
                            / x.shape[0]
                        )

                    # append results
                    increment_nodes: int = num_extended_nodes - x.shape[0]
                    increment_nodes_percentage: float = (
                        100 * increment_nodes / x.shape[0]
                    )
                    increment_edges: int = num_extended_edges - edge_index.shape[1]
                    increment_edges_percentage: float = (
                        100 * increment_edges / edge_index.shape[1]
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

                    # update progress bar
                    progress_bar.update()

            # build dataframe
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

            # create results dir if it doesn't exist
            if not os.path.isdir(f"{RESULTS_PATH}"):
                os.makedirs(f"{RESULTS_PATH}")

            # save dataframe
            df.to_csv(
                f"{RESULTS_PATH}/{dataset_name}_{model_name}.csv", float_format="%.2f"
            )

    return None


if __name__ == "__main__":
    main()
