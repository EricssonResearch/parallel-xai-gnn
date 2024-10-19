# deep learning libraries
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
from scipy.sparse import lil_matrix

# other libraries
import os
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm.auto import tqdm
from typing import Literal, Type

# own modules
from src.train.models import GCN, GAT
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
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# static variables
DATA_PATH: str = "data"
LOAD_PATH: str = "models"
RESULTS_PATH: str = "results"
METHODS: dict[str, Type[Explainer]] = {
    "Saliency Map": SaliencyMap,
    "Smoothgrad": SmoothGrad,
    "Deconvnet": DeConvNet,
    "Guided-Backprop": GuidedBackprop,
    "GNNExplainer": GNNExplainer,
}
DATASETS_NAME: tuple[Literal["Cora", "CiteSeer", "PubMed"], ...] = ("Cora",)
MODEL_NAMES: tuple[Literal["gcn", "gat"], ...] = ("gcn",)
CLUSTER_SIZES: tuple[int, ...] = (8, 16, 32, 64, 128)
DROPOUT_RATES: tuple[float, ...] = (0.0, 0.2, 0.5, 0.7, 1.0)


@torch.no_grad
def main() -> None:
    """
    This function is the main program for the explain module.

    Returns:
        None.
    """

    # empty nohup file
    open("nohup.out", "w").close()

    # check device
    print(f"device: {device}")

    # define progress bar
    progress_bar = tqdm(
        range(
            len(DATASETS_NAME)
            * len(MODEL_NAMES)
            # * len(METHODS)
            * len(CLUSTER_SIZES)
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
            test_mask: torch.Tensor = dataset[0].test_mask
            node_ids: torch.Tensor = torch.arange(x.shape[0])

            # define explainer
            explainer: Explainer = SaliencyMap(model)

            # compute feature maps
            # start = time.time()
            # original_feature_maps: lil_matrix = original_xai(
            #     explainer, x, edge_index, node_ids, test_mask, device=device
            # )
            # print(time.time() - start)

            # init results
            global_results = []

            # iterate over cluster size and dropout rate
            for cluster_size in CLUSTER_SIZES:
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
                        x,
                        edge_index,
                        node_ids,
                        test_mask,
                        cluster_size,
                        3,
                        dropout_rate,
                        device=device,
                    )

                    # compute execution time
                    exec_time = time.time() - start

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
                        cluster_size,
                        dropout_rate,
                        exec_time,
                        increment_nodes,
                        increment_nodes_percentage,
                        increment_edges,
                        increment_edges_percentage,
                    ]
                    global_results.append(results)

                    # update progress bar
                    progress_bar.update()

            # build dataframe
            df = pd.DataFrame(
                data=global_results,
                columns=[
                    "Cluster Size",
                    "Dropout Rate",
                    "Execution time",
                    "Increment in Number of Nodes",
                    "Increment in Number of Nodes (%)",
                    "Increment in Number of Edges",
                    "Increment in Number of Edges (%)",
                ],
            )

            # create results dir if it doesn't exist
            if not os.path.isdir(f"{RESULTS_PATH}"):
                os.makedirs(f"{RESULTS_PATH}")

            # save dataframe
            df.to_csv(
                f"{RESULTS_PATH}/{dataset_name}_{model_name}.csv", float_format="%.2f"
            )

            # # check if they are equal
            # equal: bool = np.allclose(
            #     original_feature_maps.todense(),
            #     parallel_feature_maps.todense(),
            # )

            # # print if they are equal
            # print(equal)

    return False


if __name__ == "__main__":
    main()
