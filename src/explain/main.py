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
METHODS: dict[str, Type[Explainer]] = {
    "Saliency Map": SaliencyMap,
    "Smoothgrad": SmoothGrad,
    "Deconvnet": DeConvNet,
    "Guided-Backprop": GuidedBackprop,
    "GNNExplainer": GNNExplainer,
}
DATASETS_NAME: tuple[Literal["Cora", "CiteSeer", "PubMed"], ...] = ("Cora",)
MODEL_NAMES: tuple[Literal["gcn", "gat"], ...] = ("gcn",)


def main() -> None:
    # empty nohup file
    open("nohup.out", "w").close()

    # check device
    print(f"device: {device}")

    # define progress bar
    progress_bar = tqdm(range(len(DATASETS_NAME) * len(MODEL_NAMES) * len(METHODS) * 2))

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

            explainer: Explainer = SaliencyMap(model)

            # compute feature maps
            start = time.time()
            original_feature_maps: lil_matrix = original_xai(
                explainer, x, edge_index, node_ids, test_mask, device=device
            )
            print(time.time() - start)
            start = time.time()
            parallel_feature_maps: lil_matrix = parallel_xai(
                explainer, x, edge_index, node_ids, test_mask, 64, device=device
            )
            print(time.time() - start)

            equal: bool = np.allclose(
                original_feature_maps[original_feature_maps != 0].todense(),
                parallel_feature_maps[parallel_feature_maps != 0].todense(),
            )


if __name__ == "__main__":
    main()
