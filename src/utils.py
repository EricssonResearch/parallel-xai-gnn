"""
This module contains code 
"""

# Standard libraries
import os
import random
from typing import Literal

# 3pps
import torch
import torch_geometric
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import NormalizeFeatures

# Static variables
DATA_PATH: str = "data"
LOAD_PATH: str = "models"
DATASETS_NAME: tuple[Literal["Cora", "CiteSeer", "PubMed"], ...] = (
    "Cora",
    # "CiteSeer",
    # "PubMed",
)
MODEL_NAMES: tuple[Literal["gcn", "gat"], ...] = (
    "gcn",
    "gat",
)


def load_data(
    dataset_name: Literal["Cora", "CiteSeer", "PubMed"], save_path: str
) -> InMemoryDataset:
    """
    This function loads the datasets.

    Args:
        dataset_name: name of the dataset.
        save_path: path for saving the dataset locally.

    Returns:
        dataset.
    """

    # get dataset
    dataset: InMemoryDataset = torch_geometric.datasets.Planetoid(
        root=save_path, name=dataset_name, transform=NormalizeFeatures()
    )

    return dataset


def get_device(dataset_name: Literal["Cora", "CiteSeer", "PubMed"]) -> torch.device:
    """
    This function returns the correct device to use for each dataset.

    Args:
        dataset_name: Name of the dataset.

    Returns:
        Pytorch device.
    """

    # Select device depending on the dataset
    if dataset_name == "PubMed":
        device = torch.device("cpu")
    else:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    return device


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior

    Args:
        seed: seed to start all random operations.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None
