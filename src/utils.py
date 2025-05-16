# 3pps
import torch
import torch_geometric
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import NormalizeFeatures

# Standard libraries
import os
import sys
import random
from typing import Literal

# Static variables
DATA_PATH: str = "data"
LOAD_PATH: str = "models"
DATASETS_NAME: tuple[Literal["Cora", "CiteSeer", "PubMed"], ...] = (
    "Cora",
    "CiteSeer",
    "PubMed",
)
MODEL_NAMES: tuple[Literal["gcn", "gat"], ...] = (
    "gcn",
    "gat",
)
NUM_CLUSTERS: tuple[int, ...] = (1, 8, 16, 32, 64, 128)
ITERATIONS: float = 3


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


class HiddenPrints:
    """
    This class avoid printing in command line. It is intended to be
    used as a context manager, using the with statement of python.

    Atributtes:
        _original_stdout
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
