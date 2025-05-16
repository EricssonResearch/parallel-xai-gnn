"""
This module is to create examples visualizations.
"""

# Standard libraries
import os
from typing import Literal, Type

# 3pps
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset
import networkx as nx
import matplotlib.pyplot as plt

# Own modules
from src.utils import set_seed, load_data
from src.explain.methods import Explainer, SaliencyMap, GNNExplainer

# Static variables
DATA_PATH: str = "data"
DATASET_NAME: Literal["Cora", "CiteSeer", "PubMed"] = "Cora"
LOAD_PATH: str = "models"
MODEL_NAME: Literal["gcn", "gat"] = "gcn"
METHODS: dict[str, Type[Explainer]] = {
    "Saliency Map": SaliencyMap,
    "GNNExplainer": GNNExplainer,
}
RESULTS_PATH: str = "results/images"

# Set seed and define device
set_seed(42)
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


@torch.no_grad()
def main() -> None:
    """
    This is the main function to generate examples.

    Returns:
        None.
    """

    # Load dataset
    dataset: InMemoryDataset = load_data(DATASET_NAME, f"{DATA_PATH}/{DATASET_NAME}")

    # Load model
    model: torch.nn.Module
    model = torch.load(f"{LOAD_PATH}/{DATASET_NAME}_{MODEL_NAME}.pt").to(device)
    model.eval()

    # Pass elements to correct device
    x: torch.Tensor = dataset[0].x.float().to(device)
    edge_index: torch.Tensor = dataset[0].edge_index.long().to(device)
    node_ids: torch.Tensor = torch.arange(x.shape[0]).to(device)

    # Iter over methods
    for method_name, method in METHODS.items():
        # Compute feature maps
        explainer = method(model)
        feature_map: torch.Tensor = explainer.explain(x, edge_index, 0)
        feature_map = (feature_map - feature_map.min()) / (
            feature_map.max() - feature_map.min()
        )

        # Create visualization
        create_visualization(x, edge_index, node_ids, feature_map, method_name)

    return None


@torch.no_grad()
def create_visualization(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    node_ids: torch.Tensor,
    feature_map: torch.Tensor,
    method_name: str,
) -> None:
    """
    This function creates visualizations.

    Args:
        x: Node matrix. Dimensions: [number of nodes, number of node
            features].
        edge_index: Edge index. Dimensions: [2, number of edges].
        node_ids: Node ids. Dimensions: [number of nodes].
        feature_map: Feature map. Dimensions: [number of nodes, number
            of nodes].
        method_name: Method name.

    Returns:
        None.
    """

    # Filter out important ids
    important_node_ids: torch.Tensor = node_ids[feature_map != 0]

    # Create graph
    data = torch_geometric.data.Data(
        x=x, edge_index=edge_index, edge_weight=feature_map
    )
    g = torch_geometric.utils.to_networkx(data)
    g = g.subgraph(important_node_ids.tolist())
    pos = nx.spring_layout(g, seed=42)
    labels = {}
    color_map = []
    alphas = []
    for node in g.nodes():
        # Set the node name as the key and the label as its value
        labels[node] = round(feature_map[node].item(), 2)
        if node != 0:
            color_map.append("#00b4d9")
        else:
            color_map.append("#C6442A")

        alphas.append(float(feature_map[node].item()))

    # Create dir if it doesn't exist
    if not os.path.isdir(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    # Create and save figure
    plt.figure()
    nx.draw_networkx_nodes(
        g, pos=pos, node_color=color_map, node_size=1000, alpha=alphas
    )
    nx.draw_networkx_edges(g, pos=pos, node_size=1000)
    nx.draw_networkx_labels(g, pos=pos, labels=labels)
    plt.axis("off")
    plt.savefig(f"{RESULTS_PATH}/{method_name}.pdf", bbox_inches="tight")
    plt.show()
    plt.close()

    return None
