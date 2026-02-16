"""
This module is to create examples visualizations.
"""

# Standard libraries
import os
from typing import Literal, Type

# 3pps
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import k_hop_subgraph
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Own modules
from src.utils import set_seed, load_data
from src.explain.utils import subgraph
from src.explain.methods import Explainer, SaliencyMap, GNNExplainer
from src.explain.cluster import get_extended_data

# Static variables
DATA_PATH: str = "data"
DATASET_NAME: Literal["Cora", "CiteSeer", "PubMed"] = "Cora"
LOAD_PATH: str = "models"
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
    model = torch.load(f"{LOAD_PATH}/toy_model.pt", weights_only=False).to(device)
    model.eval()

    # Pass elements to correct device
    x: torch.Tensor = dataset[0].x.float().to(device)
    edge_index: torch.Tensor = dataset[0].edge_index.long().to(device)

    # Get small example
    subset, edge_index, _, _ = k_hop_subgraph(
        0, num_hops=2, edge_index=edge_index, relabel_nodes=True
    )
    x = x[subset, :]
    node_ids: torch.Tensor = torch.arange(x.shape[0]).to(device)
    data = Data(x=x, edge_index=edge_index, node_ids=node_ids)
    data_extended = get_extended_data(data, 2, 1, 0.0, device)

    # Update data extended
    data_extended = get_updated_data(data_extended)

    # Create visualizations
    # draw_original(data)
    # draw_clusters(data, data_extended)
    # full_reconstruction(data, data_extended)
    draw_original_xai(data, model, torch.tensor([0]))
    draw_original_xai(data, model, torch.tensor([5]))
    draw_original_xai(data, model, torch.tensor([0, 5]))
    draw_cluster_xai(data, data_extended, model, torch.tensor([0, 5]))

    return None


@torch.no_grad()
def draw_original(
    data,
) -> None:
    """
    This function creates visualizations.

    Args:
        data: Data original object.

    Returns:
        None.
    """

    # Create graph
    g = torch_geometric.utils.to_networkx(data)
    pos = nx.spring_layout(g, seed=42)

    labels = {}
    color_map = []
    alphas = []
    for node in g.nodes():
        # Set the node name as the key and the label as its value
        labels[node] = data.node_ids[node].item()
        color_map.append("#00b4d9")

        # Alpha
        alphas.append(1.0)

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
    plt.savefig(f"{RESULTS_PATH}/original.pdf", bbox_inches="tight")
    plt.show()
    plt.close()

    return None


@torch.no_grad()
def draw_clusters(
    data,
    data_extended,
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

    # Create graph
    g = torch_geometric.utils.to_networkx(data)
    pos = nx.spring_layout(g, seed=42)

    # Draw the complete graph
    data = data.clone()
    data.edge_index = data_extended.edge_index[
        :,
        (data_extended.edge_index[0] < data.x.shape[0])
        & (data_extended.edge_index[1] < data.x.shape[0]),
    ]
    g = torch_geometric.utils.to_networkx(data)
    pos = nx.spring_layout(g, seed=42, pos=pos, fixed=data.node_ids.tolist())

    labels = {}
    color_map = []
    alphas = []
    for node in g.nodes():
        # Set the node name as the key and the label as its value
        labels[node] = data_extended.node_ids[node].item()
        if data_extended.cluster_ids[node] != 0:
            color_map.append("#00b4d9")
        else:
            color_map.append("#C6442A")

        # Alpha
        alphas.append(1.0)

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
    plt.savefig(f"{RESULTS_PATH}/clusters.pdf", bbox_inches="tight")
    plt.show()
    plt.close()

    return None


@torch.no_grad()
def full_reconstruction(
    data,
    data_extended,
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

    # Create graph
    g = torch_geometric.utils.to_networkx(data)
    pos = nx.spring_layout(g, seed=42)

    # Draw the complete graph
    g = torch_geometric.utils.to_networkx(data_extended)
    pos = nx.spring_layout(g, seed=42, pos=pos, fixed=data.node_ids.tolist())

    # Update positions
    pos[10] = np.array([0.1, -0.1])
    pos[11] = np.array([-0.54, 0.56])

    labels = {}
    color_map = []
    alphas = []
    for node in g.nodes():
        # Set the node name as the key and the label as its value
        labels[node] = data_extended.node_ids[node].item()
        if data_extended.batch_indexes[node] == -1:
            color_map.append("#565b5c")
        else:
            if data_extended.cluster_ids[node] != 0:
                color_map.append("#00b4d9")
            else:
                color_map.append("#C6442A")

        # Alpha
        alphas.append(1.0)

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
    plt.savefig(f"{RESULTS_PATH}/full_reconstruction.pdf", bbox_inches="tight")
    plt.show()
    plt.close()

    return None


@torch.no_grad()
def draw_original_xai(
    data: Data, model: torch.nn.Module, node_ids: torch.Tensor
) -> None:
    """
    This function creates visualizations.

    Args:
        data: Data object.

    Returns:
        None.
    """

    explainer = SaliencyMap(model)
    feature_map = explainer.explain(data.x, data.edge_index, node_ids)
    feature_map = (feature_map - torch.amin(feature_map)) / (
        torch.amax(feature_map) - torch.amin(feature_map)
    )

    # Create graph
    g = torch_geometric.utils.to_networkx(data)
    pos = nx.spring_layout(g, seed=42)

    labels = {}
    color_map = []
    alphas = []
    for node in g.nodes():
        # Set the node name as the key and the label as its value
        labels[node] = round(feature_map[node].item(), 2)
        if data.node_ids[node].cpu() in node_ids:
            color_map.append("#b100cd")
        else:
            color_map.append("#00b4d9")

        # Add alpha to node
        alphas.append(float(feature_map[node].item()) * 0.8 + 0.2)

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
    plt.savefig(
        f"{RESULTS_PATH}/original_xai_{node_ids.tolist()}.pdf", bbox_inches="tight"
    )
    plt.show()
    plt.close()

    return None


@torch.no_grad()
def draw_cluster_xai(
    data: Data, data_extended: Data, model: torch.nn.Module, node_ids: torch.Tensor
) -> None:
    """
    This function creates visualizations.

    Args:
        data: Data object.

    Returns:
        None.
    """

    # Get explanations
    explainer = SaliencyMap(model)
    feature_map = explainer.explain(data_extended.x, data_extended.edge_index, node_ids)
    feature_map[data_extended.cluster_ids == 0] = (
        feature_map[data_extended.cluster_ids == 0]
        - torch.amin(feature_map[data_extended.cluster_ids == 0])
    ) / (
        torch.amax(feature_map[data_extended.cluster_ids == 0])
        - torch.amin(feature_map[data_extended.cluster_ids == 0])
    )
    feature_map[data_extended.cluster_ids == 1] = (
        feature_map[data_extended.cluster_ids == 1]
        - torch.amin(feature_map[data_extended.cluster_ids == 1])
    ) / (
        torch.amax(feature_map[data_extended.cluster_ids == 1])
        - torch.amin(feature_map[data_extended.cluster_ids == 1])
    )

    # Create graph
    g = torch_geometric.utils.to_networkx(data)
    pos = nx.spring_layout(g, seed=42)

    # Draw the complete graph
    g = torch_geometric.utils.to_networkx(data_extended)
    pos = nx.spring_layout(g, seed=42, pos=pos, fixed=data.node_ids.tolist())

    # Update positions
    pos[10] = np.array([0.1, -0.1])
    pos[11] = np.array([-0.54, 0.56])

    labels = {}
    color_map = []
    alphas = []
    for node in g.nodes():
        # Set the node name as the key and the label as its value
        labels[node] = round(feature_map[node].item(), 2)
        if node in node_ids:
            color_map.append("#b100cd")
        else:
            if data_extended.batch_indexes[node] == -1:
                color_map.append("#565b5c")
            else:
                if data_extended.cluster_ids[node] != 0:
                    color_map.append("#00b4d9")
                else:
                    color_map.append("#C6442A")

        # Add alpha to node
        alphas.append(float(feature_map[node].item()) * 0.8 + 0.2)

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
    plt.savefig(
        f"{RESULTS_PATH}/cluster_xai_{node_ids.tolist()}.pdf", bbox_inches="tight"
    )
    plt.show()
    plt.close()

    return None


@torch.no_grad()
def create_visualization(
    data,
    data_extended,
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
    # important_node_ids: torch.Tensor = node_ids

    data_extended = get_updated_data(data_extended)

    # Create graph
    g = torch_geometric.utils.to_networkx(data)
    pos = nx.spring_layout(g, seed=42)

    # Draw the complete graph
    g = torch_geometric.utils.to_networkx(data_extended)
    pos = nx.spring_layout(g, seed=42, pos=pos, fixed=data.node_ids.tolist())

    # Update positions
    pos[10] = np.array([0.1, -0.1])
    pos[11] = np.array([-0.54, 0.56])

    labels = {}
    color_map = []
    alphas = []
    for node in g.nodes():
        # Set the node name as the key and the label as its value
        labels[node] = round(feature_map[node].item(), 2)
        if data_extended.batch_indexes[node] == -1:
            color_map.append("#565b5c")
        else:
            if data_extended.cluster_ids[node] != 0:
                color_map.append("#00b4d9")
            else:
                color_map.append("#C6442A")

        # alphas.append(float(feature_map[node].item()))
        alphas.append(1.0)

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
    plt.savefig(f"{RESULTS_PATH}/example_extended.pdf", bbox_inches="tight")
    plt.show()
    plt.close()

    return None


def get_updated_data(data_extended: Data) -> Data:
    """
    This functions updated the data_extended object so the node ids are
    the same as before the reconstruction of the borders.

    Args:
        data_extended: Data with the reconstructed borders.

    Returns:
        Data with the node ids updated.
    """

    # Get updated node ids
    update_node_ids = torch.arange(
        data_extended.node_ids.shape[0], device=data_extended.node_ids.device
    )

    _, sort_indexes = torch.sort(
        data_extended.node_ids[data_extended.batch_indexes != -1]
    )
    update_node_ids = torch.concat(
        (
            update_node_ids[data_extended.batch_indexes != -1][sort_indexes],
            update_node_ids[data_extended.batch_indexes == -1],
        ),
        dim=0,
    )

    # Update edge index
    _, data_extended.edge_index = subgraph(update_node_ids, data_extended.edge_index)

    # Update x, cluster_ids and batch_indexes
    data_extended.x = data_extended.x[update_node_ids]
    data_extended.cluster_ids = data_extended.cluster_ids[update_node_ids]
    data_extended.batch_indexes = data_extended.batch_indexes[update_node_ids]
    data_extended.node_ids = data_extended.node_ids[update_node_ids]

    return data_extended
