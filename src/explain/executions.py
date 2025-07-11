"""
This module contains the code for XAI executions.
"""

# Standard libraries
import os
import time
import pickle

# 3pps
import torch
from torch_geometric.data import Data
from scipy.sparse import lil_matrix

# Own modules
from src.explain.methods import Explainer
from src.explain.cluster import get_extended_data
from src.explain.sparse import normalize_sparse_matrix
from src.explain.utils import ITERATIONS


@torch.no_grad()
def compute_xai(
    explainer: Explainer,
    num_clusters: int,
    dropout_rate: float,
    data: Data,
    checkpoints_folder_path: str,
    device: torch.device,
) -> tuple[lil_matrix, int, int, float]:
    """
    This function executes the explainability, and compute the execution
    time.

    Args:
        explainer: _description_
        num_clusters: _description_
        dropout_rate: _description_
        data: _description_
        checkpoints_folder_path: _description_
        device: _description_

    Returns:
        _description_
    """

    # Define file path
    checkpoint_path: str = (
        f"{checkpoints_folder_path}/{num_clusters}_{dropout_rate}.pkl"
    )

    # Check if checkpoint already exists
    if not os.path.isfile(checkpoint_path):
        # Init exec time
        exec_time = 0.0

        # Iter over executions
        for _ in range(ITERATIONS):
            # Start time
            start = time.time()

            # Compute xai
            if num_clusters == 1:
                global_feature_maps = original_xai(explainer, data, device=device)
                num_extended_nodes = data.x.shape[0]
                num_extended_edges = data.edge_index.shape[1]
            else:
                (
                    global_feature_maps,
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

        # Save with pickle
        with open(checkpoint_path, "wb") as file_:
            pickle.dump(
                (
                    global_feature_maps,
                    num_extended_nodes,
                    num_extended_edges,
                    exec_time,
                ),
                file_,
            )

    else:
        with open(checkpoint_path, "rb") as file_:
            (
                global_feature_maps,
                num_extended_nodes,
                num_extended_edges,
                exec_time,
            ) = pickle.load(file_)

    return global_feature_maps, num_extended_nodes, num_extended_edges, exec_time


@torch.no_grad()
def original_xai(
    explainer: Explainer,
    data: Data,
    device: torch.device = torch.device("cpu"),
) -> lil_matrix:
    """
    This function computes the original xai, iterating over the nodes
    and computing the explainability for each node (without batches).

    Args:
        explainer: Explainer to use.
        data: data object. Attributes: [node_ids, x, edge_index].
        device: device for torch tensors. Defaults to
            torch.device("cpu").

    Returns:
        Global feature maps. Dimension: [number of nodes,
            number of nodes].
    """

    # create feature maps
    num_nodes: int = data.x.shape[0]
    global_feature_maps: lil_matrix = lil_matrix((num_nodes, num_nodes))

    # pass to right device
    data = data.to(device)

    # iter over node ids
    for node_id in data.node_ids.tolist():
        # compute feature map
        feature_map: torch.Tensor = explainer.explain(data.x, data.edge_index, node_id)

        # add to global feature map
        global_feature_maps[node_id] = feature_map.cpu().numpy()

    # normalize
    global_feature_maps = normalize_sparse_matrix(global_feature_maps)

    return global_feature_maps


@torch.no_grad()
def parallel_xai(
    explainer: Explainer,
    data: Data,
    num_clusters: int,
    num_hops: int = 3,
    dropout_rate: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> tuple[lil_matrix, int, int]:
    """
    This function computes XAI in a parallel way.

    Args:
        explainer: explainer to use.
        data: data object. Attributes: [node_ids, x, edge_index].
        test_mask: test mask tensor. Dimensions: [number of nodes].
        num_clusters: number of clusters.
        num_hops: number of hops of GNN model. Defaults to 3.
        dropout_rate: rate for the dropout in the reconstruction.
            Defaults to 0.0.
        device: device for the tensors. Defaults to cpu.

    Returns:
        Global features maps. Dimensions: [number of nodes,
            number of nodes].
        Number of nodes in extended matrix.
        Number of edges in extended matrix.
    """

    # create feature maps
    num_nodes: int = data.x.shape[0]
    global_feature_maps: lil_matrix = lil_matrix((num_nodes, num_nodes))

    # pass data to the right device
    data = data.to(device)

    # compute extended data
    data_extended: Data = get_extended_data(
        data, num_clusters, num_hops, dropout_rate, device
    )

    # compute number of batches and node ids of extended graph
    num_batches: int = data_extended.batch_indexes.max().item() + 1
    new_node_ids: torch.Tensor = torch.arange(data_extended.x.shape[0]).to(device)

    # compute cluster sizes
    _, cluster_sizes = torch.unique(
        data_extended.cluster_ids[data_extended.batch_indexes != -1],
        return_counts=True,
        sorted=False,
    )
    _, cluster_sizes_extended = torch.unique(
        data_extended.cluster_ids,
        return_counts=True,
        sorted=False,
    )

    # Iter over batches
    for batch_index in range(num_batches):
        # compute xai ids
        xai_ids: torch.Tensor = new_node_ids[data_extended.batch_indexes == batch_index]

        # compute feature map
        feature_map: torch.Tensor = explainer.explain(
            data_extended.x, data_extended.edge_index, xai_ids
        )

        # get cluster sizes and cluster elements
        cluster_elements = -1 * torch.ones_like(cluster_sizes)
        cluster_elements_fill = data_extended.node_ids[
            data_extended.batch_indexes == batch_index
        ]
        cluster_elements[cluster_sizes > batch_index] = cluster_elements_fill

        # compute source ids
        source_ids_tensor: torch.Tensor = torch.repeat_interleave(
            cluster_elements, cluster_sizes_extended
        )
        source_ids: list[int] = source_ids_tensor[feature_map != 0].cpu().tolist()

        # compute neighbor ids
        neighbor_ids: list[int] = (
            data_extended.node_ids[feature_map != 0].cpu().tolist()
        )

        # add feature map
        global_feature_maps[source_ids, neighbor_ids] = (
            feature_map[feature_map != 0].cpu().numpy()
        )

    # Normalize
    global_feature_maps = normalize_sparse_matrix(global_feature_maps)

    return (
        global_feature_maps,
        data_extended.x.shape[0],
        data_extended.edge_index.shape[1],
    )
