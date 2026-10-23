"""
This module contains the code for XAI executions.
"""

# Standard libraries
import os
import pickle
import time

# 3pps
import torch
from scipy.sparse import lil_matrix

# Own modules
from src.xai.cluster import get_extended_data
from src.xai.constants import ITERATIONS, XAIType
from src.xai.methods import Explainer
from src.xai.sparse import normalize_sparse_matrix
from src.xai.utils import Data


@torch.no_grad()
def compute_xai(
    explainer: Explainer,
    xai_type: XAIType,
    num_clusters: int | None,
    rate: float | None,
    data: Data,
    checkpoints_folder_path: str,
    device: torch.device,
) -> tuple[lil_matrix, int, int, float]:
    """
    This function executes the explainability, and compute the execution
    time.

    Args:
        explainer: Explainer.
        xai_type: Type of XAI execution. Possible values:
            [original, k_hop, cluster].
        num_clusters: Number of clusters.
        rate: Reconstruction rate.
        data: Data object.
        checkpoints_folder_path: Path to the folder where checkpoints
            are saved.
        device: Device to execute operations (CPU or GPU).

    Returns:
        Global feature maps. Dimensions: [number of nodes,
            number of nodes].
    """

    # Define file path
    if xai_type == "original":
        checkpoint_path = f"{checkpoints_folder_path}/{xai_type}.pkl"
    elif xai_type == "cluster":
        checkpoint_path = (
            f"{checkpoints_folder_path}/{xai_type}_{num_clusters}_{rate}.pkl"
        )

    # Check if checkpoint already exists
    if not os.path.isfile(checkpoint_path):
        # Init exec time
        exec_time = 0.0

        # Iter over executions
        for _ in range(ITERATIONS):
            # Start time
            torch.cuda.synchronize()
            start = time.time()

            # Compute xai
            if xai_type == "original":
                feature_maps = original_xai(explainer, data, device=device)
                num_extended_nodes = data.x.shape[0]
                num_extended_edges = data.edge_index.shape[1]

            else:
                # Raise error if clusters or drop rate are not
                # initialized
                if (num_clusters is None) or (rate is None):
                    raise ValueError(
                        "'num_cluster' and 'dropout_rate' arguments must be "
                        "initialized for 'cluster' explainability"
                    )

                # Get feature maps
                (
                    feature_maps,
                    num_extended_nodes,
                    num_extended_edges,
                ) = parallel_xai(
                    explainer,
                    data,
                    num_clusters,
                    rate,
                    device=device,
                )

            # Compute execution time
            torch.cuda.synchronize()
            exec_time += time.time() - start

        # Divide between executions
        exec_time /= ITERATIONS

        # Save with pickle
        with open(checkpoint_path, "wb") as file_:
            pickle.dump(
                (
                    feature_maps,
                    num_extended_nodes,
                    num_extended_edges,
                    exec_time,
                ),
                file_,
            )

    else:
        with open(checkpoint_path, "rb") as file_:
            (
                feature_maps,
                num_extended_nodes,
                num_extended_edges,
                exec_time,
            ) = pickle.load(file_)

    return feature_maps, num_extended_nodes, num_extended_edges, exec_time


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
    rate: float,
    num_hops: int = 2,
    device: torch.device = torch.device("cpu"),
) -> tuple[lil_matrix, int, int]:
    """
    This function computes XAI in a parallel way.

    Args:
        explainer: explainer to use.
        data: data object. Attributes: [node_ids, x, edge_index].
        num_clusters: number of clusters.
        rate: Reconstruction rate.
        num_hops: number of hops of GNN model. Defaults to 3.
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
    data_extended: Data = get_extended_data(data, num_clusters, num_hops, rate, device)

    # pass data to the right device
    data.edge_index = data.edge_index.long()
    data_extended.edge_index = data_extended.edge_index.long()

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
    global_feature_maps_lil = normalize_sparse_matrix(global_feature_maps)

    return (
        global_feature_maps_lil,
        data_extended.x.shape[0],
        data_extended.edge_index.shape[1],
    )


def get_k_hop_batches_node_ids(
    node_ids: torch.Tensor, num_batches: int
) -> list[torch.Tensor]:
    """
    This function gets a list with the node_ids.

    Args:
        node_ids: Node ids in the dataset.
        num_batches: Number of batches.

    Returns:
        List of node ids for each batch.
    """

    # Get number of batches
    batch_size: int = node_ids.shape[0] // num_batches

    # Iter over
    batches_ids: list[torch.Tensor] = []
    for i in range(num_batches):
        # Compute batch
        batch_node_ids: torch.Tensor
        if (i + 1) == num_batches:
            batch_node_ids = node_ids[i * batch_size :]
        else:
            batch_node_ids = node_ids[i * batch_size : i * batch_size + batch_size]

        # Add batch node ids to list
        batches_ids.append(batch_node_ids)

    return batches_ids
