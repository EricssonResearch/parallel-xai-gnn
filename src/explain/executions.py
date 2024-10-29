# ai learning libraries
import torch
from torch_geometric.data import Data
from scipy.sparse import lil_matrix

# other libraries
from typing import Optional

# own modules
from src.explain.methods import Explainer
from src.explain.cluster import get_extended_data
from src.explain.sparse import normalize_sparse_matrix


def original_xai(
    explainer: Explainer,
    data: Data,
    device: torch.device = torch.device("cpu"),
) -> lil_matrix:
    """
    This function computes the original xai, iterating over the nodes
    and computing the explainability for each node (without bacthes).

    Args:
        explainer: _description_
        data: data object. Attributes: [node_ids, x, edge_index].
        device: device for torch tensors. Defaults to
            torch.device("cpu").

    Returns:
        _description_
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
) -> tuple[lil_matrix, int, int, Optional[torch.Tensor]]:
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
        _description_
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

    # iter over bacthes
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

    # normalize
    global_feature_maps = normalize_sparse_matrix(global_feature_maps)

    return (
        global_feature_maps,
        data_extended.x.shape[0],
        data_extended.edge_index.shape[1],
    )
