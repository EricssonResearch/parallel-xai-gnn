# deep learning libraries
import torch
import torch_geometric
import pandas as pd
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import ClusterData
from scipy.sparse import lil_matrix

# other libraries
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm.auto import tqdm
from typing import Literal, Type

# own modules
from src.train.models import GCN, GAT
from src.utils import set_seed, load_data
from src.explain.methods import Explainer
from src.explain.utils import k_hop_subgraph


def original_xai(
    explainer: Explainer,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    node_ids: torch.Tensor,
    test_mask: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> lil_matrix:
    # create feature maps
    num_nodes: int = x.shape[0]
    global_feature_maps: lil_matrix = lil_matrix((num_nodes, num_nodes))

    # construct Data object
    data = Data(
        node_ids=node_ids,
        x=x,
        edge_index=edge_index,
        test_mask=test_mask,
    ).to(device)

    for node_id in node_ids:
        # compute feature map
        feature_map: torch.Tensor = explainer.explain(data.x, data.edge_index, node_id)

        # add to global feature map
        global_feature_maps[node_id] = feature_map.cpu().numpy()

    return global_feature_maps


def parallel_xai(
    explainer: Explainer,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    node_ids: torch.Tensor,
    test_mask: torch.Tensor,
    num_parts: int,
    num_hops: int = 3,
    device: torch.device = torch.device("cpu"),
) -> lil_matrix:
    """
    This function computes XAI in a parallel way.

    Args:
        explainer: explainer to use.
        x: _description_
        edge_index: _description_
        node_ids: _description_
        test_mask: _description_
        num_parts: _description_
        num_hops:
        device: _description_

    Returns:
        _description_
    """

    # create feature maps
    num_nodes: int = x.shape[0]
    global_feature_maps: lil_matrix = lil_matrix((num_nodes, num_nodes))

    # construct Data object
    data = Data(
        node_ids=node_ids,
        x=x,
        edge_index=edge_index,
        test_mask=test_mask,
    ).to(device)

    # compute partitions
    cluster_data: ClusterData = ClusterData(
        data, num_parts=num_parts, keep_inter_cluster_edges=False
    )

    # load extended batches
    for cluster_id in range(len(cluster_data)):
        # extract data
        cluster = cluster_data[cluster_id]

        # compute reconstructed tensors
        re_ids, re_edge_index, re_edge_index_relabelled = k_hop_subgraph(
            cluster.node_ids, num_hops, data.edge_index
        )

        # compute batch index for each element
        cloned_node_ids: torch.Tensor = ~torch.isin(re_ids, cluster.node_ids)
        batch_indexes = -1 * torch.ones_like(re_ids)
        batch_indexes[~cloned_node_ids] = torch.arange(len(cluster.node_ids)).to(device)

        # concat data objects
        if cluster_id == 0:
            data_extended: Data = Data(
                node_ids=re_ids,
                x=data.x[re_ids, :],
                edge_index=re_edge_index_relabelled,
                batch_indexes=batch_indexes,
                cluster_ids=cluster_id * torch.ones_like(batch_indexes),
            )

        else:
            data_extended = data_extended.concat(
                Data(
                    node_ids=re_ids,
                    x=data.x[re_ids, :],
                    edge_index=re_edge_index_relabelled + data_extended.x.shape[0],
                    batch_indexes=batch_indexes,
                    cluster_ids=cluster_id * torch.ones_like(batch_indexes),
                )
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

    return global_feature_maps
