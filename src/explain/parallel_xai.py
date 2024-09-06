# deep learning libraries
import torch
import torch_geometric
import pandas as pd
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import subgraph
from scipy.sparse import lil_matrix

# other libraries
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm.auto import tqdm
from typing import Literal, List, Dict, Type, Tuple

# own modules
from src.train.models import GCN, GAT
from src.utils import set_seed, load_data
from src.explain.methods import Explainer
from src.explain.cluster import ClusterData


def parallel_xai(
    explainer: Explainer,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    node_ids: torch.Tensor,
    test_mask: torch.Tensor,
    num_parts: int,
    num_hops: int = 3,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
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
    feature_maps: lil_matrix = lil_matrix((num_nodes, num_nodes))

    # construct Data object
    data = Data(
        node_ids=node_ids,
        x=x,
        edge_index=edge_index,
        test_mask=test_mask,
    )

    # compute partitions
    cluster_data: ClusterData = ClusterData(
        data, num_parts=num_parts, keep_inter_cluster_edges=False
    )

    # load extended batches
    data_extended: Optional[Data] = None
    for i in range(len(cluster_data)):
        # extract data
        batch = cluster_data[i]

        # change source of edge_index to global ordering
        # batch.edge_index[0] = batch.node_ids[batch.edge_index[0]]
        # out_ids: torch.Tensor = batch.node_ids[batch.edge_index[0, !torch.isin(batch.edge_index[1], batch.edge_index[0])]]

        # # change source of edge_index to global ordering
        # batch.edge_index[0] = batch.node_ids[batch.edge_index[0]]
        # out_ids: torch.Tensor = batch.node_ids[batch.edge_index[0, ~torch.isin(batch.edge_index[1], batch.edge_index[0])]]

        # # get extended batch
        # extended_batch_node_ids, _, _, _ = torch_geometric.utils.k_hop_subgraph(
        #     out_ids, num_hops, edge_index, relabel_nodes=True
        # )
        # batch_node_ids = torch.unique(torch.concat(batch.node_ids, extended_batch_node_ids))
        # batch_extended = data.subgraph(extended_batch_node_ids)

        # get extended batch
        extended_batch_node_ids, _, _, _ = torch_geometric.utils.k_hop_subgraph(
            batch.node_ids, num_hops, edge_index, relabel_nodes=True
        )
        batch_extended = data.subgraph(extended_batch_node_ids)

        # save the batch to compute node ids
        batch_extended.xai_batch_indexes = -1 * torch.ones_like(
            batch_extended.x[:, 0], dtype=torch.long
        )
        batch_extended.xai_batch_indexes[batch_extended.test_mask] = torch.arange(
            batch_extended.test_mask.sum()
        )

        # define cluster id
        batch_extended.cluster_id = i * torch.ones_like(batch_extended.x[:, 0]).int()
        i += 1

        # append batch
        if data_extended is None:
            data_extended = batch_extended
        else:
            data_extended = data_extended.concat(batch_extended)

    # get number of batches for xai
    num_xai_batches: int = data_extended.xai_batch_indexes.max().item()

    # get extended node ids and cluster ids
    node_ids_extended: torch.Tensor = torch.arange(data_extended.x.shape[0]).to(device)

    # pass elements to correct device
    x: torch.Tensor = data_extended.x.to(device)
    edge_index: torch.Tensor = data_extended.edge_index.to(device)
    xai_batch_indexes: torch.Tensor = data_extended.xai_batch_indexes.to(device)

    node_ids_list = data_extended.node_ids.detach().cpu().numpy().tolist()
    cluster_ids_list = data_extended.cluster_id.detach().numpy().tolist()

    # compute xai
    for batch_index in range(num_xai_batches):
        # compute xai ids
        xai_ids: torch.Tensor = node_ids_extended[
            data_extended.xai_batch_indexes == batch_index
        ]

        # compute feature map
        feature_map: torch.Tensor = explainer.explain(x, edge_index, xai_ids)

        # add to feature maps
        feature_maps[cluster_ids_list, node_ids_list] = (
            feature_map.detach().cpu().numpy()
        )

    return feature_maps
