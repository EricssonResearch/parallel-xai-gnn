"""
This module contains the code to create clusters.
"""

# 3pps
import torch
from torch_geometric.loader import ClusterData

# Own modules
from src.explain.utils import Data, k_hop_subgraph, subgraph


def get_extended_data(
    data: Data,
    num_clusters: int,
    num_hops: int,
    dropout_rate: float,
    device: torch.device,
) -> Data:
    """
    This function creates the extended data object, where the graph
    dataset is divided with METIS, and then the borders are
    reconstructed.

    Args:
        data: initial data object. Mandatory attributes: [node_ids, x,
            edge_index].
        xai_type: Explainability type.
        num_clusters: Number of clusters to use to get extended data.
        num_hops: number of hops for the reconstruction.
        dropout_rate: rate for the dropout.
        device: device for the tensor objects.

    Returns:
        data extended object. Attributes: [node_ids, x, edge_index,
            batch_indexes, cluster_ids].
    """

    # Compute partitions
    cluster_data: ClusterData = ClusterData(data, num_parts=num_clusters, log=False)

    # Load extended batches
    data_extended_node_ids: list[torch.Tensor] = []
    data_extended_x: list[torch.Tensor] = []
    data_extended_edge_index: list[torch.Tensor] = []
    data_extended_batch_indexes: list[torch.Tensor] = []
    data_extended_cluster_ids: list[torch.Tensor] = []
    offset: int = 0
    for cluster_id in range(len((cluster_data))):
        # Extract data
        cluster = cluster_data[cluster_id]

        # Compute reconstructed tensors
        re_ids, re_edge_index, re_edge_index_relabelled = k_hop_subgraph(
            cluster.node_ids, num_hops, data.edge_index
        )

        # Reduce reconstruction if dropout rate is higher than 1
        if dropout_rate > 0.0:
            # Compute clone node ids
            cloned_node_ids_mask: torch.Tensor = ~torch.isin(re_ids, cluster.node_ids)

            # Define mask
            dropout_mask: torch.Tensor = torch.rand(
                int(cloned_node_ids_mask.sum().item()), device=device
            )
            dropout_mask = dropout_mask < dropout_rate

            # Compute filter ids
            filter_ids: torch.Tensor = re_ids[cloned_node_ids_mask][dropout_mask]

            # Filter out non reconstructed nodes
            re_ids = re_ids[~torch.isin(re_ids, filter_ids)]
            re_edge_index, re_edge_index_relabelled = subgraph(re_ids, re_edge_index)

        # Compute batch index for each element
        original_node_ids_mask: torch.Tensor = torch.isin(re_ids, cluster.node_ids)
        batch_indexes = -1 * torch.ones_like(re_ids)
        batch_indexes[original_node_ids_mask] = torch.arange(len(cluster.node_ids)).to(
            device
        )
        
        # Add to the list
        data_extended_node_ids.append(re_ids)
        data_extended_x.append(data.x[re_ids, :])
        data_extended_edge_index.append(re_edge_index_relabelled + offset)
        data_extended_batch_indexes.append(batch_indexes)
        data_extended_cluster_ids.append(cluster_id * torch.ones_like(batch_indexes))
        offset += re_ids.shape[0]
    
    # Build data extended
    data_extended: Data = Data(
        node_ids=torch.concat(data_extended_node_ids, dim=0),
        x=torch.concat(data_extended_x, dim=0),
        edge_index=torch.concat(data_extended_edge_index, dim=1),
        batch_indexes=torch.concat(data_extended_batch_indexes, dim=0),
        cluster_ids=torch.concat(data_extended_cluster_ids, dim=0),
    )

    return data_extended


def k_hop_data(data: Data) -> list[Data]:
    """
    This function creates right data format for k_hop parallel
    execution.

    Args:
        data: Data object with all the data of the dataset.

    Returns:
        List of Data objects. Only one node in each object.
    """

    # Init data list
    data_list: list[Data] = []

    # Iter over
    for node_id in data.node_ids:
        data_list.append(
            Data(
                x=data.x[node_id, :].view(1, -1),
                node_ids=torch.tensor([node_id]).to(data.x.device),
            )
        )

    return data_list
