# ai learning libraries
import torch
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData

# own modules
from src.explain.utils import k_hop_subgraph, subgraph


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
        data: initial data object. Mandaroty attributes: [node_ids, x,
            edge_index].
        num_hops: number of hops for the reconstruction.
        dropout_rate: rate for the dropout.
        device: device for the tensor objects.

    Returns:
        data extended object. Attributes: [node_ids, x, edge_index,
            batch_indexes, cluster_ids].
    """

    # compute partitions
    cluster_data: ClusterData = ClusterData(data, num_parts=num_clusters)

    # load extended batches
    for cluster_id, _ in enumerate(cluster_data):
        # extract data
        cluster = cluster_data[cluster_id]

        # compute reconstructed tensors
        re_ids, re_edge_index, re_edge_index_relabelled = k_hop_subgraph(
            cluster.node_ids, num_hops, data.edge_index
        )

        # reduce reconstruction if dropout rate is higher than 1
        if dropout_rate > 0.0:
            # compute clone node ids
            cloned_node_ids_mask: torch.Tensor = ~torch.isin(re_ids, cluster.node_ids)

            # define mask
            dropout_mask: torch.Tensor = torch.rand(
                cloned_node_ids_mask.sum(), device=device
            )
            dropout_mask = dropout_mask < dropout_rate

            # compute filter ids
            filter_ids: torch.Tensor = re_ids[cloned_node_ids_mask][dropout_mask]

            # filter out non reconstructed nodes
            re_ids = re_ids[~torch.isin(re_ids, filter_ids)]
            re_edge_index, re_edge_index_relabelled = subgraph(re_ids, re_edge_index)

        # compute batch index for each element
        original_node_ids_mask: torch.Tensor = torch.isin(re_ids, cluster.node_ids)
        batch_indexes = -1 * torch.ones_like(re_ids)
        batch_indexes[original_node_ids_mask] = torch.arange(len(cluster.node_ids)).to(
            device
        )

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

    return data_extended
