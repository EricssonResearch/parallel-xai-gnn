# ai libraries
import torch
from torch_geometric.utils.num_nodes import maybe_num_nodes
from scipy.sparse import lil_matrix, coo_matrix

# other libraries
from typing import Optional, Literal


def k_hop_subgraph(
    node_idx: int | list[int] | torch.Tensor,
    num_hops: int,
    edge_index: torch.Tensor,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = "source_to_target",
    directed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    _summary_

    Args:
        node_idx: _description_
        num_hops: _description_
        edge_index: _description_
        relabel_nodes: _description_. Defaults to False.
        num_nodes: _description_. Defaults to None.
        flow: _description_. Defaults to 'source_to_target'.
        directed: _description_. Defaults to False.

    Returns:
        _description_
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ["source_to_target", "target_to_source"]
    if flow == "target_to_source":
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[: node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True

    if not directed:
        edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    node_idx = row.new_full((num_nodes,), -1)
    node_idx[subset] = torch.arange(subset.size(0), device=row.device)
    new_edge_index = node_idx[edge_index]

    return subset, edge_index, new_edge_index


def normalize_sparse_matrix(
    sparse_matrix: lil_matrix, return_type: Literal["lil", "coo"] = "lil"
) -> lil_matrix | coo_matrix:
    """
    This function normalized a sparse matrix. The normalization is
    done by row.

    Args:
        sparse_matrix: sparse matrix in lil format. Dimensions:
            [number of rows, number of columns].

    Raises:
        ValueError: Invalid return type value.

    Returns:
        sparse matrix normalized.
    """

    # transform to coo format
    sparse_matrix_coo: coo_matrix = sparse_matrix.tocoo()

    # sort coo matrix by row indexes
    if np.any(sparse_matrix_coo.row[:-1] > sparse_matrix_coo.row[1:]):
        row_indexes: np.ndarray = np.argsort(sparse_matrix_coo.row)
        sparse_matrix_coo.row = sparse_matrix_coo.row[row_indexes]
        sparse_matrix_coo.col = sparse_matrix_coo.col[row_indexes]
        sparse_matrix_coo.data = sparse_matrix_coo.data[row_indexes]

    # get min and max values
    min_: np.ndarray = sparse_matrix_coo.min(axis=1).toarray()
    max_: np.ndarray = sparse_matrix_coo.max(axis=1).toarray()
    min_ = min_[max_ != 0]
    max_ = max_[max_ != 0]

    # compute counts
    counts: np.ndarray = np.unique(sparse_matrix_coo.row, return_counts=True)[1]

    # compute repeated
    min_repeated = np.repeat(min_, counts)
    max_repeated = np.repeat(max_, counts)

    # normalize
    sparse_matrix_coo.data = (sparse_matrix_coo.data - min_repeated) / (
        max_repeated - min_repeated
    )

    # change return type
    if return_type == "lil":
        sparse_matrix = sparse_matrix_coo.tolil()
    elif return_type == "coo":
        pass
    else:
        raise ValueError("Invalid return type value")

    return sparse_matrix_coo
