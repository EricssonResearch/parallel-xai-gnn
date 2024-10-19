# ai libraries
import torch
from torch_geometric.utils.map import map_index
from torch_geometric.utils.mask import index_to_mask
from torch_geometric.utils.num_nodes import maybe_num_nodes
from scipy.sparse import lil_matrix, coo_matrix

# other libraries
from typing import Optional, Literal, Union


def k_hop_subgraph(
    node_idx: torch.Tensor,
    num_hops: int,
    edge_index: torch.Tensor,
    directed: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function generates a subgraph with k hops.

    Args:
        node_idx: node indexes. Dimensions [number of subset of nodes].
        num_hops: number of hops for the subgraph.
        edge_index: edge index tensor. Dimensions: [2, number of edges].
        directed: If set to True will only include the directed edges
            to the seed nodes. Defaults to True.

    Returns:
        _description_
    """

    num_nodes = maybe_num_nodes(edge_index, None)
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


def subgraph(
    subset: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    num_nodes: Optional[int] = None,
    return_edge_mask: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.

    Args:
        subset (LongTensor, BoolTensor or [int]): The nodes to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        return_edge_mask (bool, optional): If set to :obj:`True`, will return
            the edge mask to filter out additional edge features.
            (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        ...                            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5]])
        >>> edge_attr = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        >>> subset = torch.tensor([3, 4, 5])
        >>> subgraph(subset, edge_index, edge_attr)
        (tensor([[3, 4, 4, 5],
                [4, 3, 5, 4]]),
        tensor([ 7.,  8.,  9., 10.]))

        >>> subgraph(subset, edge_index, edge_attr, return_edge_mask=True)
        (tensor([[3, 4, 4, 5],
                [4, 3, 5, 4]]),
        tensor([ 7.,  8.,  9., 10.]),
        tensor([False, False, False, False, False, False,  True,
                True,  True,  True,  False, False]))
    """

    # get device
    device = edge_index.device

    #
    if isinstance(subset, (list, tuple)):
        subset = torch.tensor(subset, dtype=torch.long, device=device)

    if subset.dtype != torch.bool:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        node_mask = index_to_mask(subset, size=num_nodes)
    else:
        num_nodes = subset.size(0)
        node_mask = subset
        subset = node_mask.nonzero().view(-1)

    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    new_edge_index, _ = map_index(
        edge_index.view(-1),
        subset,
        max_index=num_nodes,
        inclusive=True,
    )
    new_edge_index = new_edge_index.view(2, -1)

    return edge_index, new_edge_index


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
