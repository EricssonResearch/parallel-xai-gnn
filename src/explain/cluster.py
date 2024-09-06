import copy
import os.path as osp
import sys
from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
import torch.utils.data
from torch import Tensor

import torch_geometric.typing
from torch_geometric.data import Data
from torch_geometric.typing import pyg_lib
from torch_geometric.utils import index_sort, narrow, select, sort_edge_index
from torch_geometric.utils.map import map_index
from torch_geometric.utils.sparse import index2ptr, ptr2index


@dataclass
class Partition:
    indptr: Tensor
    index: Tensor
    partptr: Tensor
    node_perm: Tensor
    edge_perm: Tensor
    sparse_format: Literal["csr", "csc"]


class ClusterData(torch.utils.data.Dataset):
    r"""Clusters/partitions a graph data object into multiple subgraphs, as
    motivated by the `"Cluster-GCN: An Efficient Algorithm for Training Deep
    and Large Graph Convolutional Networks"
    <https://arxiv.org/abs/1905.07953>`_ paper.

    .. note::
        The underlying METIS algorithm requires undirected graphs as input.

    Args:
        data (torch_geometric.data.Data): The graph data object.
        num_parts (int): The number of partitions.
        recursive (bool, optional): If set to :obj:`True`, will use multilevel
            recursive bisection instead of multilevel k-way partitioning.
            (default: :obj:`False`)
        save_dir (str, optional): If set, will save the partitioned data to the
            :obj:`save_dir` directory for faster re-use. (default: :obj:`None`)
        log (bool, optional): If set to :obj:`False`, will not log any
            progress. (default: :obj:`True`)
        keep_inter_cluster_edges (bool, optional): If set to :obj:`True`,
            will keep inter-cluster edge connections. (default: :obj:`False`)
        sparse_format (str, optional): The sparse format to use for computing
            partitions. (default: :obj:`"csr"`)
    """

    def __init__(
        self,
        data,
        num_parts: int,
        recursive: bool = False,
        save_dir: Optional[str] = None,
        log: bool = True,
        keep_inter_cluster_edges: bool = False,
        sparse_format: Literal["csr", "csc"] = "csr",
    ):
        assert data.edge_index is not None
        assert sparse_format in ["csr", "csc"]

        self.num_parts = num_parts
        self.recursive = recursive
        self.keep_inter_cluster_edges = keep_inter_cluster_edges
        self.sparse_format = sparse_format

        recursive_str = "_recursive" if recursive else ""
        filename = f"metis_{num_parts}{recursive_str}.pt"
        path = osp.join(save_dir or "", filename)

        if save_dir is not None and osp.exists(path):
            self.partition = torch.load(path)
        else:
            if log:  # pragma: no cover
                print("Computing METIS partitioning...", file=sys.stderr)

            cluster = self._metis(data.edge_index, data.num_nodes)
            self.partition = self._partition(data.edge_index, cluster)

            if save_dir is not None:
                torch.save(self.partition, path)

            if log:  # pragma: no cover
                print("Done!", file=sys.stderr)

        self.data = self._permute_data(data, self.partition)

    def _metis(self, edge_index: Tensor, num_nodes: int) -> Tensor:
        # Computes a node-level partition assignment vector via METIS.
        if self.sparse_format == "csr":  # Calculate CSR representation:
            row, index = sort_edge_index(edge_index, num_nodes=num_nodes)
            indptr = index2ptr(row, size=num_nodes)
        else:  # Calculate CSC representation:
            index, col = sort_edge_index(
                edge_index, num_nodes=num_nodes, sort_by_row=False
            )
            indptr = index2ptr(col, size=num_nodes)

        # Compute METIS partitioning:
        cluster: Optional[Tensor] = None

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            try:
                cluster = torch.ops.torch_sparse.partition(
                    indptr.cpu(),
                    index.cpu(),
                    None,
                    self.num_parts,
                    self.recursive,
                ).to(edge_index.device)
            except (AttributeError, RuntimeError):
                pass

        if cluster is None and torch_geometric.typing.WITH_METIS:
            cluster = pyg_lib.partition.metis(
                indptr.cpu(),
                index.cpu(),
                self.num_parts,
                recursive=self.recursive,
            ).to(edge_index.device)

        if cluster is None:
            raise ImportError(
                f"'{self.__class__.__name__}' requires either "
                f"'pyg-lib' or 'torch-sparse'"
            )

        return cluster

    def _partition(self, edge_index: Tensor, cluster: Tensor) -> Partition:
        # Computes node-level and edge-level permutations and permutes the edge
        # connectivity accordingly:

        # Sort `cluster` and compute boundaries `partptr`:
        cluster, node_perm = index_sort(cluster, max_value=self.num_parts)
        partptr = index2ptr(cluster, size=self.num_parts)

        # Permute `edge_index` based on node permutation:
        edge_perm = torch.arange(edge_index.size(1), device=edge_index.device)
        arange = torch.empty_like(node_perm)
        arange[node_perm] = torch.arange(cluster.numel(), device=cluster.device)
        edge_index = arange[edge_index]

        # Compute final CSR representation:
        (row, col), edge_perm = sort_edge_index(
            edge_index,
            edge_attr=edge_perm,
            num_nodes=cluster.numel(),
            sort_by_row=self.sparse_format == "csr",
        )
        if self.sparse_format == "csr":
            indptr, index = index2ptr(row, size=cluster.numel()), col
        else:
            indptr, index = index2ptr(col, size=cluster.numel()), row

        return Partition(
            indptr, index, partptr, node_perm, edge_perm, self.sparse_format
        )

    def _permute_data(self, data: Data, partition: Partition) -> Data:
        # Permute node-level and edge-level attributes according to the
        # calculated permutations in `Partition`:
        out = copy.copy(data)
        for key, value in data.items():
            if key == "edge_index":
                continue
            elif data.is_node_attr(key):
                cat_dim = data.__cat_dim__(key, value)
                out[key] = select(value, partition.node_perm, dim=cat_dim)
            elif data.is_edge_attr(key):
                cat_dim = data.__cat_dim__(key, value)
                out[key] = select(value, partition.edge_perm, dim=cat_dim)
        out.edge_index = None

        return out

    def __len__(self) -> int:
        return self.partition.partptr.numel() - 1

    def __getitem__(self, idx: int) -> Data:
        node_start = int(self.partition.partptr[idx])
        node_end = int(self.partition.partptr[idx + 1])
        node_length = node_end - node_start

        indptr = self.partition.indptr[node_start : node_end + 1]
        edge_start = int(indptr[0])
        edge_end = int(indptr[-1])
        edge_length = edge_end - edge_start
        indptr = indptr - edge_start

        if self.sparse_format == "csr":
            row = ptr2index(indptr)
            col = self.partition.index[edge_start:edge_end]
            if not self.keep_inter_cluster_edges:
                edge_mask = (col >= node_start) & (col < node_end)
                row = row[edge_mask]
                col = col[edge_mask]
            col -= node_start
        else:
            col = ptr2index(indptr)
            row = self.partition.index[edge_start:edge_end]
            if not self.keep_inter_cluster_edges:
                edge_mask = (row >= node_start) & (row < node_end)
                col = col[edge_mask]
                row = row[edge_mask]
            col -= node_start

        out = copy.copy(self.data)

        for key, value in self.data.items():
            if key == "num_nodes":
                out.num_nodes = node_length
            elif self.data.is_node_attr(key):
                cat_dim = self.data.__cat_dim__(key, value)
                out[key] = narrow(value, cat_dim, node_start, node_length)
            elif self.data.is_edge_attr(key):
                cat_dim = self.data.__cat_dim__(key, value)
                out[key] = narrow(value, cat_dim, edge_start, edge_length)
                if not self.keep_inter_cluster_edges:
                    out[key] = out[key][edge_mask]

        out.edge_index = torch.stack([row, col], dim=0)

        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.num_parts})"
