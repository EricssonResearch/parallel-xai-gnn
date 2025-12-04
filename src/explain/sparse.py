"""
This module contains the code to normalize sparse tensors.
"""

# Standard libraries
from typing import Literal, Union

# 3pps
import numpy as np
from scipy.sparse import lil_matrix, coo_matrix


def normalize_sparse_matrix(
    sparse_matrix: lil_matrix, return_type: Literal["lil", "coo"] = "lil"
) -> Union[lil_matrix, coo_matrix]:
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


# class SparseTensor:
#     """
#     This class is a SparseTensor to save the global features maps,
#     which is expcted to be a sparse matrix with dimensions
#     [number of nodes, number of nodes]. Therefore, the sparse matrix
#     implemented here is just a 2D SparseTensor based on PyTorch.

#     Attr:
#         shape: shape of the SparseTensor.
#         indexes: tensor with the indexes of the non zero elements.
#             Dimensions: [2, number of non zero elements].
#         values: tensor with the values of the non zero elements.
#             Dimensions: [number of non zero elements].
#     """

#     def __init__(self, shape: tuple[int, int]) -> None:
#         """
#         This method is the constructor of SparseTensor.

#         Args:
#             shape: shape of the tensor.
#         """

#         # set attributes
#         self.shape = shape
#         self.indexes = torch.tensor([]).view(2, 1)
#         self.values = torch.tensor([]).view(1)

#     def __getitem__(
#         self, indexes: tuple[int, int] | tuple[torch.Tensor, torch.Tensor]
#     ) -> torch.Tensor:
#         """
#         This method implements the getitem to be able to access

#         Args:
#             indexes: _description_

#         Returns:
#             _description_
#         """

#         value: torch.Tensor = (
#             self.values[self.indexes[0] == indexes[0]]
#             & self.values[self.indexes[1] == indexes[1]]
#         )

#         return value

#     def __setitem__(
#         self, indexes: tuple[int, int] | torch.Tensor, value: float | torch.Tensor
#     ) -> None:
#         """
#         _summary_

#         Args:
#             indexes: _description_
#             value: _description_

#         Returns:
#             _description_
#         """

#         self.indexes = torch.concat(
#             (self.indexes, torch.tensor(indexes).view(2, 1)), dim=1
#         )
#         self.values = torch.concat((self.values))

#         return None
