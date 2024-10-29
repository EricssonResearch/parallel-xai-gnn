# ai libraries
import torch
import numpy as np
from scipy.sparse import lil_matrix, coo_matrix

# other libraries
from typing import Literal, Union


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


class SparseTensor:
    def __init__(self, shape: tuple[int, int]) -> None:
        """
        This method is the constructor of SparseTensor.

        Args:
            shape: shape of the tensor.
        """

        # set attributes
        self.shape = shape
        self.indexes = torch.tensor([]).view(2, 1)
        self.values = torch.tensor([]).view(1)

    def __getitem__(self, indexes: tuple[int, int]) -> torch.Tensor:
        value: torch.Tensor = (
            self.values[self.indexes[0] == indexes[0]]
            & self.values[self.indexes[1] == indexes[1]]
        )

        return value

    def __setitem__(self, indexes: tuple[int, int], value: float) -> None:
        self.indexes = torch.concat(
            (self.indexes, torch.tensor(indexes).view(2, 1)), dim=1
        )
        self.values = torch.concat((self.values))

        return None
