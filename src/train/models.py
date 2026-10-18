"""
This module contains the models.
"""

# 3pps
import torch
import torch_geometric


class GCN(torch.nn.Module):
    """
    This class defines a model with two GCN layers.

    Attributes:
        conv1: first GNN.
        relu: non linearity function.
        dropout: dropout layer.
        conv2: second GNN.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        drop_rate: float = 0.5,
    ) -> None:
        """
        This method is the constructor for the GCN class.

        Args:
            input_dim: Number of node features.
            output_dim: Number of possible classes.
            hidden_dim: Size between the layers.
            drop_rate: Dropout rate for the model.

        Returns:
            None.
        """

        # Call superclass constructor
        super().__init__()

        # Define layers
        self.gnns = torch_geometric.nn.Sequential(
            "x, edge_index",
            [
                (
                    torch_geometric.nn.GCNConv(input_dim, hidden_dim, normalize=False),
                    "x, edge_index -> x",
                ),
                torch.nn.ReLU(),
                torch.nn.Dropout(drop_rate),
                (
                    torch_geometric.nn.GCNConv(hidden_dim, output_dim, normalize=False),
                    "x, edge_index -> x",
                ),
            ],
        )

        return None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        This method defines the forward pass.

        Args:
            x: Node matrix. Dimensions: [batch size, input size].
            edge_index: Edge index tensor that represents the adj
                matrix. Dimensions: [2, number of edges].

        Returns:
            Predictions of the classes. Dimensions: [batch size,
                output size].
        """

        # Compute outputs
        outputs: torch.Tensor = self.gnns(x, edge_index)

        return outputs


class GAT(torch.nn.Module):
    """
    This class defines a model with two GAT layers.

    Attributes:
        conv1: first GNN
        relu: non linearity function
        dropout: dropout layer
        conv2: second GNN
    """

    # Define attributes
    gnn_backbone: torch.nn.Module
    mlp: torch.nn.Module

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        drop_rate: float = 0.5,
    ) -> None:
        """
        This method is the constructor for the class.

        Args:
            input_dim: Number of node features.
            output_dim: Number of possible classes.
            hidden_dim: Size between the layers.
            drop_rate: Dropout rate for the model.

        Returns:
            None.
        """

        # Call superclass constructor
        super().__init__()

        # Define layers
        self.gnns = torch_geometric.nn.Sequential(
            "x, edge_index",
            [
                (
                    torch_geometric.nn.GATv2Conv(input_dim, hidden_dim),
                    "x, edge_index -> x",
                ),
                torch.nn.ReLU(),
                torch.nn.Dropout(drop_rate),
                (
                    torch_geometric.nn.GATv2Conv(hidden_dim, output_dim),
                    "x, edge_index -> x",
                ),
            ],
        )

        return None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        This method defines the forward pass.

        Args:
            x: Node matrix. Dimensions: [batch size, input size].
            edge_index: Edge index tensor that represents the adj
                matrix. Dimensions: [2, number of edges].

        Returns:
            Predictions of the classes. Dimensions: [batch size,
                output size].
        """

        # Compute outputs
        outputs: torch.Tensor = self.gnns(x, edge_index)

        return outputs
