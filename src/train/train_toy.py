"""
This module contains code to train the toy model.
"""

# Standard libraries
import os

# 3pps
import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy
import torch_geometric
from torch_geometric.data import InMemoryDataset
from tqdm.auto import tqdm

# Own modules
from src.utils import (
    set_seed,
    load_data,
    DATA_PATH,
    LOAD_PATH,
    DatasetName,
)

# set seed and define device
set_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main() -> None:
    """
    This is the function to train the models.

    Raises:
        ValueError: Invalid model_name.
    """

    # Define hyperparameters
    dataset_name: DatasetName = "Cora"
    lr: float = 1e-3
    epochs: int = 100
    model_name: str = "toy_model"

    # Empty nohup file
    open("nohup.out", "w", encoding="utf-8").close()

    # Check device
    print(f"device: {device}")

    # Define name and tensorboard writer
    name: str = f"{dataset_name}_{model_name}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # Define dataset
    dataset: InMemoryDataset = load_data(dataset_name, f"{DATA_PATH}/{dataset_name}")

    # Define model
    model: torch.nn.Module
    model = torch_geometric.nn.GCNConv(
        dataset.num_features, dataset.num_classes, normalize=False
    ).to(device)

    # Define loss and optimizer
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Define metrics
    accuracy: torch.nn.Module = MulticlassAccuracy(dataset.num_classes).to(device)

    # Pass elements to correct device
    x: torch.Tensor = dataset[0].x.float().to(device)
    edge_index: torch.Tensor = dataset[0].edge_index.long().to(device)
    y: torch.Tensor = dataset[0].y.long().to(device)
    train_mask: torch.Tensor = dataset[0].train_mask.to(device)
    val_mask: torch.Tensor = dataset[0].val_mask.to(device)

    # Iter over epochs
    epoch: int
    for epoch in tqdm(range(epochs)):
        # activate train mode
        model.train()

        # compute outputs and loss
        outputs: torch.Tensor = model(x, edge_index)
        loss_value = loss(outputs[train_mask, :], y[train_mask])

        # optimize
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        # writer on tensorboard
        writer.add_scalar("loss/train", loss_value.item(), epoch)
        writer.add_scalar(
            "accuracy/train",
            accuracy(outputs[train_mask, :], y[train_mask]).item(),
            epoch,
        )

        # activate eval mode
        model.eval()

        # Decativate the gradient
        with torch.no_grad():
            # compute outputs
            outputs = model(x, edge_index)

            # write on tensorboard
            writer.add_scalar(
                "accuracy/val",
                accuracy(outputs[val_mask, :], y[val_mask]).item(),
                epoch,
            )

    # create dirs to save model
    if not os.path.exists(f"{LOAD_PATH}"):
        os.makedirs(f"{LOAD_PATH}")

    # save model
    model = model.cpu()
    torch.save(model, f"{LOAD_PATH}/{model_name}.pt")

    return None


if __name__ == "__main__":
    main()
