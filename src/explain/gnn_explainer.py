"""
This module implements a custom version of the GNNExplainer algorithm. 
This custom version initializes the mask to ones instead of random, 
so we can compare if the original explainability and the parallel one 
are exactly the same.
"""

# Standard libraries
from typing import Union, Optional

# 3pps
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import MaskType, ModelMode


class GNNExplainerAlgorithm(ExplainerAlgorithm):
    coeffs = {
        "edge_size": 0.005,
        "edge_reduction": "sum",
        "node_feat_size": 1.0,
        "node_feat_reduction": "sum",
        "edge_ent": 1.0,
        "node_feat_ent": 0.1,
        "EPS": 1e-15,
    }

    def __init__(self, epochs: int = 100, lr: float = 0.01, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(
                f"Heterogeneous graphs not yet supported in "
                f"'{self.__class__.__name__}'"
            )

        self._train(model, x, edge_index, target=target, index=index, **kwargs)

        node_mask = self._post_process_mask(
            self.node_mask,
            self.hard_node_mask,
            apply_sigmoid=True,
        )
        edge_mask = self._post_process_mask(
            self.edge_mask,
            self.hard_edge_mask,
            apply_sigmoid=True,
        )

        self._clean_model(model)

        return Explanation(node_mask=node_mask, edge_mask=edge_mask)

    def supports(self) -> bool:
        return True

    def _train(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        target: torch.Tensor,
        index: int | torch.Tensor | None = None,
        **kwargs,
    ):
        self._initialize_masks(x, edge_index)

        parameters = []
        if self.node_mask is not None:
            parameters.append(self.node_mask)
        if self.edge_mask is not None:
            set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            parameters.append(self.edge_mask)

        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for i in range(self.epochs):
            optimizer.zero_grad()

            h = x if self.node_mask is None else x * self.node_mask.sigmoid()
            y_hat, y = model(h, edge_index, **kwargs), target

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y)

            loss.backward()
            optimizer.step()

            # In the first iteration, we collect the nodes and edges that are
            # involved into making the prediction. These are all the nodes and
            # edges with gradient != 0 (without regularization applied).
            if i == 0 and self.node_mask is not None:
                if self.node_mask.grad is None:
                    raise ValueError(
                        "Could not compute gradients for node "
                        "features. Please make sure that node "
                        "features are used inside the model or "
                        "disable it via `node_mask_type=None`."
                    )
                self.hard_node_mask = self.node_mask.grad != 0.0
            if i == 0 and self.edge_mask is not None:
                if self.edge_mask.grad is None:
                    raise ValueError(
                        "Could not compute gradients for edges. "
                        "Please make sure that edges are used "
                        "via message passing inside the model or "
                        "disable it via `edge_mask_type=None`."
                    )
                self.hard_edge_mask = self.edge_mask.grad != 0.0

    def _initialize_masks(self, x: torch.Tensor, edge_index: torch.Tensor):
        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type

        device = x.device
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        if node_mask_type is None:
            self.node_mask = None
        elif node_mask_type == MaskType.object:
            self.node_mask = torch.nn.Parameter(torch.ones(N, 1, device=device) * std)
        elif node_mask_type == MaskType.attributes:
            self.node_mask = torch.nn.Parameter(torch.ones(N, F, device=device) * std)
        elif node_mask_type == MaskType.common_attributes:
            self.node_mask = torch.nn.Parameter(torch.ones(1, F, device=device) * std)
        else:
            assert False

        if edge_mask_type is None:
            self.edge_mask = None
        elif edge_mask_type == MaskType.object:
            std = torch.nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * N))
            self.edge_mask = torch.nn.Parameter(torch.ones(E, device=device) * std)
        else:
            assert False

    def _loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = F.binary_cross_entropy_with_logits(y_hat, y.float(), reduction="sum")
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = F.cross_entropy(y_hat, y, reduction="sum")
        elif self.model_config.mode == ModelMode.regression:
            loss = F.mse_loss(y_hat, y, reduction="sum")
        else:
            assert False

        # if self.hard_edge_mask is not None:
        #     assert self.edge_mask is not None
        #     m = self.edge_mask[self.hard_edge_mask].sigmoid()
        #     edge_reduce = getattr(torch, self.coeffs["edge_reduction"])
        #     loss = loss + self.coeffs["edge_size"] * edge_reduce(m)
        #     ent = -m * torch.log(m + self.coeffs["EPS"]) - (1 - m) * torch.log(
        #         1 - m + self.coeffs["EPS"]
        #     )
        #     loss = loss + self.coeffs["edge_ent"] * ent.mean()

        if self.hard_node_mask is not None:
            assert self.node_mask is not None
            m = self.node_mask[self.hard_node_mask].sigmoid()
            node_reduce = getattr(torch, self.coeffs["node_feat_reduction"])
            loss = loss + self.coeffs["node_feat_size"] * node_reduce(m)
            ent = -m * torch.log(m + self.coeffs["EPS"]) - (1 - m) * torch.log(
                1 - m + self.coeffs["EPS"]
            )
            loss = loss + self.coeffs["node_feat_ent"] * ent.sum()

        return loss

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None
