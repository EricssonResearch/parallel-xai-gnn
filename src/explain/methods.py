"""
This module contains the code for the explainability methods.
"""

# Standard libraries
import copy
from typing import Callable
from abc import ABC, abstractmethod

# 3pps
import torch
import torch.nn.functional as F
import torch_geometric


class Explainer(ABC):
    """
    This class is an abstract class for explainers.

    Attributes:
        model: Model to make predictions.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Constructor for Explainer class

        Args:
            model: Model to make predictions.

        Returns:
            None.
        """

        # Set attributes
        self.model = model

        return None

    @abstractmethod
    @torch.no_grad()
    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        This method computes the explanation. It should be implemented
        by any class that it inherit Explainer.

        Args:
            x: Node tensor. Dimensions: [number of nodes,
                number of node features].
            edge_index: Edge index tensor. Dimensions: [2,
                number of edges].
            node_ids: Node ids tensor. Dimensions: [number of ids to
                explain].

        Returns:
            Gradients tensor. Dimensions: [number of nodes,
                number of node features].
        """

        raise NotImplementedError("Must implement this method in subclasses")


class GradientExplainer(Explainer):
    """
    This class

    Attributes:
        model: Model to make predictions.
    """

    @torch.enable_grad()
    def _compute_gradients(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        This method computes the gradients of the outputs with respect
        to the input.

        Args:
            x: Node tensor. Dimensions: [number of nodes,
                number of node features].
            edge_index: Edge index tensor. Dimensions: [2,
                number of edges].
            node_ids: Node ids tensor. Dimensions: [number of ids to
                explain].

        Raises:
            RuntimeError: Error in gradient computation.

        Returns:
            Gradients tensor. Dimensions: [number of nodes,
                number of node features].
        """

        # forward pass
        inputs = x.detach().clone()
        inputs.requires_grad_(True)
        outputs = self.model(inputs, edge_index)
        max_scores = torch.amax(outputs, dim=1)

        # backward pass
        max_scores[node_ids].sum().backward()

        # raise exception
        if inputs.grad is None:
            raise RuntimeError("Error in gradient computation")

        # compute gradients
        gradients = inputs.grad.clone()

        return gradients

    # Overriding
    @torch.no_grad()
    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        This method computes the explainability of node matrix.

        Args:
            x: Node matrix. Dimensions: [number of nodes, number of
                node features].
            edge_index: Edge index. Dimensions: [2, number of edges].
            node_ids: Node ids. Dimensions: [number of node ids].

        Returns:
            Feature explainability maps unnormalized. Dimensions:
                [number of nodes].
        """

        # Compute saliency maps
        gradients: torch.Tensor = self._compute_gradients(x, edge_index, node_ids)

        # compute feature maps
        feature_maps = torch.mean(torch.abs(gradients), dim=1)

        return feature_maps


class DeConvNet(GradientExplainer):
    """
    This class implements the deconvnet explainability technique.

    Attributes:
        model: Model to make predictions.
    """

    @torch.no_grad()
    def __init__(self, model: torch.nn.Module) -> None:
        """
        This method is the constructor of the DeConvNet class.

        Args:
            model: Model to make predictions.

        Returns:
            None.
        """

        # Call super class constructor
        super().__init__(copy.deepcopy(model))

        # Register hooks
        self.register_hooks()

        return None

    def register_hooks(self) -> None:
        """
        This methods registers the hooks in the model.

        Returns:
            None.
        """

        # Define backward hook
        def backward_hook_fn(
            module: torch.nn.Module, grad_in: torch.Tensor, grad_out: torch.Tensor
        ) -> tuple[torch.Tensor]:
            """
            Backward hook to modify gradient.

            Args:
                module: Module of pytorch
                grad_in: Gradient of the inputs.
                grad_out: Gradient of the outputs.

            Returns:
                tuple of one element with the new gradient of the
                    outputs.
            """

            # compute
            new_grad_out: torch.Tensor = F.relu(grad_out[0])

            return (new_grad_out,)

        # Define hooks variables
        backward_hook: Callable = backward_hook_fn

        # Get modules
        modules: list[tuple[str, torch.nn.Module]] = list(self.model.named_children())

        # Register hooks in relus
        module: torch.nn.Module
        for _, module in modules:
            if isinstance(module, torch.nn.ReLU):
                module.register_backward_hook(backward_hook)

        return None


class GuidedBackprop(GradientExplainer):
    """
    This class implements the guided backpropagation technique.

    Attributes:
        model: Model to make predictions.
    """

    @torch.no_grad()
    def __init__(self, model: torch.nn.Module) -> None:
        """
        This method is the constructor of the class.

        Args:
            model: Model to make predictions.

        Returns:
            None.
        """

        # Call super class constructor
        super().__init__(copy.deepcopy(model))

        # Init activations maps of model
        self.activation_maps: list[torch.Tensor] = []

        # Register hooks
        self.register_hooks()

        return None

    def register_hooks(self) -> None:
        """
        This method registers the hooks in the model.

        Returns:
            None.
        """

        # Define forward hook
        def forward_hook_fn(
            module: torch.nn.Module, inputs: torch.Tensor, outputs: torch.Tensor
        ) -> None:
            """
            This function is the forward hook.

            Args:
                module: Module of pytorch.
                input: Inputs to the layer.
                output: Outputs to the layer.

            Returns:
                None.
            """

            # Save activation maps
            self.activation_maps.append(outputs)

            return None

        # Define backward hook
        def backward_hook_fn(
            module: torch.nn.Module, grad_in: torch.Tensor, grad_out: torch.Tensor
        ) -> tuple[torch.Tensor]:
            """
            This function is the backward hook.

            Args:
                module: Module of pytorch.
                grad_in: Gradients of the inputs.
                grad_out: Gradients of the outputs.

            Returns:
                tuple of one element with the new gradient of the
                    outputs.
            """

            # create forward pass
            forward_grad: torch.Tensor = self.activation_maps.pop()
            forward_grad[forward_grad > 0] = 1

            # compute
            new_grad_out: torch.Tensor = F.relu(grad_out[0]) * forward_grad

            return (new_grad_out,)

        # define hooks variables
        forward_hook: Callable = forward_hook_fn
        backward_hook: Callable = backward_hook_fn

        # get modules
        modules: list[tuple[str, torch.nn.Module]] = list(self.model.named_children())

        # register hooks in relus
        module: torch.nn.Module
        for _, module in modules:
            if isinstance(module, torch.nn.ReLU):
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

        return None


class SaliencyMap(GradientExplainer):
    """
    This class implements a Saliency Map XAI method.

    Attributes:
        model: model to classify.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        This function is the constructor of the SaliencyMap class.

        Args:
            model: model to classify.

        Returns:
            None.
        """

        # call super class constructor
        super().__init__(model)

        return None


class SmoothGrad(GradientExplainer):
    """
    This class creates smoothgrad saliency map visualizations.
    This class inherits from SaliencyMap class.

    Attributes:
        model for classifying images
        threshold for masking part of saliency map
        noise level for the creation of smoothgrad visualizations
        sample size for creation of noise duplicates
    """

    def __init__(
        self, model: torch.nn.Module, noise_level: float = 0.2, sample_size: int = 50
    ):
        """
        This method is the constructor of the class.

        Args:
            model: Model for classifying images.
            noise_level: Noise level for the creation of smoothgrad
                visualizations. Defaults to 0.2.
            sample_size: Sample size for creation of noise duplicates.
                Defaults to 50.
        """

        # Call super class constructor
        super().__init__(model)

        # set noise level and sample size
        self.noise_level = noise_level
        self.sample_size = sample_size

        return None

    # Overriding
    @torch.no_grad()
    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        This method computes the explainability of node matrix.

        Args:
            x: Node matrix. Dimensions: [number of nodes, number of
                node features].
            edge_index: Edge index. Dimensions: [2, number of edges].
            node_ids: Node ids. Dimensions: [number of node ids].

        Returns:
            Feature explainability maps unnormalized. Dimensions:
                [number of nodes].
        """

        # Init gradients
        gradients: torch.Tensor = torch.zeros_like(x)

        # Compute x
        min_ = torch.amin(x)
        max_ = torch.amax(x)
        std = (max_ - min_) * self.noise_level * torch.ones_like(x)
        noise = torch.normal(mean=0, std=std)

        # Compute gradients for each noise batch
        for _ in range(x.shape[0]):
            # Clone batch
            x_batch: torch.Tensor = x + noise[torch.randperm(x.shape[0]), :]

            # Pass the noise batch through the model
            gradients += self._compute_gradients(x_batch, edge_index, node_ids)

        # Create feature maps
        feature_maps: torch.Tensor = torch.mean(
            torch.abs(gradients / self.sample_size), dim=1
        )

        return feature_maps


class GNNExplainer(Explainer):
    """
    This class implements the GNNExplainer.

    Args:
        explainer: Pytorch geometric explainer.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Constructor for Explainer class

        Args:
            model: Model to make predictions.

        Returns:
            None.
        """

        # Call super class constructor
        super().__init__(model)

        # Define explainer and targets
        self.explainer = torch_geometric.explain.Explainer(
            self.model,
            torch_geometric.explain.GNNExplainer(),
            explanation_type="model",
            node_mask_type="object",
            model_config={
                "mode": "multiclass_classification",
                "task_level": "node",
                "return_type": "raw",
            },
        )

        return None

    # Overriding method
    @torch.enable_grad()
    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        This method computes the explainability of node matrix.

        Args:
            x: Node matrix. Dimensions: [number of nodes, number of
                node features].
            edge_index: Edge index. Dimensions: [2, number of edges].
            node_ids: Node ids. Dimensions: [number of node ids].

        Returns:
            Feature explainability maps unnormalized. Dimensions:
                [number of nodes].
        """

        # Compute feature maps
        explanation = self.explainer(x, edge_index, index=node_ids)
        feature_maps = explanation.node_mask[:, 0]

        return feature_maps
