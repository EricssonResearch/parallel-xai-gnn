# deep learning libraries
import torch
import torch_geometric
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle

# other libraries
import copy
from tqdm.auto import tqdm
from typing import Optional, List, Tuple, Callable
from abc import ABC, abstractmethod


class Explainer(ABC):
    def __init__(self, model: torch.nn.Module):
        """
        Constructor for Explainer class

        Args:
            model: pytorch model
        """

        self.model = model

    @abstractmethod
    @torch.no_grad()
    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_ids: torch.Tensor
    ) -> torch.Tensor:
        pass


# class GradientExplainer(Explainer):

#     def __init__(self, model: torch.nn.Module):
#         """
#         Constructor for Explainer class

#         Args:
#             model: pytorch model
#         """

#         self.model = model

#     def


class DeConvNet(Explainer):
    @torch.no_grad()
    def __init__(self, model: torch.nn.Module) -> None:
        # call super class constructor
        super().__init__(copy.deepcopy(model))

        # register hooks
        self.register_hooks()

    def register_hooks(self) -> None:
        # define backward hook
        def backward_hook_fn(
            module: torch.nn.Module, grad_in: torch.Tensor, grad_out: torch.Tensor
        ) -> Tuple[torch.Tensor]:
            # compute
            new_grad_out: torch.Tensor = F.relu(grad_out[0])

            return (new_grad_out,)

        # define hooks variables
        backward_hook: Callable = backward_hook_fn

        # get modules
        modules: List[Tuple[str, torch.nn.Module]] = list(self.model.named_children())

        # register hooks in relus
        module: torch.nn.Module
        for _, module in modules:
            if isinstance(module, torch.nn.ReLU):
                module.register_backward_hook(backward_hook)

    @torch.enable_grad()
    def _compute_gradients(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_id: int
    ) -> Optional[torch.Tensor]:
        # forward pass
        inputs = x.clone()
        inputs.requires_grad_(True)
        outputs = self.model(inputs, edge_index)
        max_scores = torch.amax(outputs, dim=1)

        # clear previous gradients and backward pass
        self.model.zero_grad()
        max_scores[node_id].backward()

        return inputs.grad.clone()

    # overriding
    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_id: int
    ) -> torch.Tensor:
        # get device
        device: torch.device = x.device

        # try gpu execution
        try:
            # compute saliency maps
            gradients: Optional[torch.Tensor] = self._compute_gradients(
                x, edge_index, node_id
            )

        except RuntimeError:
            # pass tensors to cpu
            x = x.cpu()
            edge_index = edge_index.cpu()
            self.model = self.model.cpu()

            # compute saliency maps
            gradients: Optional[torch.Tensor] = self._compute_gradients(
                x, edge_index, node_id
            )

        # handle exception and get back to device
        if gradients is None:
            raise RuntimeError("Error in gradient computation")
        else:
            gradients = gradients.to(device)
            self.model = self.model.to(device)

        # compute feature maps
        feature_maps = torch.mean(torch.abs(gradients), dim=1)

        # normalize
        min_ = torch.amin(feature_maps)
        max_ = torch.amax(feature_maps)
        feature_maps = (feature_maps - min_) / (max_ - min_)

        return feature_maps


class GuidedBackprop(Explainer):
    @torch.no_grad()
    def __init__(self, model: torch.nn.Module) -> None:
        # call super class constructor
        super().__init__(copy.deepcopy(model))

        # init activations maps of model
        self.activation_maps: List[torch.Tensor] = []

        # register hooks
        self.register_hooks()

    def register_hooks(self) -> None:
        # define forward hook
        def forward_hook_fn(
            module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
        ) -> None:
            self.activation_maps.append(output)

        # define backward hook
        def backward_hook_fn(
            module: torch.nn.Module, grad_in: torch.Tensor, grad_out: torch.Tensor
        ) -> Tuple[torch.Tensor]:
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
        modules: List[Tuple[str, torch.nn.Module]] = list(self.model.named_children())

        # register hooks in relus
        module: torch.nn.Module
        for _, module in modules:
            if isinstance(module, torch.nn.ReLU):
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    @torch.enable_grad()
    def _compute_gradients(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_id: int
    ) -> Optional[torch.Tensor]:
        # forward pass
        inputs = x.clone()
        inputs.requires_grad_(True)
        outputs = self.model(inputs, edge_index)
        max_scores = torch.amax(outputs, dim=1)

        # clear previous gradients and backward pass
        self.model.zero_grad()
        max_scores[node_id].backward()

        return inputs.grad

    # overriding
    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_id: int
    ) -> torch.Tensor:
        # get device
        device: torch.device = x.device

        # try gpu execution
        try:
            # compute saliency maps
            gradients: Optional[torch.Tensor] = self._compute_gradients(
                x, edge_index, node_id
            )

        except RuntimeError:
            # pass tensors to cpu
            x = x.cpu()
            edge_index = edge_index.cpu()
            self.model = self.model.cpu()

            # compute saliency maps
            gradients: Optional[torch.Tensor] = self._compute_gradients(
                x, edge_index, node_id
            )

        # handle exception and get back to device
        if gradients is None:
            raise RuntimeError("Error in gradient computation")
        else:
            gradients = gradients.to(device)
            self.model = self.model.to(device)

        # compute feature maps
        feature_maps = torch.mean(torch.abs(gradients), dim=1)

        # normalize
        min_ = torch.amin(feature_maps)
        max_ = torch.amax(feature_maps)
        feature_maps = (feature_maps - min_) / (max_ - min_)

        return feature_maps


class SaliencyMap(Explainer):
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
        inputs = x.clone()
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
        self, x: torch.Tensor, edge_index: torch.Tensor, node_ids: int
    ) -> torch.Tensor:
        """
        This method computes the explainability of node matrix.

        Args:
            x: Node matrix. Dimensions: [number of nodes, number of
                node features].
            edge_index: Edge index. Dimensions: [2, number of edges].
            node_ids: Node ids. Dimensions: [number of node ids].

        Returns:
            Saliency Map explainability unnormalized. Dimensions:
                [number of nodes].
        """

        # Try gpu execution
        gradients: torch.Tensor
        try:
            # Compute saliency maps
            gradients = self._compute_gradients(x, edge_index, node_ids)

        except RuntimeError as e:
            # Catch exception only if it´s from cuda
            if "out of memory" in str(e):
                # get device
                device: torch.device = x.device

                # pass tensors to cpu
                x = x.cpu()
                edge_index = edge_index.cpu()
                node_ids = node_ids.cpu()
                self.model = self.model.cpu()

                # compute saliency maps
                gradients = self._compute_gradients(x, edge_index, node_ids)

                # get back to device
                gradients = gradients.to(device)
                node_ids = node_ids.to(device)
                self.model = self.model.to(device)

            else:
                raise e

        # compute feature maps
        feature_maps = torch.mean(torch.abs(gradients), dim=1)

        return feature_maps


class SmoothGrad(Explainer):
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
        Constructor of SmoothGradSaliencyMap class

        Args:
            model for classifying images
            noise level for the creation of smoothgrad visualizations. Default value: 0.2
            sample size for creation of noise duplicates. Default value: 50
        """

        # set noise level and sample size
        self.model = model
        self.noise_level = noise_level
        self.sample_size = sample_size

    # overriding super class method
    @torch.no_grad()
    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_id: int
    ) -> torch.Tensor:
        """
        This method computes smoothgrad saliency maps

        Args:
            batch of images. Dimensions: [bath size, channels, height, width]

        Returns:
            batch of saliency maps. Dimensions: [batch size, height, width]
        """

        # compute inputs with noise
        min_ = torch.amin(x)
        max_ = torch.amax(x)
        std = (
            (max_ - min_)
            * self.noise_level
            * torch.ones(self.sample_size, *x.size()).to(x.device)
        )
        noise = torch.normal(mean=0, std=std)
        inputs = x.clone().unsqueeze(0)
        inputs = inputs + noise

        # create gradients tensor
        gradients = torch.zeros_like(inputs)

        # compute gradients for each noise batch
        for i in range(inputs.size(0)):
            # clone batch
            inputs_batch = inputs[i].clone()

            # pass the noise batch through the model
            with torch.enable_grad():
                inputs_batch.requires_grad_()
                outputs = self.model(inputs_batch, edge_index)
                max_scores = torch.amax(outputs, dim=1)

                # compute gradients
                self.model.zero_grad()
                max_scores[node_id].backward()
                if inputs_batch.grad is None:
                    raise RuntimeError("Error in gradient computation")
                gradients[i] = inputs_batch.grad

        # create saliency maps
        feature_maps = torch.mean(gradients, dim=0) / self.sample_size
        feature_maps = torch.amax(torch.abs(feature_maps), dim=1)

        # normalize
        min_ = torch.amin(feature_maps)
        max_ = torch.amax(feature_maps)
        feature_maps = (feature_maps - min_) / (max_ - min_)

        return feature_maps


class GNNExplainer(Explainer):
    def __init__(self, model: torch.nn.Module):
        """
        Constructor for Explainer class

        Args:
            model: pytorch model
        """

        # call super class constructor
        super().__init__(model)

        # define explainer and targets
        self.explainer = torch_geometric.explain.Explainer(
            self.model,
            torch_geometric.explain.GNNExplainer(),
            explanation_type="model",
            node_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="raw",
            ),
        )

    # overriding method
    @torch.enable_grad()
    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_id: int
    ) -> torch.Tensor:
        # compute feature maps
        explanation = self.explainer(x, edge_index, index=node_id)
        feature_maps = explanation.node_mask[:, 0]

        # normalize
        min_ = torch.amin(feature_maps)
        max_ = torch.amax(feature_maps)
        feature_maps = (feature_maps - min_) / (max_ - min_)

        return feature_maps
