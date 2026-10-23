"""
This module defines the constants for the src.explain module.
"""

# Standard libraries
from typing import Final, Literal, Type, TypeAlias, cast

# Own modules
from src.utils import DatasetName, ModelName
from src.xai.methods import (
    DeConvNet,
    Explainer,
    GNNExplainer,
    GuidedBackprop,
    IntegratedGradients,
    SaliencyMap,
    SmoothGrad,
)

# Define types
MethodName: TypeAlias = Literal[
    "DeConvNet",
    "GNNExplainer",
    "GraphMaskExplainer",
    "GuidedBackprop",
    "IntegratedGradients",
    "SaliencyMap",
    "SmoothGrad",
]
XAIType: TypeAlias = Literal["original", "cluster"]

# Static variables
METHODS: dict[MethodName, Type[Explainer]] = {
    "DeConvNet": DeConvNet,
    "GNNExplainer": GNNExplainer,
    "GuidedBackprop": GuidedBackprop,
    "IntegratedGradients": IntegratedGradients,
    "SaliencyMap": SaliencyMap,
    "SmoothGrad": SmoothGrad,
}
XAI_TYPES: Final[tuple[XAIType, ...]] = cast(
    tuple[XAIType, ...], ("original", "cluster")
)
NUM_CLUSTERS: Final[tuple[int, ...]] = (8, 16, 32, 64, 128)
RATES: tuple[float, ...] = (1.0, 0.7, 0.5, 0.2, 0.0)
ITERATIONS: int = 3
CHECKPOINTS_PATH: str = "checkpoints"
