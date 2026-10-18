"""
This module defines the constants for the src.explain module.
"""

# Standard libraries
from typing import Final, Literal, Type, TypeAlias, cast

# Own modules
from src.explain.methods import (
    DeConvNet,
    Explainer,
    GNNExplainer,
    GraphMaskExplainer,
    GuidedBackprop,
    IntegratedGradients,
    SaliencyMap,
    SmoothGrad,
)
from src.utils import DatasetName, ModelName

# Define types
MethodName: TypeAlias = Literal[
    "DeConvNet",
    "GNNExplainer",
    "GraphMaskExplainer",
    "GuidedBackprop",
    "IntegratedGradients",
    "SaliencyMap",
    "SmoothGrad"
]
XAIType: TypeAlias = Literal["original", "cluster"]

# Static variables
METHODS: dict[MethodName, Type[Explainer]] = {
    "DeConvNet": DeConvNet,
    # "GNNExplainer": GNNExplainer,
    # "GraphMaskExplainer": GraphMaskExplainer,
    "GuidedBackprop": GuidedBackprop,
    "IntegratedGradients": IntegratedGradients,
    "SaliencyMap": SaliencyMap,
    "SmoothGrad": SmoothGrad
}
XAI_TYPES: Final[tuple[XAIType, ...]] = cast(
    tuple[XAIType, ...], ("original", "cluster")
)
NUM_CLUSTERS: Final[tuple[int, ...]] = (8, 16, 32, 64, 128)
DROPOUT_RATES: tuple[float, ...] = (0.0, 0.2, 0.5, 0.7, 1.0)
LIMITS: dict[tuple[DatasetName, ModelName], int] = {
    ("PubMed", "gcn"): 32,
    ("PubMed", "gat"): 16,
}
ITERATIONS: int = 3
CHECKPOINTS_PATH: str = "checkpoints"
