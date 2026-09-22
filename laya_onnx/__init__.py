"""Laya typed decisions on ONNX Runtime (CPU / OpenVINO / CUDA)."""

from .agent import Agent, RLAgent, load

__version__ = "0.1.0"
__all__ = [
    "Agent",
    "RLAgent",
    "load",
]
