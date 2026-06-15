"""
Flower - Federated ProtoNet Package

A federated learning implementation of Prototypical Networks using Flower framework.
"""

__version__ = "0.1.0"

from .models import ProtoNet, build_model, get_parameters, set_parameters
from .flower_client import ProtoNetClient
from .data_utils import load_episodic_data, split_support_query

__all__ = [
    "ProtoNet",
    "build_model",
    "get_parameters",
    "set_parameters",
    "ProtoNetClient",
    "load_episodic_data",
    "split_support_query",
]
