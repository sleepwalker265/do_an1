"""
Flower - Federated ProtoNet Package
"""
__version__ = "0.1.0"

from .models import EpisodicProtoNet, build_model, get_parameters, set_parameters
from .flower_client import ProtoNetClient
from .data_utils import get_dataloaders

__all__ = [
    "EpisodicProtoNet",
    "build_model",
    "get_parameters",
    "set_parameters",
    "ProtoNetClient",
    "get_dataloaders",
]
