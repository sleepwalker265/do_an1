"""
Flower Client for Federated ProtoNet

Implements the client-side logic for federated learning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

import flwr as fl
from flwr.common import NDArrays, Scalar


class ProtoNetClient(fl.client.NumPyClient):
    """
    Federated learning client for ProtoNet training.
    """
    def __init__(self, client_id: int, cfg: dict, device: torch.device):
        """
        Initialize the Flower client.
        
        Args:
            client_id: Unique identifier for this client
            cfg: Configuration dictionary
            device: torch device (cpu or cuda)
        """
        self.client_id = client_id
        self.cfg = cfg
        self.device = device
        
        # Initialize model
        self.model = self._build_model(cfg)
        self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = self._build_optimizer(cfg)
        
        # Training state
        self.local_epochs = cfg.get("client", {}).get("local_epochs", 1)
    
    def _build_model(self, cfg: dict):
        """
        Build the ProtoNet model.
        """
        from .models import build_model
        return build_model(cfg.get("model", {}))
    
    def _build_optimizer(self, cfg: dict):
        """
        Build the optimizer.
        """
        opt_cfg = cfg.get("optimizer", {})
        optimizer_name = opt_cfg.get("name", "SGD")
        lr = opt_cfg.get("lr", 0.01)
        momentum = opt_cfg.get("momentum", 0.9)
        weight_decay = opt_cfg.get("weight_decay", 5e-4)
        
        if optimizer_name == "SGD":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            return torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """
        Return the current model parameters as a list of NumPy arrays.
        """
        from .models import get_parameters
        return get_parameters(self.model)
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """
        Update the model with parameters from the server.
        """
        from .models import set_parameters
        set_parameters(self.model, parameters)
    
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        """
        Train the model locally.
        
        Args:
            parameters: Model parameters from the server
            config: Configuration from the server
        
        Returns:
            Tuple of (updated_parameters, num_examples, metrics_dict)
        """
        # Update with server parameters
        self.set_parameters(parameters)
        
        # Train locally
        num_examples = self._train()
        
        # Return updated parameters and metrics
        return self.get_parameters(config), num_examples, {"loss": 0.0}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on local data.
        
        Args:
            parameters: Model parameters from the server
            config: Configuration from the server
        
        Returns:
            Tuple of (loss, num_examples, metrics_dict)
        """
        # Update with server parameters
        self.set_parameters(parameters)
        
        # Evaluate locally
        loss, accuracy, num_examples = self._evaluate()
        
        return loss, num_examples, {"accuracy": accuracy}
    
    def _train(self) -> int:
        """
        Local training loop.
        """
        self.model.train()
        # TODO: Implement actual training logic
        return 1  # num_examples
    
    def _evaluate(self) -> Tuple[float, float, int]:
        """
        Local evaluation loop.
        """
        self.model.eval()
        # TODO: Implement actual evaluation logic
        return 0.0, 0.0, 1  # loss, accuracy, num_examples
    
    def to_client(self):
        """
        Return a Flower Client instance.
        """
        return self
