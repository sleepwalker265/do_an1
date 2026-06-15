"""
Model utilities for Federated ProtoNet

Builds and manages ProtoNet models with ResNet backbones.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from collections import OrderedDict
import numpy as np


class ResNet12Backbone(nn.Module):
    """
    ResNet-12 backbone for few-shot learning.
    """
    def __init__(self, num_classes: int = 64):
        super().__init__()
        self.num_classes = num_classes
        # TODO: Implement ResNet12 architecture
        self.fc = nn.Linear(640, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ProtoNetHead(nn.Module):
    """
    Prototypical Network classification head.
    """
    def __init__(self, feat_dim: int = 64, num_ways: int = 5):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_ways = num_ways
    
    def forward(self, support: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Compute classification logits using prototypical networks.
        
        Args:
            support: Support set features [num_ways*num_support, feat_dim]
            query: Query set features [num_ways*num_query, feat_dim]
        
        Returns:
            Classification logits [num_ways*num_query, num_ways]
        """
        # Compute prototypes
        support_reshaped = support.view(self.num_ways, -1, self.feat_dim)
        prototypes = support_reshaped.mean(dim=1)  # [num_ways, feat_dim]
        
        # Compute distances
        query_expanded = query.unsqueeze(1)  # [num_query, 1, feat_dim]
        distances = torch.norm(query_expanded - prototypes.unsqueeze(0), p=2, dim=2)
        
        # Negative distances as logits
        logits = -distances
        return logits


class ProtoNet(nn.Module):
    """
    Complete ProtoNet model combining backbone and head.
    """
    def __init__(self, backbone_name: str = "resnet12", num_ways: int = 5):
        super().__init__()
        
        if backbone_name == "resnet12":
            self.backbone = ResNet12Backbone(num_classes=64)
            feat_dim = 64
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        self.head = ProtoNetHead(feat_dim=feat_dim, num_ways=num_ways)
    
    def forward(self, support: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            support: Support set images
            query: Query set images
        
        Returns:
            Classification logits
        """
        # Extract features
        support_feat = self.backbone(support)
        query_feat = self.backbone(query)
        
        # Classify
        logits = self.head(support_feat, query_feat)
        return logits


def build_model(cfg: dict) -> nn.Module:
    """
    Build a ProtoNet model from configuration.
    
    Args:
        cfg: Configuration dictionary containing 'backbone' and 'classifier' keys
    
    Returns:
        ProtoNet model
    """
    backbone_name = cfg.get("backbone", "resnet12")
    classifier_name = cfg.get("classifier", "proto_head")
    
    if classifier_name != "proto_head":
        raise ValueError(f"Unknown classifier: {classifier_name}")
    
    model = ProtoNet(backbone_name=backbone_name)
    return model


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """
    Extract model parameters as a list of NumPy arrays.
    
    Args:
        model: PyTorch model
    
    Returns:
        List of parameter arrays
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Update model parameters from a list of NumPy arrays.
    
    Args:
        model: PyTorch model
        parameters: List of parameter arrays
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=False)


def load_checkpoint_into_model(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    """
    Load a checkpoint into a model.
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)


import os
