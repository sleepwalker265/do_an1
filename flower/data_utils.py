"""
Data utilities for Federated ProtoNet

Handles episodic dataset loading and preparation for distributed learning.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple


class EpisodicDataset(Dataset):
    """
    Base class for episodic datasets used in few-shot learning.
    Generates tasks (episodes) with support and query sets.
    """
    def __init__(self, data_root: str, img_size: int = 84, 
                 mean: List[float] = None, std: List[float] = None):
        self.data_root = data_root
        self.img_size = img_size
        self.mean = mean or [0.4712, 0.4499, 0.4031]
        self.std = std or [0.2726, 0.2634, 0.2794]

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError


def load_episodic_data(data_root: str, 
                       num_ways: int = 5,
                       num_support: int = 5,
                       num_query: int = 15,
                       num_tasks: int = 100,
                       img_size: int = 84,
                       mean: List[float] = None,
                       std: List[float] = None) -> DataLoader:
    """
    Load episodic data for few-shot learning.
    
    Args:
        data_root: Path to dataset root
        num_ways: Number of classes per task
        num_support: Number of support examples per class
        num_query: Number of query examples per class
        num_tasks: Number of tasks (episodes) per epoch
        img_size: Image size
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        DataLoader for episodic tasks
    """
    dataset = EpisodicDataset(
        data_root=data_root,
        img_size=img_size,
        mean=mean,
        std=std
    )
    
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    return loader


def split_support_query(data: torch.Tensor, num_support: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split data into support and query sets.
    
    Args:
        data: Input data tensor
        num_support: Number of support examples
    
    Returns:
        Tuple of (support_set, query_set)
    """
    support = data[:num_support]
    query = data[num_support:]
    return support, query


def normalize_batch(batch: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """
    Normalize a batch of images.
    
    Args:
        batch: Batch of images [B, C, H, W]
        mean: Normalization mean per channel
        std: Normalization std per channel
    
    Returns:
        Normalized batch
    """
    mean_t = torch.tensor(mean).view(1, -1, 1, 1)
    std_t = torch.tensor(std).view(1, -1, 1, 1)
    return (batch - mean_t) / std_t
