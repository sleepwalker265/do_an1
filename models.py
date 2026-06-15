"""
Federated ProtoNet model wrapper.

Reuses the existing EpisodicTraining model (backbone + proto_head classifier)
from the main codebase.  Only the model-construction helpers live here so that
the Flower client code can stay clean.
"""

import sys
import os

# Make sure the project root is on the path so we can import architectures/models
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from architectures import get_backbone, get_classifier
from utils import accuracy


# ---------------------------------------------------------------------------
# Core episodic model (identical logic to models/Episodic_Model.py)
# Duplicated here so federated_protonet can be used as a self-contained
# package without modifying the original training code.
# ---------------------------------------------------------------------------

class EpisodicProtoNet(nn.Module):
    """
    Prototypical Network = backbone feature extractor + PN head.

    forward() runs a full episodic batch:
      img_tasks  : list[dict]  – each dict has 'support' and 'query' tensors
      label_tasks: list[dict]  – each dict has 'support' and 'query' label tensors

    Returns (loss, list_of_per_task_acc).
    """

    def __init__(self, backbone_name: str, backbone_params: list,
                 classifier_name: str, classifier_params: list):
        super().__init__()
        self.backbone   = get_backbone(backbone_name,   *backbone_params)
        self.classifier = get_classifier(classifier_name, *classifier_params)

    # ------------------------------------------------------------------
    def forward(self, img_tasks, label_tasks):
        device     = next(self.parameters()).device
        batch_size = len(img_tasks)
        loss = 0.
        accs = []

        for i, img_task in enumerate(img_tasks):
            support_feat = self.backbone(img_task["support"].squeeze_(0).to(device))
            query_feat   = self.backbone(img_task["query"].squeeze_(0).to(device))

            score = self.classifier(
                query_feat, support_feat,
                label_tasks[i]["support"].squeeze_(0).to(device)
            )
            loss += F.cross_entropy(score, label_tasks[i]["query"].squeeze_(0).to(device))
            accs.append(accuracy(score, label_tasks[i]["query"].to(device))[0])

        loss /= batch_size
        return loss, accs

    # Convenience aliases used by Flower client
    def train_forward(self, img_tasks, label_tasks, *args, **kwargs):
        return self(img_tasks, label_tasks)

    def val_forward(self, img_tasks, label_tasks, *args, **kwargs):
        return self(img_tasks, label_tasks)

    def test_forward(self, img_tasks, label_tasks, *args, **kwargs):
        return self(img_tasks, label_tasks)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> EpisodicProtoNet:
    """
    Build the EpisodicProtoNet from a plain dict (from config.yaml).

    cfg is the 'model' section of config.yaml, e.g.:
        backbone: "resnet12"
        backbone_hyperparameters: []
        classifier: "proto_head"
        classifier_parameters: []
    """
    return EpisodicProtoNet(
        backbone_name    = cfg["backbone"],
        backbone_params  = cfg.get("backbone_hyperparameters", []),
        classifier_name  = cfg["classifier"],
        classifier_params= cfg.get("classifier_parameters", []),
    )


def load_checkpoint_into_model(model: EpisodicProtoNet, ckpt_path: str,
                                device: torch.device) -> EpisodicProtoNet:
    """
    Load a checkpoint saved by the main training pipeline into `model`.
    Works with both full checkpoints (dict with 'model' key) and raw
    state-dicts.
    """
    import yacs
    torch.serialization.add_safe_globals([yacs.config.CfgNode])
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"[federated_protonet] Loaded checkpoint '{ckpt_path}': {msg}")
    return model


def get_parameters(model: nn.Module) -> list:
    """Return model parameters as a list of numpy arrays (for Flower)."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: list) -> None:
    """Set model parameters from a list of numpy arrays (from Flower server)."""
    import numpy as np
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict  = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
