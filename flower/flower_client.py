"""
Flower Client for Federated ProtoNet.

Each client:
  1. Receives global model weights from the Flower server.
  2. Trains locally for `local_epochs` epochs using episodic ProtoNet loss.
  3. Reports updated weights + local metrics back to the server.
"""

import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import time
import datetime
import torch
import torch.nn.functional as F
import numpy as np
import flwr as fl
from flwr.common import (
    Code, EvaluateIns, EvaluateRes,
    FitIns, FitRes, GetParametersIns, GetParametersRes,
    Parameters, Status, ndarrays_to_parameters, parameters_to_ndarrays,
)

from optimizer import build_optimizer, build_scheduler
from utils import accuracy, AverageMeter

from .models import build_model, load_checkpoint_into_model, get_parameters, set_parameters
from .data_utils import get_dataloaders


# ---------------------------------------------------------------------------
# Flower NumPy client
# ---------------------------------------------------------------------------

class ProtoNetClient(fl.client.NumPyClient):
    """
    Flower NumPy client that wraps the Prototypical Network training loop.

    Parameters
    ----------
    client_id    : int   – index used to select the local dataset root
    cfg          : dict  – full config loaded from config.yaml
    device       : torch.device
    """

    def __init__(self, client_id: int, cfg: dict, device: torch.device):
        self.client_id    = client_id
        self.cfg          = cfg
        self.device       = device
        self.local_epochs = cfg["client"].get("local_epochs", 1)

        # ---- Build model ----
        self.model = build_model(cfg["model"]).to(device)

        # ---- Optionally warm-start from a pretrained checkpoint ----
        pretrained = cfg.get("pretrained", None)
        if pretrained and os.path.exists(pretrained):
            print(f"[Client {client_id}] Loading pretrained: {pretrained}")
            load_checkpoint_into_model(self.model, pretrained, device)
        elif pretrained:
            print(f"[Client {client_id}] WARNING: pretrained path not found: {pretrained}")

        # ---- Build data loaders ----
        (self.train_loader, self.train_dataset,
         self.val_loader,   self.val_dataset) = get_dataloaders(cfg, client_id)

        # ---- Build optimiser — construct a minimal yacs-like object ----
        self.optimizer, self.lr_scheduler = self._build_optimizer()

        self._step = 0  # global step counter across rounds

    # ------------------------------------------------------------------
    # Flower API
    # ------------------------------------------------------------------

    def get_parameters(self, config: dict) -> list:
        """Return current model weights as numpy arrays."""
        return get_parameters(self.model)

    def fit(self, parameters: list, config: dict) -> tuple:
        """
        1. Set weights received from server.
        2. Train for `local_epochs` epochs.
        3. Return updated weights + metrics.
        """
        set_parameters(self.model, parameters)

        train_loss, train_acc = 0.0, 0.0
        for epoch in range(self.local_epochs):
            loss, acc, self._step = self._train_one_epoch(epoch, self._step)
            train_loss += loss
            train_acc  += acc

        train_loss /= self.local_epochs
        train_acc  /= self.local_epochs

        num_examples = self.cfg["data"]["episode"].get("num_tasks_per_epoch", 100)

        return (
            get_parameters(self.model),
            num_examples,
            {"train_loss": float(train_loss), "train_acc": float(train_acc)},
        )

    def evaluate(self, parameters: list, config: dict) -> tuple:
        """
        Set weights received from server, evaluate on local validation set.
        """
        set_parameters(self.model, parameters)
        val_loss, val_acc = self._evaluate()

        num_examples = self.cfg["data"]["episode"].get("num_tasks_per_epoch", 100) // 3

        return (
            float(val_loss),
            num_examples,
            {"val_acc": float(val_acc)},
        )

    # ------------------------------------------------------------------
    # Internal training helpers
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int, step: int):
        """Run one epoch of episodic ProtoNet training."""
        self.model.train()
        self.optimizer.zero_grad()

        loss_meter = AverageMeter()
        acc_meter  = AverageMeter()
        batch_time = AverageMeter()

        self.train_dataset.set_epoch()

        print_freq = self.cfg.get("print_freq", 10)
        schedule_per_step = self.cfg["lr_scheduler"].get("schedule_per_step", True)

        end = time.time()
        for idx, batches in enumerate(self.train_loader):
            dataset_index, imgs, labels = batches

            loss, acc = self.model.train_forward(imgs, labels, dataset_index)
            acc_val   = torch.mean(torch.stack(acc))

            loss.backward()
            self.optimizer.step()

            if schedule_per_step:
                self.lr_scheduler.step_update(step)
                step += 1

            self.optimizer.zero_grad()

            loss_meter.update(loss.item())
            acc_meter.update(acc_val.item())
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % print_freq == 0:
                lr  = self.optimizer.param_groups[0]["lr"]
                etas = batch_time.avg * (len(self.train_loader) - idx - 1)
                print(
                    f"[Client {self.client_id}] "
                    f"Epoch [{epoch+1}][{idx+1}/{len(self.train_loader)}] "
                    f"eta {datetime.timedelta(seconds=int(etas))} "
                    f"lr {lr:.6f}  "
                    f"loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})  "
                    f"acc {acc_meter.val:.2f} ({acc_meter.avg:.2f})"
                )

        if not schedule_per_step:
            self.lr_scheduler.step_update(step)
            step += 1

        return loss_meter.avg, acc_meter.avg, step

    @torch.no_grad()
    def _evaluate(self):
        """Run validation and return (avg_loss, avg_acc)."""
        self.model.eval()

        loss_meter = AverageMeter()
        acc_meter  = AverageMeter()

        self.val_dataset.set_epoch()

        for idx, batches in enumerate(self.val_loader):
            dataset_index, imgs, labels = batches
            loss, acc = self.model.val_forward(imgs, labels, dataset_index)
            acc_val   = torch.mean(torch.stack(acc))

            loss_meter.update(loss.item())
            acc_meter.update(acc_val.item())

        print(
            f"[Client {self.client_id}] "
            f"Val → loss {loss_meter.avg:.3f}  acc {acc_meter.avg:.2f}%"
        )
        return loss_meter.avg, acc_meter.avg

    # ------------------------------------------------------------------
    # Optimiser / scheduler construction
    # ------------------------------------------------------------------

   def _build_optimizer(self):
        """
        Construct optimiser and LR scheduler using the existing helpers
        from the main codebase.  We build a lightweight yacs-compatible
        config object on the fly.
        """
        from yacs.config import CfgNode as CN

        opt_cfg = self.cfg.get("optimizer", {})
        sched_cfg = self.cfg.get("lr_scheduler", {})
        ep_cfg    = self.cfg["data"]["episode"]

        # Minimal training config understood by build_optimizer / build_scheduler
        train_node = CN()
        train_node.OPTIMIZER               = CN()
        train_node.OPTIMIZER.NAME          = opt_cfg.get("name", "SGD")
        train_node.OPTIMIZER.MOMENTUM      = opt_cfg.get("momentum", 0.9)
        train_node.BASE_LR                 = opt_cfg.get("lr", 0.01)
        train_node.WEIGHT_DECAY            = opt_cfg.get("weight_decay", 5e-4)
        train_node.WARMUP_EPOCHS           = 0
        train_node.WARMUP_LR_INIT          = 0.0
        train_node.EPOCHS                  = self.local_epochs
        train_node.LR_SCHEDULER            = CN()
        train_node.LR_SCHEDULER.NAME       = sched_cfg.get("name", "cosine")
        train_node.SCHEDULE_PER_STEP       = sched_cfg.get("schedule_per_step", True)
        train_node.START_EPOCH             = 0
        train_node.AUTO_RESUME             = False

        # Wrap in a root config so build_optimizer can access config.TRAIN
        root_cfg = CN()
        root_cfg.TRAIN = train_node

        # Thêm cấu hình DATA cho build_scheduler / build_optimizer nếu cần
        root_cfg.DATA = CN()
        root_cfg.DATA.TRAIN = CN()
        root_cfg.DATA.TRAIN.IS_EPISODIC = ep_cfg.get("is_episodic", True)
        root_cfg.DATA.TRAIN.ITERATION_PER_EPOCH = None
        root_cfg.DATA.TRAIN.DATASET_NAMES = [self.cfg["data"].get("dataset_name", "miniImageNet")]

        optimizer  = build_optimizer(root_cfg, self.model)
        scheduler  = build_scheduler(root_cfg, optimizer, len(self.train_loader))
        
        return optimizer, scheduler
