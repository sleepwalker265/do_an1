"""
Federated ProtoNet — Entry Point (main.py)

Usage
-----
# 1. Start the Flower server (aggregator):
    python -m federated_protonet.main --mode server --config federated_protonet/config.yaml

# 2. Start one client per terminal / machine:
    python -m federated_protonet.main --mode client --client-id 0 --config federated_protonet/config.yaml
    python -m federated_protonet.main --mode client --client-id 1 --config federated_protonet/config.yaml

Options
-------
--mode        server | client
--config      path to config.yaml  (default: federated_protonet/config.yaml)
--client-id   integer id of this client (used to pick the local dataset root)
--server-addr host:port of the Flower server  (default from config.yaml)
--rounds      override number of federated rounds
--gpu         GPU id (overrides config.yaml)
"""

import sys
import os

# Ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import random
import numpy as np
import torch
import yaml
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Weighted metric aggregation callbacks for FedAvg
# ---------------------------------------------------------------------------

def weighted_average_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate metrics (loss / acc) weighted by number of local examples.
    Called by Flower server after each fit / evaluate round.
    """
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}

    aggregated: Dict[str, float] = {}
    for n, m in metrics:
        for k, v in m.items():
            aggregated[k] = aggregated.get(k, 0.0) + v * n / total

    return aggregated


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def run_server(cfg: dict, server_addr: str, num_rounds: int):
    """Start the Flower server with FedAvg strategy."""
    print(f"\n[Server] Starting on {server_addr}  rounds={num_rounds}\n")

    # ---- optionally initialise global model weights ----
    # We let Flower initialise from the first client's weights (default).
    # Alternatively, load the pretrained checkpoint here and broadcast:
    pretrained = cfg.get("pretrained", None)
    initial_parameters = None

    if pretrained and os.path.exists(pretrained):
        from .models import build_model, load_checkpoint_into_model, get_parameters
        from flwr.common import ndarrays_to_parameters
        print(f"[Server] Loading pretrained checkpoint: {pretrained}")
        model = build_model(cfg["model"])
        load_checkpoint_into_model(model, pretrained, torch.device("cpu"))
        initial_parameters = ndarrays_to_parameters(get_parameters(model))
        del model

    server_cfg  = cfg["server"]
    strategy    = FedAvg(
        fraction_fit            = server_cfg.get("fraction_fit", 1.0),
        fraction_evaluate       = server_cfg.get("fraction_evaluate", 1.0),
        min_fit_clients         = server_cfg.get("min_fit_clients", 2),
        min_evaluate_clients    = server_cfg.get("min_evaluate_clients", 2),
        min_available_clients   = server_cfg.get("min_available_clients", 2),
        initial_parameters      = initial_parameters,
        fit_metrics_aggregation_fn      = weighted_average_metrics,
        evaluate_metrics_aggregation_fn = weighted_average_metrics,
    )

    fl.server.start_server(
        server_address=server_addr,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def run_client(cfg: dict, client_id: int, server_addr: str, device: torch.device):
    """Instantiate a ProtoNetClient and connect to the Flower server."""
    from .flower_client import ProtoNetClient

    print(f"\n[Client {client_id}] Connecting to {server_addr}\n")
    client = ProtoNetClient(client_id=client_id, cfg=cfg, device=device)
    fl.client.start_client(
        server_address=server_addr,
        client=client.to_client(),
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated ProtoNet (Flower)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["server", "client"], required=True,
        help="Run as 'server' (aggregator) or 'client' (local trainer).",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to config.yaml  (default: federated_protonet/config.yaml)",
    )
    parser.add_argument(
        "--client-id", type=int, default=0,
        help="[client only] Integer ID to select the local dataset root.",
    )
    parser.add_argument(
        "--server-addr", type=str, default=None,
        help="host:port of the Flower server  (overrides config.yaml server.address)",
    )
    parser.add_argument(
        "--rounds", type=int, default=None,
        help="[server only] Override number of federated rounds.",
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="GPU id to use (overrides config.yaml gpu_id). Use -1 for CPU.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # ---- Seed ----
    setup_seed(cfg.get("seed", 42))

    # ---- Device ----
    gpu_id = args.gpu if args.gpu is not None else cfg.get("gpu_id", 0)
    if gpu_id < 0 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{gpu_id}")
    print(f"[main] Using device: {device}")

    # ---- Server address ----
    server_addr = args.server_addr or cfg["server"].get("address", "0.0.0.0:8080")

    # ---- Output dir ----
    out_dir = cfg.get("output_dir", "federated_protonet/results")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Dispatch ----
    if args.mode == "server":
        num_rounds = args.rounds or cfg["server"].get("num_rounds", 50)
        run_server(cfg, server_addr, num_rounds)
    else:
        run_client(cfg, args.client_id, server_addr, device)


if __name__ == "__main__":
    main()
