"""
Federated ProtoNet — Flower SIMULATION mode (dùng cho Kaggle / Colab)
=====================================================================
Thay thế main.py khi chạy trên môi trường notebook (1 process duy nhất).

Cách dùng trong Kaggle notebook:
    %run federated_protonet/simulation.py \
        --config federated_protonet/config.yaml \
        --num-clients 2 \
        --rounds 10

Hoặc import trực tiếp:
    from federated_protonet.simulation import run_simulation
    run_simulation(cfg, num_clients=2, num_rounds=10, device_str="cuda:0")
"""

import sys
import os

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
from flwr.common import Metrics, ndarrays_to_parameters
from typing import List, Tuple, Dict


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


def weighted_average_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    aggregated: Dict[str, float] = {}
    for n, m in metrics:
        for k, v in m.items():
            aggregated[k] = aggregated.get(k, 0.0) + v * n / total
    return aggregated


# ---------------------------------------------------------------------------
# Client factory (quan trọng: simulation cần 1 hàm trả về client theo cid)
# ---------------------------------------------------------------------------

def make_client_fn(cfg: dict, device: torch.device):
    """
    Trả về hàm client_fn(cid: str) -> fl.client.Client
    Flower simulation gọi hàm này để tạo client theo yêu cầu.
    """
    from flower.flower_client import ProtoNetClient

    def client_fn(cid: str) -> fl.client.Client:
        client_id = int(cid)
        print(f"\n[Simulation] Khởi tạo Client {client_id} ...")
        client = ProtoNetClient(client_id=client_id, cfg=cfg, device=device)
        return client.to_client()

    return client_fn


# ---------------------------------------------------------------------------
# Main simulation runner
# ---------------------------------------------------------------------------

def run_simulation(
    cfg: dict,
    num_clients: int,
    num_rounds: int,
    device_str: str = "cpu",
    ray_init_args: dict = None,
):
    """
    Chạy federated simulation hoàn toàn trong 1 process.

    Parameters
    ----------
    cfg         : dict từ config.yaml
    num_clients : tổng số virtual clients
    num_rounds  : số federated rounds
    device_str  : "cpu" | "cuda:0" | "cuda" ...
    ray_init_args : tuỳ chỉnh Ray (num_cpus, num_gpus, ...). None = tự động.
    """
    device = torch.device(device_str)
    print(f"\n{'='*60}")
    print(f"  Federated ProtoNet Simulation")
    print(f"  Clients : {num_clients}")
    print(f"  Rounds  : {num_rounds}")
    print(f"  Device  : {device}")
    print(f"{'='*60}\n")

    # ---- Kiểm tra số lượng dataset roots đủ không ----
    roots = cfg["data"]["client_dataset_roots"]
    if num_clients > len(roots):
        # Tự động replicate root cuối cùng (dùng cho test nhanh)
        print(
            f"[WARNING] num_clients={num_clients} > số roots={len(roots)}. "
            f"Các client còn lại sẽ dùng lại root cuối."
        )
        cfg["data"]["client_dataset_roots"] = roots + [roots[-1]] * (num_clients - len(roots))

    # ---- Khởi tạo global model từ checkpoint (warm-start) ----
    initial_parameters = None
    pretrained = cfg.get("pretrained", None)
    if pretrained and os.path.exists(pretrained):
        from federated_protonet.models import build_model, load_checkpoint_into_model, get_parameters
        print(f"[Server] Warm-start từ checkpoint: {pretrained}")
        model = build_model(cfg["model"])
        load_checkpoint_into_model(model, pretrained, torch.device("cpu"))
        initial_parameters = ndarrays_to_parameters(get_parameters(model))
        del model
    else:
        print("[Server] Không tìm thấy checkpoint, khởi tạo ngẫu nhiên.")

    # ---- FedAvg strategy ----
    server_cfg = cfg["server"]
    strategy = FedAvg(
        fraction_fit=server_cfg.get("fraction_fit", 1.0),
        fraction_evaluate=server_cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=min(server_cfg.get("min_fit_clients", 2), num_clients),
        min_evaluate_clients=min(server_cfg.get("min_evaluate_clients", 2), num_clients),
        min_available_clients=min(server_cfg.get("min_available_clients", 2), num_clients),
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=weighted_average_metrics,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
    )

    # ---- Client resources cho Ray ----
    # Kaggle GPU: 1 GPU tổng → mỗi client dùng 0.5 GPU (chạy 2 client song song)
    # Nếu chỉ có CPU thì để num_gpus=0.0
    has_gpu = torch.cuda.is_available()
    client_resources = {
        "num_cpus": max(1, os.cpu_count() // num_clients),
        "num_gpus": (1.0 / num_clients) if has_gpu else 0.0,
    }
    print(f"[Simulation] Client resources: {client_resources}")

    # ---- Ray init args mặc định cho Kaggle ----
    if ray_init_args is None:
        ray_init_args = {
            "ignore_reinit_error": True,
            "include_dashboard": False,  # tắt dashboard để tránh lỗi trên Kaggle
        }
        if has_gpu:
            ray_init_args["num_gpus"] = torch.cuda.device_count()

    # ---- Chạy simulation ----
    history = fl.simulation.start_simulation(
        client_fn=make_client_fn(cfg, device),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args=ray_init_args,
    )

    # ---- In kết quả ----
    print(f"\n{'='*60}")
    print("  Simulation hoàn tất!")
    print(f"{'='*60}")
    _print_history(history)

    # ---- Lưu history ----
    out_dir = cfg.get("output_dir", "federated_protonet/results")
    os.makedirs(out_dir, exist_ok=True)
    _save_history(history, out_dir)

    return history


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

def _print_history(history):
    print("\n--- Distributed Losses (server-side eval) ---")
    for rnd, loss in history.losses_distributed:
        print(f"  Round {rnd:3d} | loss = {loss:.4f}")

    print("\n--- Metrics Distributed ---")
    for metric_name, values in history.metrics_distributed.items():
        for rnd, val in values:
            print(f"  Round {rnd:3d} | {metric_name} = {val:.4f}")


def _save_history(history, out_dir: str):
    import json
    record = {
        "losses_distributed": history.losses_distributed,
        "metrics_distributed": {
            k: list(v) for k, v in history.metrics_distributed.items()
        },
        "losses_centralized": history.losses_centralized,
        "metrics_centralized": {
            k: list(v) for k, v in history.metrics_centralized.items()
        },
    }
    save_path = os.path.join(out_dir, "simulation_history.json")
    with open(save_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n[Simulation] History đã lưu → {save_path}")


# ---------------------------------------------------------------------------
# CLI (dùng khi %run simulation.py hoặc python simulation.py)
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Federated ProtoNet Simulation (Kaggle/Colab)")
    parser.add_argument("--config", default="federated_protonet/config.yaml")
    parser.add_argument("--num-clients", type=int, default=2,
                        help="Số virtual clients trong simulation")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Số federated rounds (override config.yaml)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU id. -1 = CPU. None = theo config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    setup_seed(cfg.get("seed", 42))

    # Device
    gpu_id = args.gpu if args.gpu is not None else cfg.get("gpu_id", 0)
    if gpu_id < 0 or not torch.cuda.is_available():
        device_str = "cpu"
    else:
        device_str = f"cuda:{gpu_id}"

    num_rounds = args.rounds or cfg["server"].get("num_rounds", 50)

    run_simulation(
        cfg=cfg,
        num_clients=args.num_clients,
        num_rounds=num_rounds,
        device_str=device_str,
    )


if __name__ == "__main__":
    main()
