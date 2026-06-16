"""
Federated ProtoNet — SIMULATION mode (Kaggle / Colab)
=====================================================
Chạy toàn bộ federated learning trong 1 process, KHÔNG dùng Ray.
Tránh hoàn toàn lỗi "No module named ..." khi Ray serialize/deserialize.

Cách dùng trong Kaggle notebook:
    import sys
    sys.path.insert(0, '/kaggle/working/do_an1')

    from flower.simulation import run_simulation
    import yaml

    cfg = yaml.safe_load(open("/kaggle/working/do_an1/flower/config.yaml"))
    cfg["server"]["num_rounds"] = 5
    cfg["pretrained"] = None

    history = run_simulation(cfg, num_clients=2, num_rounds=5, device_str="cuda:0")
"""

import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import random
import copy
import json
import numpy as np
import torch
import yaml
from typing import List, Tuple, Dict, Any


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
# FedAvg aggregation (thay thế Flower FedAvg — chạy thuần Python)
# ---------------------------------------------------------------------------

def fedavg_aggregate(results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """
    Weighted average of model parameters (FedAvg).

    Parameters
    ----------
    results : list of (parameters, num_examples)
              parameters = list of numpy arrays (model weights)
              num_examples = số lượng ví dụ client đã train

    Returns
    -------
    Aggregated parameters (list of numpy arrays).
    """
    total_examples = sum(n for _, n in results)
    if total_examples == 0:
        return results[0][0]

    num_layers = len(results[0][0])
    aggregated = []
    for i in range(num_layers):
        weighted_sum = np.sum(
            [params[i] * (n / total_examples) for params, n in results],
            axis=0,
        )
        aggregated.append(weighted_sum)

    return aggregated


def aggregate_metrics(results: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Weighted average of metrics."""
    total = sum(n for n, _ in results)
    if total == 0:
        return {}
    aggregated: Dict[str, float] = {}
    for n, m in results:
        for k, v in m.items():
            aggregated[k] = aggregated.get(k, 0.0) + v * n / total
    return aggregated


# ---------------------------------------------------------------------------
# History container (tương thích cấu trúc Flower History)
# ---------------------------------------------------------------------------

class SimulationHistory:
    """Lưu kết quả qua các round, tương tự flwr.server.History."""

    def __init__(self):
        self.losses_distributed: List[Tuple[int, float]] = []
        self.metrics_distributed: Dict[str, List[Tuple[int, float]]] = {}
        self.losses_centralized: List[Tuple[int, float]] = []
        self.metrics_centralized: Dict[str, List[Tuple[int, float]]] = {}

    def add_loss_distributed(self, rnd: int, loss: float):
        self.losses_distributed.append((rnd, loss))

    def add_metrics_distributed(self, rnd: int, metrics: Dict[str, float]):
        for k, v in metrics.items():
            if k not in self.metrics_distributed:
                self.metrics_distributed[k] = []
            self.metrics_distributed[k].append((rnd, v))

    def add_metrics_centralized(self, rnd: int, metrics: Dict[str, float]):
        for k, v in metrics.items():
            if k not in self.metrics_centralized:
                self.metrics_centralized[k] = []
            self.metrics_centralized[k].append((rnd, v))


# ---------------------------------------------------------------------------
# Main simulation runner — chạy thuần trong 1 process, KHÔNG dùng Ray
# ---------------------------------------------------------------------------

def run_simulation(
    cfg: dict,
    num_clients: int,
    num_rounds: int,
    device_str: str = "cpu",
    ray_init_args: dict = None,   # giữ tham số này để backward-compatible, nhưng bỏ qua
):
    """
    Chạy federated simulation hoàn toàn trong 1 process (không Ray).

    Parameters
    ----------
    cfg         : dict từ config.yaml
    num_clients : tổng số virtual clients
    num_rounds  : số federated rounds
    device_str  : "cpu" | "cuda:0" | "cuda" ...
    """
    # Import từ cùng package (relative import — hoạt động dù folder tên gì)
    from .models import build_model, load_checkpoint_into_model, get_parameters, set_parameters
    from .flower_client import ProtoNetClient

    device = torch.device(device_str)
    print(f"\n{'='*60}")
    print(f"  Federated ProtoNet Simulation (single-process, no Ray)")
    print(f"  Clients : {num_clients}")
    print(f"  Rounds  : {num_rounds}")
    print(f"  Device  : {device}")
    print(f"{'='*60}\n")

    setup_seed(cfg.get("seed", 42))

    # ---- Kiểm tra số lượng dataset roots đủ không ----
    roots = cfg["data"]["client_dataset_roots"]
    if num_clients > len(roots):
        print(
            f"[WARNING] num_clients={num_clients} > số roots={len(roots)}. "
            f"Các client còn lại sẽ dùng lại root cuối."
        )
        cfg["data"]["client_dataset_roots"] = roots + [roots[-1]] * (num_clients - len(roots))

    # ---- Khởi tạo global parameters (warm-start từ checkpoint) ----
    pretrained = cfg.get("pretrained", None)
    if pretrained and os.path.exists(pretrained):
        print(f"[Server] Warm-start từ checkpoint: {pretrained}")
        model = build_model(cfg["model"])
        load_checkpoint_into_model(model, pretrained, torch.device("cpu"))
        global_params = get_parameters(model)
        del model
    else:
        print("[Server] Không tìm thấy checkpoint, khởi tạo ngẫu nhiên.")
        global_params = None   # sẽ lấy từ client đầu tiên

    # ---- Tạo tất cả clients ----
    print(f"\n[Simulation] Đang tạo {num_clients} clients ...")
    clients: List[ProtoNetClient] = []
    for cid in range(num_clients):
        print(f"  → Client {cid}")
        client = ProtoNetClient(client_id=cid, cfg=cfg, device=device)
        clients.append(client)

    # Nếu chưa có global params, lấy từ client đầu tiên
    if global_params is None:
        global_params = clients[0].get_parameters(config={})
        print("[Server] Khởi tạo global params từ Client 0.")

    # ---- History ----
    history = SimulationHistory()

    # ==================================================================
    #  FEDERATED ROUNDS
    # ==================================================================
    for rnd in range(1, num_rounds + 1):
        print(f"\n{'─'*60}")
        print(f"  ROUND {rnd}/{num_rounds}")
        print(f"{'─'*60}")

        # ---- FIT (mỗi client train local) ----
        fit_results = []
        for cid, client in enumerate(clients):
            updated_params, num_examples, fit_metrics = client.fit(
                parameters=global_params, config={}
            )
            fit_results.append((updated_params, num_examples, fit_metrics))
            print(
                f"  [Client {cid}] fit → "
                f"examples={num_examples}, "
                + ", ".join(f"{k}={v:.4f}" for k, v in fit_metrics.items())
            )

        # ---- AGGREGATE (FedAvg) ----
        global_params = fedavg_aggregate([
            (params, n) for params, n, _ in fit_results
        ])

        # Aggregate fit metrics
        agg_fit = aggregate_metrics([
            (n, m) for _, n, m in fit_results
        ])
        print(f"\n  [Server] Aggregated fit: "
              + ", ".join(f"{k}={v:.4f}" for k, v in agg_fit.items()))
        history.add_metrics_centralized(rnd, {f"fit_{k}": v for k, v in agg_fit.items()})

        # ---- EVALUATE (mỗi client evaluate với global params mới) ----
        eval_results = []
        for cid, client in enumerate(clients):
            loss, num_examples, eval_metrics = client.evaluate(
                parameters=global_params, config={}
            )
            eval_results.append((loss, num_examples, eval_metrics))
            print(
                f"  [Client {cid}] eval → "
                f"loss={loss:.4f}, "
                + ", ".join(f"{k}={v:.4f}" for k, v in eval_metrics.items())
            )

        # Aggregate eval
        total_eval = sum(n for _, n, _ in eval_results)
        avg_loss = sum(loss * n for loss, n, _ in eval_results) / max(total_eval, 1)
        agg_eval = aggregate_metrics([
            (n, m) for _, n, m in eval_results
        ])

        history.add_loss_distributed(rnd, avg_loss)
        history.add_metrics_distributed(rnd, agg_eval)

        print(f"\n  [Server] Aggregated eval: loss={avg_loss:.4f}, "
              + ", ".join(f"{k}={v:.4f}" for k, v in agg_eval.items()))

    # ==================================================================
    #  DONE
    # ==================================================================
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

def _print_history(history: SimulationHistory):
    print("\n--- Distributed Losses ---")
    for rnd, loss in history.losses_distributed:
        print(f"  Round {rnd:3d} | loss = {loss:.4f}")

    print("\n--- Distributed Metrics ---")
    for metric_name, values in history.metrics_distributed.items():
        for rnd, val in values:
            print(f"  Round {rnd:3d} | {metric_name} = {val:.4f}")


def _save_history(history: SimulationHistory, out_dir: str):
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
# CLI
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
