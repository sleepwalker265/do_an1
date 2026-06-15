"""
Data utilities for Federated ProtoNet.

Builds an episodic DataLoader by delegating to the existing
`create_torch_dataloader` infrastructure in the main codebase,
but constructing a lightweight yacs config on-the-fly from the
plain-dict config read from config.yaml.
"""

import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from yacs.config import CfgNode as CN
from data import create_torch_dataloader
from data.dataset_spec import Split


# ---------------------------------------------------------------------------
# Build a minimal yacs config that satisfies create_torch_dataloader
# ---------------------------------------------------------------------------

def _build_split_cfg(split_dict: dict, episode_cfg: dict) -> CN:
    """Create the DATA.TRAIN / DATA.VALID CN from a plain dict."""
    node = CN()
    node.BATCH_SIZE         = split_dict.get("batch_size", 1)
    node.DATASET_NAMES      = split_dict["dataset_names"]
    node.DATASET_ROOTS      = split_dict["dataset_roots"]
    node.SAMPLING_FREQUENCY = split_dict.get("sampling_frequency", [1.0])
    node.IS_EPISODIC        = episode_cfg.get("is_episodic", True)
    node.SHUFFLE            = split_dict.get("shuffle", True)
    node.ITERATION_PER_EPOCH = split_dict.get("iteration_per_epoch", None)

    ep = CN()
    ep.NUM_TASKS_PER_EPOCH               = episode_cfg.get("num_tasks_per_epoch", 100)
    ep.SEQUENTIAL_SAMPLING               = episode_cfg.get("sequential_sampling", 0)
    ep.NUM_WAYS                          = episode_cfg.get("num_ways", 5)
    ep.NUM_SUPPORT                       = episode_cfg.get("num_support", 5)
    ep.NUM_QUERY                         = episode_cfg.get("num_query", 15)
    ep.MIN_WAYS                          = episode_cfg.get("min_ways", 5)
    ep.MAX_WAYS_UPPER_BOUND              = episode_cfg.get("max_ways_upper_bound", 50)
    ep.MAX_NUM_QUERY                     = episode_cfg.get("max_num_query", 15)
    ep.MIN_EXAMPLES_IN_CLASS             = episode_cfg.get("min_examples_in_class", 0)
    ep.MAX_SUPPORT_SET_SIZE              = episode_cfg.get("max_support_set_size", 500)
    ep.MAX_SUPPORT_SIZE_CONTRIB_PER_CLASS= episode_cfg.get("max_support_size_contrib_per_class", 100)
    ep.MIN_LOG_WEIGHT                    = episode_cfg.get("min_log_weight", -0.693147)
    ep.MAX_LOG_WEIGHT                    = episode_cfg.get("max_log_weight",  0.693147)
    ep.USE_DAG_HIERARCHY                 = episode_cfg.get("use_dag_hierarchy", False)
    ep.USE_BILEVEL_HIERARCHY             = episode_cfg.get("use_bilevel_hierarchy", False)

    node.EPISODE_DESCR_CONFIG = ep
    return node


def build_yacs_config(cfg: dict, client_id: int, split: Split) -> CN:
    """
    Construct a full yacs CfgNode compatible with create_torch_dataloader.

    Parameters
    ----------
    cfg       : the top-level dict from config.yaml
    client_id : index into cfg['data']['client_dataset_roots']
    split     : Split.TRAIN | Split.VALID | Split.TEST
    """
    data_cfg     = cfg["data"]
    episode_cfg  = data_cfg["episode"]
    client_roots = data_cfg["client_dataset_roots"]

    if client_id >= len(client_roots):
        raise ValueError(
            f"client_id={client_id} but only {len(client_roots)} roots defined "
            f"in config.yaml → data.client_dataset_roots"
        )

    dataset_root = client_roots[client_id]
    dataset_name = data_cfg["dataset_name"]

    # ---- augmentation ----
    aug = CN()
    aug.MEAN            = data_cfg["mean"]
    aug.STD             = data_cfg["std"]
    aug.COLOR_JITTER    = None
    aug.GRAY_SCALE      = None
    aug.GAUSSIAN_BLUR   = None
    aug.FLIP            = 0.5
    aug.TEST_CROP       = False

    # ---- per-split dataset config ----
    split_dict = {
        "dataset_names"  : [dataset_name],
        "dataset_roots"  : [dataset_root],
        "batch_size"     : 1 if split == Split.TRAIN else 4,
        "shuffle"        : split == Split.TRAIN,
        "iteration_per_epoch": None,
    }

    # For validation reduce num_tasks
    ep_cfg = dict(episode_cfg)
    if split == Split.VALID:
        ep_cfg["num_tasks_per_epoch"] = max(30, episode_cfg.get("num_tasks_per_epoch", 100) // 3)

    root = CN()
    root.AUG  = aug
    root.DATA = CN()
    root.DATA.IMG_SIZE         = data_cfg["img_size"]
    root.DATA.NUM_WORKERS      = data_cfg.get("num_workers", 0)
    root.DATA.PIN_MEMORY       = data_cfg.get("pin_memory", False)
    root.DATA.DATASET_ROOT     = dataset_root
    root.DATA.PATH_TO_WORDS    = data_cfg.get("path_to_words", "data/words.txt")
    root.DATA.PATH_TO_IS_A     = data_cfg.get("path_to_is_a", "data/wordnet.is_a.txt")
    root.DATA.PATH_TO_NUM_LEAF_IMAGES = data_cfg.get(
        "path_to_num_leaf_images", "data/ImageNet_num_images_perclass.json"
    )
    root.DATA.TRAIN_SPLIT_ONLY = False

    root.DATA.TRAIN = _build_split_cfg(split_dict, ep_cfg)
    root.DATA.VALID = _build_split_cfg(
        {**split_dict, "batch_size": 4, "shuffle": False}, ep_cfg
    )
    root.DATA.TEST  = _build_split_cfg(
        {**split_dict, "batch_size": 4, "shuffle": False}, ep_cfg
    )

    return root


def get_dataloaders(cfg: dict, client_id: int):
    """
    Return (train_loader, train_dataset, val_loader, val_dataset).
    """
    yacs_cfg = build_yacs_config(cfg, client_id, Split.TRAIN)

    train_loader, train_dataset = create_torch_dataloader(Split.TRAIN, yacs_cfg)
    val_loader,   val_dataset   = create_torch_dataloader(Split.VALID, yacs_cfg)

    return train_loader, train_dataset, val_loader, val_dataset
