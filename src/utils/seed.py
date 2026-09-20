"""随机种子固定。

plan §26 要求最终实验跑 3 seeds 并报告 mean ± std，因此每个 run 的种子
必须显式设置且可记录。
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """统一设置 python / numpy / torch（若已安装）的随机源。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
