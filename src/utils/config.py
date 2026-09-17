"""YAML 配置加载。

所有脚本通过 `--config` 接收一个 yaml 路径，再用这里的 load_config 读取，
使得切换数据集（1K -> 27K）只需换 config 文件，不改任何代码。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent


def project_path(rel: str | Path) -> Path:
    """把 config 里的相对路径解析为项目根目录下的绝对路径。"""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 yaml；若含 `data_config` 字段，则把被引用的配置并入 `data` 键下。"""
    cfg_path = project_path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    ref = cfg.get("data_config")
    if ref:
        cfg["data"] = load_config(ref)
    return cfg


def require(cfg: dict[str, Any], *keys: str) -> Any:
    """按层级取值，缺失或为 None 时报出完整路径，避免下游拿到 None 静默出错。"""
    node: Any = cfg
    for i, key in enumerate(keys):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"配置缺少 `{'.'.join(keys[: i + 1])}`")
        node = node[key]
    if node is None:
        raise ValueError(f"配置项 `{'.'.join(keys)}` 为 null，请先填写（见 results/schema_report.md）")
    return node
