"""DIN-softmax target-aware history pooling 的 Single-Task 对照实验。

这里不复制训练循环：五个任务仍由 :mod:`src.ranking.single_task` 逐个独立训练，唯一结构
变化在公共 :class:`src.ranking.features.RankingEncoder` 的历史聚合。独立入口只负责三件事：

1. 默认读取 ``configs/din.yaml``；
2. 强制配置确实选择 ``history_pooling=din``；
3. 强制 ``normalization=masked_softmax``，并沿用已完成实验的
   ``din_single_task_*`` 产物名，绝不覆盖 masked-mean 基线或 DIN-raw。

正式训练前先运行 ``scripts/verify_din.py`` 和 DIN 口径的捷径扫描。预注册见
``results/din_prediction.json``。

用法：
    python -m src.ranking.din --seed 42
    python -m src.ranking.din --seed 42 --tasks click --limit-train 4096 --limit-eval 4096
"""

from __future__ import annotations

from src.ranking.single_task import main as single_task_main


def main() -> int:
    return single_task_main(
        default_config="configs/din.yaml",
        artifact_prefix="din_single_task",
        model_name="din_single_task",
        expected_history_pooling="din",
        expected_din_normalization="masked_softmax",
    )


if __name__ == "__main__":
    raise SystemExit(main())
