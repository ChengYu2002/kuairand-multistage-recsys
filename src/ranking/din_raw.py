"""不做 softmax 的 DIN-raw Single-Task 对照实验。

该入口复用 :mod:`src.ranking.single_task` 的完整训练协议，只把 DIN attention 的
normalization 从 ``masked_softmax`` 改为 ``raw``。它恢复原论文权重和不固定为 1 的
pooling 语义，但仍使用本项目的 ReLU 和普通 Adam weight decay，不声称完整复现 Dice / MBA。

正式训练前必须运行 ``scripts/verify_din.py`` 与 DIN-raw 口径的捷径扫描。
预注册见 ``results/din_raw_prediction.json``。
"""

from __future__ import annotations

from src.ranking.single_task import main as single_task_main


def main() -> int:
    return single_task_main(
        default_config="configs/din_raw.yaml",
        artifact_prefix="din_raw_single_task",
        model_name="din_raw_single_task",
        expected_history_pooling="din",
        expected_din_normalization="raw",
    )


if __name__ == "__main__":
    raise SystemExit(main())
