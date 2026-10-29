"""比对两个 checkpoint 的权重是否**逐位相同**，用来证明某次改动是 no-op。

动机：给 `RetrievalData.batch()` 新增 `user_id` 这类"纯新增键"的改动，推理上不改变
任何数值（不进前向、不进 loss、不消耗随机数发生器）。但 rolling.py 那次已经说明，
"推理上是 no-op"和"实测是 no-op"是两件事，而一旦不是，已跑完的 seed 42 就全部作废。

用法（改动前后各跑一次 50 步，然后比）：

    python -m src.retrieval.two_tower --negative random --seed 42 \
        --max-steps 50 --no-eval --tag noop_before
    # ...改代码...
    python -m src.retrieval.two_tower --negative random --seed 42 \
        --max-steps 50 --no-eval --tag noop_after
    python scripts/prove_noop.py two_tower_noop_before two_tower_noop_after

比的是 state_dict 里每个张量的字节，不是文件字节 —— 文件里还存了 tag 等元信息，
那部分本来就不同。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import project_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    a = ap.parse_args()
    exp = project_path("experiments")
    ck = [torch.load(exp / f"{t}.pt", weights_only=False) for t in (a.before, a.after)]
    sd = [c["state_dict"] for c in ck]

    if set(sd[0]) != set(sd[1]):
        print(f"FAIL 参数名集合不同：{set(sd[0]) ^ set(sd[1])}")
        return 1
    diff = []
    for k in sd[0]:
        x, y = sd[0][k], sd[1][k]
        if x.shape != y.shape or not torch.equal(x, y):
            d = (
                (x.float() - y.float()).abs().max().item()
                if x.shape == y.shape
                else float("nan")
            )
            diff.append((k, tuple(x.shape), tuple(y.shape), d))
    for k, sx, sy, d in diff:
        print(f"FAIL {k}: {sx} vs {sy}，最大绝对差 {d:.3e}")
    lt = [c.get("loss_tail") for c in ck]
    same_loss = lt[0] == lt[1]
    print(f"{'OK  ' if same_loss else 'FAIL'} loss_tail {lt[0]!r} vs {lt[1]!r}")
    if diff or not same_loss:
        print(
            f"\n改动**不是** no-op：{len(diff)}/{len(sd[0])} 个张量不同。"
            "已跑完的结果不能再与新结果混在一张表里。"
        )
        return 1
    print(
        f"\nOK   {len(sd[0])} 个张量全部逐位相同，loss_tail 一致 —— 改动是 no-op，"
        "已跑完的 seed 可以继续用。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
