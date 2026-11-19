"""DIN-softmax / DIN-raw 历史聚合的独立结构验算（不训练、不读取 test 标签）。

它专门拦截几类“代码能跑、AUC 也会出，但实验问题已经变了”的静默失败：

1. 旧配置因为新增默认项而改变输出或 fingerprint，导致三 seed 基线失效；
2. attention 忘记 mask，或全空历史对全 -inf softmax 产生 NaN；
3. query 只使用目标 ID，使 test 段 89.2% 的 OOV 候选共享同一个 query；
4. DIN 入口沿用 single_task 产物名，覆盖既有 masked-mean 结果；
5. config 除聚合函数外还悄悄改了训练预算或下游 MLP。

捷径强度扫描不在这里复制实现。运行完本脚本后还必须执行：

    python scripts/verify_ranking_data.py --encoder-config configs/din.yaml
    python scripts/verify_ranking_data.py --encoder-config configs/din_raw.yaml

该命令复用现有自维护扫描器，实测哪些 batch 键经新 attention 路径影响 x。
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.diagnose_din_attention import (
    MAX_RAW_POOL_L2_P95,
    MIN_HISTORY_EMBEDDING_NORM_RATIO,
    MIN_PARAMETER_NORM_RATIO,
    MIN_RAW_ABS_WEIGHT_RATIO,
    MIN_SPREAD_RATIO,
)
from src.ranking.dataset import OOV, RankingData, RankItemStatic
from src.ranking.features import (
    DINHistoryPool,
    RankingEncoder,
    count_params,
    masked_mean,
)
from src.ranking.single_task import artifact_tag, checkpoint_tag
from src.utils.config import load_config, project_path, require
from src.utils.seed import set_seed

EXPECTED_BASELINE_SHA = "df312ebad132ea41"
PROTOCOL_KEYS = (
    "tasks",
    "embedding_dim",
    "optimizer",
    "lr",
    "batch_size",
    "epochs",
    "dropout",
    "weight_decay",
    "eval_every_steps",
    "seeds",
    "tower_hidden",
)


def _clone_batch(b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in b.items()}


def _nonzero_grad(p: torch.Tensor | None) -> bool:
    return (
        p is not None and bool(torch.isfinite(p).all()) and float(p.abs().sum()) > 0.0
    )


def main() -> int:
    failures: list[str] = []
    checks = 0

    def check(ok: bool, msg: str) -> None:
        nonlocal checks
        checks += 1
        print(f"  {'OK  ' if ok else 'FAIL'} {msg}")
        if not ok:
            failures.append(msg)

    base_cfg = load_config("configs/single_task.yaml")
    din_cfg = load_config("configs/din.yaml")
    raw_cfg = load_config("configs/din_raw.yaml")

    print("\n=== A. 预注册与受控配置 ===")
    pred_path = ROOT / "results" / "din_prediction.json"
    pred = json.loads(pred_path.read_text("utf-8"))
    check(
        pred["primary_metric"]["name"] == "valid AUC",
        "预注册在训练前固定主指标为 valid AUC",
    )
    check(
        "|mean(delta)| / std(delta) >= 2.0" in pred["primary_metric"]["bar"]
        and "3/3 seeds" in pred["primary_metric"]["bar"],
        "预注册固定 2.0 + 3/3 同号门槛",
    )
    check(
        pred["contrast"]["seeds"] == [42, 43, 44]
        and len(pred["contrast"]["tasks"]) == 5,
        "正式范围在训练前固定为五任务 x 3 seeds",
    )
    raw_pred = json.loads(
        (ROOT / "results" / "din_raw_prediction.json").read_text("utf-8")
    )
    survival = raw_pred["module_survival_gate"]
    check(
        survival["minimum_parameter_norm_ratio_to_initial"] == MIN_PARAMETER_NORM_RATIO
        and survival["minimum_history_embedding_norm_ratio_to_initial"]
        == MIN_HISTORY_EMBEDDING_NORM_RATIO
        and survival["minimum_p95_within_row_weight_spread_ratio_to_initial"]
        == MIN_SPREAD_RATIO
        and survival["minimum_mean_abs_valid_weight_ratio_to_initial"]
        == MIN_RAW_ABS_WEIGHT_RATIO,
        "DIN-raw 预注册与诊断代码使用同一组 attention / 历史表存活下界",
    )
    check(
        survival["masked_mean_reference_pool_l2_p95_max"] == 0.05401233956217766
        and survival["maximum_raw_pool_l2_p95_multiplier"] == 100.0
        and survival["maximum_raw_pool_l2_p95"] == MAX_RAW_POOL_L2_P95,
        "DIN-raw 预注册与诊断代码使用同一个有限数值爆炸上界",
    )

    protocol_diff = {
        k: (base_cfg.get(k), din_cfg.get(k))
        for k in PROTOCOL_KEYS
        if base_cfg.get(k) != din_cfg.get(k)
    }
    check(not protocol_diff, f"训练协议与下游塔逐字相同（差异 {protocol_diff}）")
    base_input = dict(require(base_cfg, "input"))
    din_input = dict(require(din_cfg, "input"))
    din_only = {"history_pooling", "din"}
    input_diff = {
        k: (base_input.get(k), din_input.get(k))
        for k in set(base_input) | (set(din_input) - din_only)
        if base_input.get(k) != din_input.get(k)
    }
    check(not input_diff, f"除 DIN 配置外输入口径相同（差异 {input_diff}）")
    check(
        din_input.get("history_pooling") == "din"
        and din_input.get("din", {}).get("attention_hidden") == [64, 32]
        and din_input.get("din", {}).get("normalization") == "masked_softmax",
        "DIN-softmax 结构显式写入 config，不依赖代码默认值",
    )
    raw_protocol_diff = {
        k: (din_cfg.get(k), raw_cfg.get(k))
        for k in PROTOCOL_KEYS
        if din_cfg.get(k) != raw_cfg.get(k)
    }
    check(
        not raw_protocol_diff,
        f"DIN-raw 与 DIN-softmax 训练协议逐字相同（差异 {raw_protocol_diff}）",
    )
    soft_input_cmp = copy.deepcopy(require(din_cfg, "input"))
    raw_input_cmp = copy.deepcopy(require(raw_cfg, "input"))
    raw_norm = raw_input_cmp["din"].pop("normalization", None)
    soft_norm = soft_input_cmp["din"].pop("normalization", None)
    check(
        soft_input_cmp == raw_input_cmp
        and soft_norm == "masked_softmax"
        and raw_norm == "raw",
        "DIN-raw 唯一输入配置差异是 normalization: masked_softmax -> raw",
    )

    print("\n=== B. 产物命名不覆盖基线 ===")
    base_tag = artifact_tag("single_task", "main", 42)
    din_tag = artifact_tag("din_single_task", "main", 42)
    check(
        base_tag == "single_task_main_seed42"
        and din_tag == "din_single_task_main_seed42"
        and base_tag != din_tag,
        f"汇总产物隔离：{base_tag} vs {din_tag}",
    )
    base_ckpt = checkpoint_tag("single_task", "is_click", "main", 42)
    din_ckpt = checkpoint_tag("din_single_task", "is_click", "main", 42)
    check(
        base_ckpt != din_ckpt and din_ckpt == "din_single_task_is_click_main_seed42",
        f"checkpoint 隔离：{base_ckpt} vs {din_ckpt}",
    )
    check(
        artifact_tag("din_single_task", "main", 42, smoke=True).endswith("_smoke"),
        "烟测产物带 _smoke，不覆盖正式 DIN 结果",
    )
    raw_tag = artifact_tag("din_raw_single_task", "main", 42)
    raw_ckpt = checkpoint_tag("din_raw_single_task", "is_click", "main", 42)
    check(
        raw_tag == "din_raw_single_task_main_seed42"
        and raw_ckpt == "din_raw_single_task_is_click_main_seed42"
        and raw_tag not in {base_tag, din_tag},
        f"DIN-raw 产物独立：{raw_tag} / {raw_ckpt}",
    )

    print("\n=== C. 旧 masked-mean 路径逐位 no-op ===")
    proc = project_path(require(base_cfg, "data", "dataset", "processed_dir"))
    st = RankItemStatic(proc, "main")
    ic = require(base_cfg, "input")
    data = RankingData(
        proc,
        "main",
        "train",
        st,
        max_hist=int(ic["max_hist"]),
        mask_oov=bool(ic["mask_oov_in_history"]),
        pair_features=bool(ic["pair_features"]),
    )
    rows = np.arange(min(512, len(data)))
    b = data.batch(rows)

    explicit_mean = copy.deepcopy(base_cfg)
    explicit_mean["input"]["history_pooling"] = "masked_mean"
    set_seed(123)
    old = RankingEncoder(data, base_cfg)
    set_seed(123)
    explicit = RankingEncoder(data, explicit_mean)
    old.eval(), explicit.eval()
    with torch.no_grad():
        x_old, x_explicit = old(b), explicit(b)
    same_state = old.state_dict().keys() == explicit.state_dict().keys() and all(
        torch.equal(v, explicit.state_dict()[k]) for k, v in old.state_dict().items()
    )
    check(same_state, "缺省配置与显式 masked_mean 的参数逐位相同")
    check(torch.equal(x_old, x_explicit), "缺省配置与显式 masked_mean 的输出逐位相同")
    check(
        old.fingerprint()["sha"]
        == explicit.fingerprint()["sha"]
        == EXPECTED_BASELINE_SHA,
        f"旧 fingerprint 保持 {EXPECTED_BASELINE_SHA}",
    )

    print("\n=== D. DIN 形状、mask 与空历史 ===")
    set_seed(123)
    din = RankingEncoder(data, din_cfg)
    din.eval()
    with torch.no_grad():
        x_din = din(b)
        weights = din.attention_weights(b)
    check(
        x_din.shape == x_old.shape and din.out_dim == old.out_dim == 323,
        f"只替换 32 维 hist_pool：总输入仍为 {din.out_dim} 维",
    )
    check(
        torch.isfinite(x_din).all() and torch.isfinite(weights).all(),
        "DIN 输出与权重没有 NaN/Inf",
    )
    check(
        weights.shape == b["hist_mask"].shape,
        f"权重形状 {tuple(weights.shape)} == hist_mask",
    )
    check(
        float(weights[~b["hist_mask"]].abs().max()) == 0.0,
        "PAD / 历史 OOV 的 attention 权重严格为 0",
    )
    nonempty = b["hist_mask"].any(dim=1)
    sums = weights.sum(dim=1)
    check(
        torch.allclose(sums[nonempty], torch.ones_like(sums[nonempty]), atol=1e-6),
        "非空历史的有效权重和为 1",
    )

    unit = DINHistoryPool(hist_dim=4, query_dim=3, attention_hidden=[5])
    hist = torch.randn(3, 4, 4)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.bool)
    query = torch.randn(3, 3)
    for p in unit.local_activation.parameters():
        p.data.zero_()
    pooled, uniform = unit(hist, mask, query, return_weights=True)
    check(
        torch.allclose(pooled, masked_mean(hist, mask), atol=1e-7),
        "attention logits 全相等时严格退化为 masked mean（对照可证伪）",
    )
    check(
        torch.equal(pooled[2], torch.zeros_like(pooled[2]))
        and float(uniform[2].sum().detach()) == 0.0,
        "全空历史输出零向量、权重和为 0",
    )

    changed_pad = _clone_batch(b)
    changed_pad["hist"][~changed_pad["hist_mask"]] = 2
    with torch.no_grad():
        x_changed_pad = din(changed_pad)
    check(
        torch.equal(x_din, x_changed_pad),
        "任意修改被 mask 的历史 ID，完整 Encoder 输出逐位不变",
    )

    print("\n=== E. query 真的依赖候选，且 OOV 仍可区分 ===")
    # 同一段历史配两个不同目标；其他用户侧特征不影响 attention，可保持原值。
    target_keys = ("item_idx", "author", "tags", "duration", "age")
    pair = None
    for j in range(1, len(rows)):
        if b["hist_mask"][0].any() and any(
            not torch.equal(b[k][0], b[k][j]) for k in target_keys
        ):
            pair = j
            break
    check(pair is not None, "抽样中存在历史非空且目标侧信息不同的候选对")
    probe = {k: v[[0, pair]].clone() for k, v in b.items()}
    probe["hist"][1] = probe["hist"][0]
    probe["hist_mask"][1] = probe["hist_mask"][0]
    probe["hist_len"][1] = probe["hist_len"][0]
    with torch.no_grad():
        w_pair = din.attention_weights(probe)
    check(
        not torch.equal(w_pair[0], w_pair[1]),
        "同一历史面对不同候选产生不同 attention 权重",
    )

    oov_rows: list[int] = []
    oov_side = None
    for start in range(0, min(len(data), 50000), 4096):
        rr = np.arange(start, min(start + 4096, len(data)))
        bb = data.batch(rr)
        candidates = (
            torch.nonzero(
                (bb["item_idx"] == OOV) & bb["hist_mask"].any(dim=1), as_tuple=False
            )
            .flatten()
            .tolist()
        )
        for pos in candidates:
            side = (
                int(bb["author"][pos]),
                tuple(bb["tags"][pos].tolist()),
                float(bb["duration"][pos]),
                float(bb["age"][pos]),
            )
            if not oov_rows:
                oov_rows = [int(rr[pos])]
                oov_side = side
            elif side != oov_side:
                oov_rows.append(int(rr[pos]))
                break
        if len(oov_rows) == 2:
            break
    check(len(oov_rows) == 2, "找到两个目标 ID 同为 OOV、侧信息不同且历史非空的样本")
    ob = data.batch(np.array(oov_rows, dtype=np.int64))
    ob["hist"][1] = ob["hist"][0]
    ob["hist_mask"][1] = ob["hist_mask"][0]
    ob["hist_len"][1] = ob["hist_len"][0]
    with torch.no_grad():
        ow = din.attention_weights(ob)
    check(
        int(ob["item_idx"][0]) == OOV == int(ob["item_idx"][1])
        and not torch.equal(ow[0], ow[1]),
        "目标 ID 都是 OOV 时，作者/标签/数值侧信息仍使 attention 不同",
    )

    # video_id_known 与 item_idx 是否 OOV 完全等价，不让它在 query 里重复打开热度路径；
    # 元数据 known flags 则必须真的参与 attention，才能区分“值为 0”和“字段缺失”。
    no_video_flag = _clone_batch(b)
    no_video_flag["video_id_known"] = 1.0 - no_video_flag["video_id_known"]
    author_flag = _clone_batch(b)
    author_flag["author_known"] = 1.0 - author_flag["author_known"]
    with torch.no_grad():
        w0 = din.attention_weights(b)
        w_video = din.attention_weights(no_video_flag)
        w_author = din.attention_weights(author_flag)
    check(
        torch.equal(w0, w_video),
        "video_id_known 不走 attention 新路径（原有 x 标志通道仍保留）",
    )
    check(
        not torch.equal(w0, w_author),
        "author_known 等元数据缺失标志确实影响 attention query",
    )

    print("\n=== F. 标签隔离、梯度与确定性 ===")
    relabeled = _clone_batch(b)
    for k in relabeled:
        if k.startswith("label_"):
            relabeled[k] = 1.0 - relabeled[k]
    with torch.no_grad():
        check(
            torch.equal(x_din, din(relabeled)), "修改所有 label_* 不改变 Encoder 输出"
        )

    grad_b = data.batch(np.arange(128))
    din.train()
    din.zero_grad(set_to_none=True)
    loss = din(grad_b)[:, :32].square().mean()
    loss.backward()
    check(_nonzero_grad(din.hist_emb.weight.grad), "历史 embedding 收到梯度")
    check(
        _nonzero_grad(din.item_emb.weight.grad)
        and _nonzero_grad(din.author_emb.weight.grad)
        and _nonzero_grad(din.tag_emb.weight.grad),
        "目标 ID / 作者 / 标签 query 分支都收到梯度",
    )
    check(
        din.din is not None
        and all(_nonzero_grad(p.grad) for p in din.din.parameters()),
        "query projection 与 attention MLP 的全部参数都收到有限非零梯度",
    )

    direct = DINHistoryPool(hist_dim=4, query_dim=3, attention_hidden=[5])
    hh = torch.randn(2, 4, 4, requires_grad=True)
    qq = torch.randn(2, 3, requires_grad=True)
    mm = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.bool)
    direct(hh, mm, qq).square().sum().backward()
    check(float(hh.grad[~mm].abs().max()) == 0.0, "被 mask 的历史位置梯度严格为 0")

    set_seed(777)
    d1 = RankingEncoder(data, din_cfg).eval()
    set_seed(777)
    d2 = RankingEncoder(data, din_cfg).eval()
    with torch.no_grad():
        y1, y2 = d1(b), d2(b)
    check(torch.equal(y1, y2), "同 seed 两个 DIN Encoder 输出逐位相同")
    check(
        din.fingerprint()["sha"] != EXPECTED_BASELINE_SHA
        and din.fingerprint()["history_pooling"] == "din",
        f"DIN 指纹独立于旧基线：{din.fingerprint()['sha']}",
    )

    p_old, p_din = count_params(old), count_params(din)
    dense_delta = p_din["dense"] - p_old["dense"]
    check(
        0 < dense_delta < 50000 and p_din["embedding"] == p_old["embedding"],
        f"只增加少量 dense 参数：+{dense_delta:,}；embedding 参数不变",
    )

    print("\n=== G. DIN-raw 只改变归一化语义 ===")
    set_seed(888)
    soft_same_init = RankingEncoder(data, din_cfg).eval()
    set_seed(888)
    raw = RankingEncoder(data, raw_cfg).eval()
    same_parameters = (
        soft_same_init.state_dict().keys() == raw.state_dict().keys()
        and all(
            torch.equal(v, raw.state_dict()[k])
            for k, v in soft_same_init.state_dict().items()
        )
    )
    check(same_parameters, "同 seed 下 softmax / raw 参数结构与初值逐位相同")

    soft_fp = copy.deepcopy(soft_same_init.fingerprint())
    raw_fp = copy.deepcopy(raw.fingerprint())
    soft_sha, raw_sha = soft_fp.pop("sha"), raw_fp.pop("sha")
    soft_fp["din"]["normalization"] = "raw"
    check(
        soft_fp == raw_fp and soft_sha != raw_sha,
        f"两臂 fingerprint 只差 normalization（{soft_sha} vs {raw_sha}）",
    )

    with torch.no_grad():
        x_raw = raw(b)
        w_raw_model = raw.attention_weights(b)
    check(
        x_raw.shape == x_din.shape
        and torch.isfinite(x_raw).all()
        and torch.isfinite(w_raw_model).all(),
        "DIN-raw 完整 Encoder 形状不变且没有 NaN/Inf",
    )
    check(
        float(w_raw_model[~b["hist_mask"]].abs().max()) == 0.0,
        "DIN-raw 的 PAD / 历史 OOV 权重严格为 0",
    )

    raw_unit = DINHistoryPool(
        hist_dim=4, query_dim=3, attention_hidden=[5], normalization="raw"
    )
    for p in raw_unit.parameters():
        p.data.zero_()
    raw_unit.local_activation[-1].bias.data.fill_(1.0)
    raw_pooled, raw_weights = raw_unit(hist, mask, query, return_weights=True)
    expected_sum = (hist * mask.unsqueeze(-1)).sum(dim=1)
    check(
        torch.equal(raw_weights, mask.to(raw_weights.dtype))
        and torch.allclose(raw_pooled, expected_sum, atol=1e-7),
        "常数 raw 权重得到历史求和而非均值（权重和不固定为 1）",
    )
    check(
        torch.equal(raw_pooled[2], torch.zeros_like(raw_pooled[2]))
        and torch.isfinite(raw_pooled).all(),
        "DIN-raw 全空历史严格输出零向量且无 NaN/Inf",
    )

    raw_h = torch.randn(2, 4, 4, requires_grad=True)
    raw_q = torch.randn(2, 3, requires_grad=True)
    raw_m = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.bool)
    raw_direct = DINHistoryPool(
        hist_dim=4, query_dim=3, attention_hidden=[5], normalization="raw"
    )
    raw_direct(raw_h, raw_m, raw_q).square().sum().backward()
    check(
        float(raw_h.grad[~raw_m].abs().max()) == 0.0,
        "DIN-raw 被 mask 的历史位置梯度严格为 0",
    )

    print(f"\n共校验 {checks} 项，失败 {len(failures)} 项。")
    if failures:
        print("DIN 验算未通过：")
        for msg in failures:
            print(f"  - {msg}")
        return 1
    print(
        "DIN-softmax / DIN-raw 结构验算通过；DIN-raw 尚未训练。"
        "下一步必须先做 DIN-raw 捷径扫描。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
