"""独立验算 Single-Task 基线（src/ranking/single_task.py + trainer.py + features.py）。

Single-Task 是 §27.1 里 ΔAUC 的**分母**。它若偏低，四个 MTL 模型会集体显得有正迁移；
偏高则集体显得有负迁移。而这两种错都不会报错、不会 NaN，只会让整张对比表偏移。

八层检查：
  A. 任务名映射：简称 -> 真实列名必须唯一；歧义或缺不到一律报错（猜错列名会训练出一个
     预测别的任务的模型，而它照样收敛、照样出 AUC）。
  B. 输入里没有标签：把 batch 里的 label_* 全部替换成随机值，x 必须**逐位不变**。
  C. 标签置换检验（最有力的泄漏检验）：把训练标签整体打乱后重训，在**没训过的行**上
     AUC 必须掉回 0.5 附近。两个坑都踩过，记在这里：

     1) 必须用留出行。这个模型有 3,100 万参数、含逐视频 ID embedding，在训练集上它会
        把打乱后的标签直接背下来（实测训练集 AUC 0.98），拿训练集评等于什么都没验。
     2) 留出行必须**随机抽**，不能切连续块。samples 表按 user_id 排序，连续切等于
        「训前 5 个用户、评另外 7 个用户」—— 有效样本量是 7 个用户而不是 2 万行，
        AUC 抖动极大。实测同一份代码：连续块切法打乱标签后得 0.6066（看着像泄漏），
        随机抽行得 0.5147（干净）。
  D. 有学习能力（可证伪）：小批量过拟合，loss 必须大幅下降、train AUC 逼近 1。
     跑不起来说明结构或量纲有问题，而不是"数据难"。
  E. 五个模型真的独立：任意两个任务的模型不共享任何参数张量；同 seed 下初值相同。
  F. 数据顺序与模型无关：epoch_order 只依赖 (seed, epoch, n)，所以 ST 的第 k 个任务
     与 MMoE 看到的是同一个顺序 —— 否则"同样 5 轮"里还藏着一个变量。
  G. 固定轮数规则被写死：结果里的 checkpoint_rule 必须是 fixed_epochs_last，
     且训练过程中没有任何按 valid 指标选择 checkpoint 的行为。
  H. 确定性与指纹：同 seed 两次训练逐位相同；输入指纹稳定，且改了输入口径指纹会变。

用法：
    python scripts/verify_single_task.py
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.ranking_metrics import auc
from src.ranking.dataset import RankingData, RankItemStatic
from src.ranking.single_task import SingleTaskDNN, resolve_tasks
from src.ranking.trainer import epoch_order
from src.utils.config import load_config, project_path, require
from src.utils.seed import set_seed


def _fit(model, data, rows, labels, task, steps, lr=1e-3, seed=0,
         held=None, held_labels=None):
    """在 rows 上跑 steps 步。

    返回 (首 loss, 末 loss, 训练集 AUC, 留出行 AUC)。留出行没给时最后一项是 nan。
    训练集 AUC 只能用来判断"模型有没有学习能力"（D 段）；判断泄漏必须看留出行 ——
    这个模型能把训练行的标签背下来，训练集 AUC 对泄漏毫无鉴别力。
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    bs = min(512, len(rows))
    first = last = None
    for i in range(steps):
        pick = rows[rng.integers(0, len(rows), bs)]
        b = data.batch(pick)
        y = torch.from_numpy(labels[pick])
        loss = F.binary_cross_entropy_with_logits(model(b)[task], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss)
        last = float(loss)
    with torch.no_grad():
        p = torch.sigmoid(model(data.batch(rows))[task]).numpy()
        a_tr = auc(p, labels[rows].astype(np.int64))
        a_ho = float("nan")
        if held is not None:
            q = torch.sigmoid(model(data.batch(held))[task]).numpy()
            a_ho = auc(q, held_labels[held].astype(np.int64))
    return first, last, a_tr, a_ho


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/single_task.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cfg = load_config(a.config)
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))

    failures: list[str] = []
    checks = 0

    def check(ok: bool, msg: str) -> None:
        nonlocal checks
        checks += 1
        print(f"  {'OK  ' if ok else 'FAIL'} {msg}")
        if not ok:
            failures.append(msg)

    st = RankItemStatic(proc, a.protocol)
    # 必须按 config 构造：编码器会断言两边口径一致，默认值碰巧相同不算接线正确
    ic = require(cfg, "input")
    data = RankingData(proc, a.protocol, "train", st, max_hist=int(ic["max_hist"]),
                       mask_oov=bool(ic["mask_oov_in_history"]))
    label_cols = require(cfg, "data", "labels", "tasks")

    # ---------- A. 任务名映射 ----------
    print("\n=== A. 任务名映射 ===")
    got = resolve_tasks(list(require(cfg, "tasks")), label_cols)
    check(got == ["is_click", "long_view", "is_like", "is_follow", "is_comment"],
          f"config 简称 -> 真实列名：{got}")
    for bad, why in (("clicked", "不存在的简称"), ("", "空简称")):
        try:
            resolve_tasks([bad], label_cols)
            check(False, f"{why} {bad!r}：未被拦下")
        except ValueError:
            check(True, f"{why} {bad!r}：已拦下")
    # 歧义：标签列里同时有 x 和 is_x 时必须报错
    try:
        resolve_tasks(["click"], ["click", "is_click"])
        check(False, "歧义（同时存在 click 与 is_click）：未被拦下")
    except ValueError:
        check(True, "歧义（同时存在 click 与 is_click）：已拦下")

    # ---------- B. 输入里没有标签 ----------
    print("\n=== B. 输入里没有标签 ===")
    set_seed(a.seed)
    m = SingleTaskDNN(data, cfg, "is_click")
    rows = np.arange(4096)
    b = data.batch(rows)
    with torch.no_grad():
        x0 = m.encoder(b).clone()
    b2 = dict(b)
    rng = np.random.default_rng(0)
    for k in [k for k in b2 if k.startswith("label_")]:
        b2[k] = torch.from_numpy(rng.random(len(rows)).astype(np.float32))
    with torch.no_grad():
        x1 = m.encoder(b2)
    check(torch.equal(x0, x1),
          f"把 {len([k for k in b if k.startswith('label_')])} 个 label_* 换成随机值，"
          "x 逐位不变 -> 输入不含标签")

    # ---------- C. 标签置换检验 ----------
    print("\n=== C. 标签置换检验（在随机留出行上，打乱标签后 AUC 必须回到 0.5）===")
    # 随机抽而不是切连续块：samples 按 user_id 排序，连续切会变成用户不相交的退化切分
    split_rng = np.random.default_rng(1)
    idx = split_rng.choice(len(data), 40000, replace=False)
    sub, held = idx[:20000], idx[20000:]         # 留出行从未参与训练
    y_true = data.labels["is_click"].copy()
    y_shuf = y_true.copy()
    rng = np.random.default_rng(7)
    y_shuf[sub] = y_shuf[sub][rng.permutation(len(sub))]
    set_seed(a.seed)
    _, _, tr_shuf, ho_shuf = _fit(SingleTaskDNN(data, cfg, "is_click"), data, sub, y_shuf,
                                  "is_click", steps=200, held=held, held_labels=y_true)
    set_seed(a.seed)
    _, _, _, ho_real = _fit(SingleTaskDNN(data, cfg, "is_click"), data, sub, y_true,
                                  "is_click", steps=200, held=held, held_labels=y_true)
    # 未训练基线。单个初值不保证接近 0.5 —— 随机投影可能碰巧与标签相关（实测某个
    # seed 下 0.4445），所以取 3 个初值的均值，容差 0.06。
    a0s = []
    for sd in (a.seed, a.seed + 1, a.seed + 2):
        set_seed(sd)
        with torch.no_grad():
            q0 = torch.sigmoid(
                SingleTaskDNN(data, cfg, "is_click")(data.batch(held))["is_click"])
        a0s.append(auc(q0.numpy(), y_true[held].astype(np.int64)))
    a0 = float(np.mean(a0s))
    check(abs(a0 - 0.5) < 0.06,
          f"未训练模型留出行 AUC 3 个初值均值 {a0:.4f} ≈ 0.5"
          f"（单值 {[f'{v:.4f}' for v in a0s]}）")
    check(ho_real > 0.58, f"真标签：留出行 AUC {ho_real:.4f} > 0.58（特征确实带信号）")
    check(abs(ho_shuf - 0.5) < 0.05,
          f"打乱标签：留出行 AUC {ho_shuf:.4f} ≈ 0.5 -> 输入里没有标签的影子")
    check(ho_real - ho_shuf > 0.15,
          f"两者差 {ho_real - ho_shuf:.4f} > 0.15（信号来自特征，不是来自泄漏）")
    check(tr_shuf > 0.8,
          f"顺带确认：打乱标签在**训练集**上仍有 AUC {tr_shuf:.4f} —— "
          "模型会背行，所以泄漏检验只能看留出行")

    # ---------- D. 有学习能力 ----------
    print("\n=== D. 小批量过拟合（可证伪）===")
    tiny = np.arange(512)
    set_seed(a.seed)
    f0, f1, a_tiny, _ = _fit(SingleTaskDNN(data, cfg, "is_click"), data, tiny,
                             y_true, "is_click", steps=300, lr=3e-3)
    check(f1 < f0 * 0.5, f"512 行 300 步：loss {f0:.4f} -> {f1:.4f}（降幅过半）")
    check(a_tiny > 0.95, f"过拟合后训练集 AUC {a_tiny:.4f} > 0.95 -> 结构与量纲没问题")

    # ---------- E. 五个模型真的独立 ----------
    print("\n=== E. 五个模型互不共享参数 ===")
    set_seed(a.seed)
    m1 = SingleTaskDNN(data, cfg, "is_click")
    set_seed(a.seed)
    m2 = SingleTaskDNN(data, cfg, "is_follow")
    ids1 = {id(p) for p in m1.parameters()}
    ids2 = {id(p) for p in m2.parameters()}
    check(not (ids1 & ids2), f"两个任务的模型没有任何共享张量（{len(ids1)} / {len(ids2)} 个）")
    check(torch.equal(m1.encoder.item_emb.weight, m2.encoder.item_emb.weight),
          "同 seed 下两个模型的 embedding 初值相同（初始化不是差异来源）")
    check(m1.encoder.hist_emb.num_embeddings != m1.encoder.item_emb.num_embeddings
          and m1.encoder.hist_emb.weight.data_ptr() != m1.encoder.item_emb.weight.data_ptr(),
          f"历史表 {m1.encoder.hist_emb.num_embeddings:,} 行与目标表 "
          f"{m1.encoder.item_emb.num_embeddings:,} 行是两张独立的表（P0 修复）")
    check(m1.encoder.item_emb.weight.data_ptr() != m2.encoder.item_emb.weight.data_ptr(),
          "但它们是两份独立内存 —— 训练时不会互相更新")

    # ---------- F. 数据顺序与模型无关 ----------
    print("\n=== F. 数据顺序只由 (seed, epoch) 决定 ===")
    o1 = epoch_order(1000, 42, 1)
    check(np.array_equal(o1, epoch_order(1000, 42, 1)), "同 (seed, epoch) 两次调用结果相同")
    check(not np.array_equal(o1, epoch_order(1000, 42, 2)), "不同 epoch 顺序不同")
    check(not np.array_equal(o1, epoch_order(1000, 43, 1)), "不同 seed 顺序不同")
    check(sorted(o1.tolist()) == list(range(1000)), "是一个完整置换（每行恰好用一次）")

    # ---------- G. 固定轮数规则 ----------
    print("\n=== G. 固定轮数、不挑 checkpoint ===")
    src = (ROOT / "src" / "ranking" / "trainer.py").read_text("utf-8")
    check('"checkpoint_rule": "fixed_epochs_last"' in src, "trainer 把规则写进结果")
    # 训练循环里不许出现按指标挑最好的动作
    banned = ["best_auc", "best_epoch", "if auc >", "argmax(", "early_stop", "patience"]
    hit = [w for w in banned if w in src]
    check(not hit, f"trainer 里没有任何早停 / 挑最佳轮的代码（命中 {hit}）")
    check(src.count("evaluate_split(model, va") == 1
          and "history.append" in src, "每轮仍评 valid 并记录（只是不据此选择）")

    # ---------- H. 确定性与输入指纹 ----------
    print("\n=== H. 确定性与输入指纹 ===")
    set_seed(a.seed)
    _, l1, _, _ = _fit(SingleTaskDNN(data, cfg, "is_click"), data, sub, y_true,
                       "is_click", steps=30)
    set_seed(a.seed)
    _, l2, _, _ = _fit(SingleTaskDNN(data, cfg, "is_click"), data, sub, y_true,
                       "is_click", steps=30)
    check(l1 == l2, f"同 seed 两次训练末 loss 逐位相同（{l1!r}）")
    fp = m1.encoder.fingerprint()
    check(fp["sha"] == m2.encoder.fingerprint()["sha"],
          f"两个任务的输入指纹相同：{fp['sha']}")
    check(fp["out_dim"] == sum(d for _, d in fp["channels"]) == m1.encoder.out_dim,
          f"指纹里的 out_dim {fp['out_dim']} == 通道维度之和")
    cfg2 = copy.deepcopy(cfg)
    cfg2["input"]["author_dim"] = int(cfg["input"]["author_dim"]) + 1
    set_seed(a.seed)
    check(SingleTaskDNN(data, cfg2, "is_click").encoder.fingerprint()["sha"] != fp["sha"],
          "改动 input 口径后指纹改变（可证伪：指纹真的在描述输入）")
    check(fp["data_params"] == {"max_hist": int(ic["max_hist"]),
                               "mask_oov_in_history": bool(ic["mask_oov_in_history"])},
          f"装载器口径进了指纹：{fp['data_params']}")
    # config 写了却没接线 -> 必须报错，不能静默沿用默认值
    cfg3 = copy.deepcopy(cfg)
    cfg3["input"]["max_hist"] = int(ic["max_hist"]) + 1
    try:
        SingleTaskDNN(data, cfg3, "is_click")
        check(False, "config 的 max_hist 与装载器不一致：未被拦下")
    except ValueError:
        check(True, "config 的 max_hist 与装载器不一致：已拦下（YAML 改了不生效会报错）")

    # ---------- H2. ID dropout 的行为 ----------
    print("\n=== H2. ID dropout（新代码路径）===")
    cfg_dp = copy.deepcopy(cfg)
    cfg_dp["input"]["id_dropout"] = 0.3
    set_seed(a.seed)
    m_dp = SingleTaskDNN(data, cfg_dp, "is_click")
    bb = data.batch(np.arange(2048))
    m_dp.eval()
    with torch.no_grad():
        check(torch.equal(m_dp(bb)["is_click"], m_dp(bb)["is_click"]),
              "eval 模式下两次前向一致（dropout 只能在训练时生效，否则评估不可复现）")
    m_dp.train()
    with torch.no_grad():
        check(not torch.equal(m_dp(bb)["is_click"], m_dp(bb)["is_click"]),
              "train 模式下两次前向不同（dropout 真的在采样，不是写了没生效）")
    # dropout 消耗全局 torch RNG —— 固定 seed 下轨迹必须仍然逐位可复现
    runs = []
    for sd in (a.seed, a.seed, a.seed + 1):
        set_seed(sd)
        _, last, _, _ = _fit(SingleTaskDNN(data, cfg_dp, "is_click"), data, sub, y_true,
                             "is_click", steps=25)
        runs.append(last)
    check(runs[0] == runs[1] and runs[0] != runs[2],
          f"同 seed 两次训练末 loss 逐位相同、换 seed 不同（{runs[0]!r}）")
    # dropout 置 0 的同时必须把 video_id_known 也置 0，否则会教出「known=1 配 OOV 向量」
    set_seed(a.seed)
    enc_dp = SingleTaskDNN(data, cfg_dp, "is_click").encoder
    enc_dp.train()
    flag_col = enc_dp.channels[-1][1]
    with torch.no_grad():
        xs = [enc_dp(bb)[:, -flag_col:] for _ in range(6)]
    ki = enc_dp.flag_keys.index("video_id_known")
    known = bb["video_id_known"].numpy()
    zeroed = np.array([(x[:, ki].numpy() == 0) & (known == 1) for x in xs]).any(0)
    check(zeroed.sum() > 0,
          f"被 dropout 掉的行其 video_id_known 也被置 0（{int(zeroed.sum())} 行命中）")
    # item_id 通道消融：out_dim 与指纹都要变
    cfg_ab = copy.deepcopy(cfg)
    cfg_ab["input"]["item_id_channel"] = False
    set_seed(a.seed)
    enc_ab = SingleTaskDNN(data, cfg_ab, "is_click").encoder
    check(enc_ab.out_dim == m1.encoder.out_dim - int(cfg["embedding_dim"]) - 1
          and "video_id_known" not in enc_ab.flag_keys,
          f"关掉 item_id 通道后 out_dim {m1.encoder.out_dim} -> {enc_ab.out_dim}"
          "，且 video_id_known 一并移除（没有 ID 向量时它只剩热度漂移）")

    # ---------- I. 四个模型的训练协议逐字相同 ----------
    print("\n=== I. 四个模型的训练协议一致 ===")
    # §27.1 的 ΔAUC 要求除共享机制外无差异。指纹能在**运行后**发现输入口径分叉，
    # 但训练协议（轮数 / batch / lr / 正则）不在指纹里，而它同样会污染 ΔAUC。
    # 这里在跑之前就比对四份 config —— 不一致直接报错，不等跑完看表。
    PROTOCOL_KEYS = ("epochs", "batch_size", "lr", "optimizer", "weight_decay",
                     "dropout", "seeds", "tasks")
    others = ["configs/mmoe.yaml", "configs/ple.yaml", "configs/selective.yaml"]
    base = {k: cfg.get(k) for k in PROTOCOL_KEYS}
    base_in = dict(require(cfg, "input"))
    for f in others:
        o = load_config(f)
        diff = {k: (base[k], o.get(k)) for k in PROTOCOL_KEYS if base[k] != o.get(k)}
        check(not diff, f"{Path(f).name} 的训练协议与 single_task 相同（差异 {diff}）")
        din = {k: (base_in.get(k), dict(require(o, "input")).get(k))
               for k in set(base_in) | set(require(o, "input"))
               if base_in.get(k) != dict(require(o, "input")).get(k)}
        check(not din, f"{Path(f).name} 的 input 口径与 single_task 相同（差异 {din}）")

    # ---------- J. 已落盘结果之间的一致性 ----------
    print("\n=== J. 已落盘结果的口径一致 ===")
    # config 一致（I 段）保证的是"打算用同一套"，这里查的是"实际跑出来的确实是同一套"：
    # 中途改过 config 再跑、或用旧 checkpoint 补表，都会在这里现形。
    import json as _json
    MODELS = {"single_task", "mmoe", "ple", "selective_sharing"}
    seen = []
    for f in sorted((ROOT / "results").glob("*.json")):
        try:
            r = _json.loads(f.read_text("utf-8"))
        except (ValueError, OSError):
            continue
        if not isinstance(r, dict) or r.get("model") not in MODELS or r.get("smoke"):
            continue
        seen.append((f.name, r))
    if not seen:
        check(True, "results/ 下暂无正式排序结果（跑完后这条会开始比对）")
    else:
        shas = {n: r.get("input_fingerprint", {}).get("sha") for n, r in seen}
        check(len(set(shas.values())) == 1,
              f"{len(seen)} 份结果的输入指纹一致：{sorted(set(shas.values()))}"
              + ("" if len(set(shas.values())) == 1 else f" -> 逐份 {shas}"))
        prot = {n: tuple(r.get(k) for k in ("epochs", "batch_size", "lr",
                                            "checkpoint_rule")) for n, r in seen}
        check(len(set(prot.values())) == 1,
              f"{len(seen)} 份结果的训练协议一致：{sorted(set(prot.values()))[0]}"
              + ("" if len(set(prot.values())) == 1 else f" -> 逐份 {prot}"))
        check(all(r.get("checkpoint_rule") == "fixed_epochs_last" for _, r in seen),
              "每份结果都记录了 fixed_epochs_last（没有谁偷偷挑了最佳轮）")
        for n, r in seen:
            if r.get("eval_test"):
                check(False, f"{n} 记录了 test 指标 —— 架构冻结前不该有")

    print(f"\n共校验 {checks} 项，失败 {len(failures)} 项。")
    if failures:
        print("Single-Task 验算未通过：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("验算通过（任务映射 / 无标签泄漏 / 标签置换 / 学习能力 / 模型独立 / "
          "顺序无关 / 固定轮数 / 确定性）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
