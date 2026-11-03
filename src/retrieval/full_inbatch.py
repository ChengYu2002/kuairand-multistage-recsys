"""完整 in-batch softmax，支持随请求日期变化的 ID+Side 物品塔。

普通完整 in-batch loss 会直接计算 ``u @ v_pos.T``：batch 中每个 target 既是
自己这一行的正样本，也是其他行的负样本。这个写法只对 query-independent 的物品
向量成立。本项目的 ID+Side 物品塔含 ``video_age = request_date - upload_date``，
所以视频 j 给请求 i 当候选时，必须按请求 i 的日期编码，不能复用视频 j 在自己请求
日期下的 ``v_pos[j]``。

这里按 batch 内的请求日期分组。对每个日期 d，把 batch 的全部 B 个 target 在 d 下
编码一次，再只给日期为 d 的用户计算分数。训练段有 16 天，因此最坏编码 16*B 行，
而不是显式构造 B*B 份物品特征；ID-only 则仍然只编码 B 行。

完整矩阵的第 i 列是第 i 条样本的 target。交叉熵标签因此是对角线 ``0..B-1``。
同一视频可能在 batch 中出现多次：这些非对角位置与当前正样本同 ID，必须作为
accidental hit 屏蔽；对角线本身必须保留。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def date_aware_inbatch_logits(
    model,
    data,
    batch: dict,
    request_dates: np.ndarray,
) -> torch.Tensor:
    """返回 B x B 的未做 logQ 校正、未屏蔽 accidental hit 的 logits。

    行 i 对应用户/请求 i，列 j 对应 batch target j。ID+Side 下列 j 的物品向量
    按行 i 的请求日期计算；ID-only 下日期不参与物品塔，只做一次矩阵乘法。
    """

    target_ids = batch["target"]
    if target_ids.ndim != 1 or len(target_ids) < 2:
        raise ValueError(
            f"完整 in-batch 至少需要 2 个一维 target，收到 {tuple(target_ids.shape)}"
        )
    dates = np.asarray(request_dates)
    if dates.shape != (len(target_ids),):
        raise ValueError(f"request_dates 应为 ({len(target_ids)},)，实际 {dates.shape}")

    u = model.user_vec(batch)
    device = u.device
    item_ids = target_ids.detach().cpu().numpy()
    all_rows = np.arange(len(target_ids), dtype=np.int64)

    if model.item_tower.config == "id_side":
        days = np.unique(dates)
        groups = [
            (int(day), np.flatnonzero(dates == day).astype(np.int64, copy=False))
            for day in days
        ]
        # 一次性编码 D*B 行，避免每个训练 step 调用物品塔 D 次。reshape 后第 g
        # 块就是「batch 全部 target 在第 g 个请求日下」的向量。
        expanded_ids = np.tile(item_ids, len(days))
        expanded_dates = np.repeat(days, len(item_ids)).astype(dates.dtype, copy=False)
        item_batch = data.item_features(expanded_ids, expanded_dates)
    else:
        groups = [(None, all_rows)]
        item_batch = data.item_features(item_ids, None)

    if device.type != "cpu":
        item_batch = {k: v.to(device) for k, v in item_batch.items()}
    vectors = model.item_vec(item_batch).view(len(groups), len(item_ids), -1)

    score_parts: list[torch.Tensor] = []
    row_parts: list[torch.Tensor] = []
    for group_idx, (_, row_np) in enumerate(groups):
        v = vectors[group_idx]
        row_idx = torch.from_numpy(row_np).to(device=device)
        score_parts.append(
            u.index_select(0, row_idx) @ v.transpose(0, 1) / model.temperature
        )
        row_parts.append(row_idx)

    grouped_rows = torch.cat(row_parts)
    grouped_scores = torch.cat(score_parts, dim=0)
    # np.unique 按日期排序，拼起来后的行序通常不再是原 batch 顺序；恢复后才能让
    # 第 i 行的正样本仍位于第 i 列。
    restore = torch.argsort(grouped_rows)
    logits = grouped_scores.index_select(0, restore)
    if logits.shape != (len(target_ids), len(target_ids)):
        raise AssertionError(
            f"完整 in-batch logits 应为 BxB，实际 {tuple(logits.shape)}"
        )
    return logits


def off_diagonal_duplicate_mask(target_ids: torch.Tensor) -> torch.Tensor:
    """同 ID 的非对角位置为 True；对角线正样本始终保留。"""

    if target_ids.ndim != 1:
        raise ValueError(f"target_ids 必须是一维，收到 {tuple(target_ids.shape)}")
    same = target_ids.unsqueeze(1) == target_ids.unsqueeze(0)
    eye = torch.eye(len(target_ids), dtype=torch.bool, device=target_ids.device)
    return same & ~eye


def full_inbatch_loss(
    model,
    data,
    batch: dict,
    request_dates: np.ndarray,
    log_q: torch.Tensor | None,
) -> torch.Tensor:
    """完整 in-batch 交叉熵；log_q 必须覆盖 batch 中每个 target。

    ``log_q`` 为 None 时可用于独立验算未校正公式；正式的
    ``inbatch_full_logq`` 训练路径始终传入全局训练正样本频率的 log Q。
    """

    target_ids = batch["target"]
    logits = date_aware_inbatch_logits(model, data, batch, request_dates)
    if log_q is not None:
        if log_q.shape != target_ids.shape:
            raise ValueError(
                f"log_q 应与 target 同形，实际 {tuple(log_q.shape)} vs "
                f"{tuple(target_ids.shape)}"
            )
        logits = logits - log_q.to(logits.device).unsqueeze(0)

    logits = logits.masked_fill(
        off_diagonal_duplicate_mask(target_ids).to(logits.device), float("-inf")
    )
    labels = torch.arange(len(target_ids), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)
