"""四种负采样策略的统一入口（plan §16 / §16.6）。

存在的理由只有一个：**让 two_tower.py 的改动最小**。

exposure / hybrid 需要一张"每个用户曝光过什么"的表，而 random / in-batch 只需要
候选库区间。如果把建池子的逻辑写进 two_tower.py，训练脚本就要多出 polars 导入、
路径拼接、配置读取 —— 而 §16.6 要求四种策略除采样外没有任何差异，训练脚本本身
就是那个"不得有差异"的地方，改得越少越好。

    sampler = build_sampler(strategy, lo, hi, data=data, cfg=cfg, proc=proc, protocol=p)

random / in-batch 走的构造路径与改动前**逐字相同**（`SAMPLERS[strategy](lo, hi)`），
所以同 seed 下它们的结果必须逐位不变 —— verify 脚本对此有断言。
"""

from __future__ import annotations

from pathlib import Path

from src.retrieval.exposure_negative import ExposureNegative, build_exposure_pool
from src.retrieval.hybrid_negative import HybridNegative
from src.retrieval.inbatch_negative import InBatchNegative
from src.retrieval.random_negative import RandomNegative
from src.utils.logger import get_logger

log = get_logger(__name__)

SAMPLERS = {
    "random": RandomNegative,
    "inbatch": InBatchNegative,
    # §16.5 的第五个实验臂：**同一个 in-batch 采样器**，只是 loss 里多了 logQ 校正。
    # 必须是独立的 strategy 名而不是 inbatch 的一个开关 —— 产物名由 strategy 拼成，
    # 共用名字会让 inbatch_logq 的 checkpoint 覆盖掉已跑完的 inbatch。
    "inbatch_logq": InBatchNegative,
    # 独立的负样本规模实验：不再抽固定 n_neg 个位置，而是让 batch 中其余
    # B-1 个 target 全部参与 softmax。训练入口由 full_inbatch.py 接管；这里仍
    # 注册同一个类，使 strategy 的 CLI、产物命名与统一构造入口保持一致。
    "inbatch_full_logq": InBatchNegative,
    "exposure": ExposureNegative,
    "hybrid": HybridNegative,
}
# 这些策略在 loss 里做 logQ 校正（Q 的口径见 src/retrieval/logq_correction.py）
USES_LOGQ = ("inbatch_logq", "inbatch_full_logq")
FULL_INBATCH = ("inbatch_full_logq",)
# 需要曝光池的策略。其余策略的构造路径必须保持与改动前完全一致。
NEEDS_POOL = ("exposure", "hybrid")


def build_sampler(
    strategy: str,
    catalog_lo: int,
    catalog_hi: int,
    *,
    data=None,
    cfg: dict | None = None,
    proc: Path | None = None,
    protocol: str = "main",
):
    if strategy not in SAMPLERS:
        raise NotImplementedError(
            f"负采样策略 {strategy!r} 不存在（当前有 {sorted(SAMPLERS)}）"
        )
    if strategy not in NEEDS_POOL:
        return SAMPLERS[strategy](catalog_lo, catalog_hi)

    if data is None or proc is None:
        raise ValueError(f"{strategy} 需要 data 与 proc 才能构造曝光池")
    # 用户数取自静态特征表的行数 —— 池子按 user_id 直接下标索引，两者必须同一个口径。
    n_users = int(data.user_static_num.shape[0])
    # 池子**只从 train 段构造**。这不是性能取舍：读 valid/test 的曝光就是直接的泄露。
    pool = build_exposure_pool(
        Path(proc) / f"samples_train_{protocol}.parquet",
        n_users,
        catalog_lo,
        catalog_hi,
    )
    cov = pool.coverage()
    log.info(
        "曝光负样本池：%s 个 (user,item) 对，覆盖 %.1f%% 的用户"
        "（空池子 %d 个；池子大小 min/p50/max = %d/%d/%d）",
        f"{cov['pool_pairs']:,}",
        100 * cov["users_nonempty_ratio"],
        cov["users_empty"],
        cov["pool_size_min"],
        cov["pool_size_p50"],
        cov["pool_size_max"],
    )

    if strategy == "exposure":
        return ExposureNegative(catalog_lo, catalog_hi, pool)
    mix = (cfg or {}).get("negative_sampling", {}).get("hybrid_mix")
    return HybridNegative(catalog_lo, catalog_hi, pool, mix)
