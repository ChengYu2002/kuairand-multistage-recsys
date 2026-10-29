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
    "exposure": ExposureNegative,
    "hybrid": HybridNegative,
}
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
