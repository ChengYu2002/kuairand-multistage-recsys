"""环境自检：Python 版本/架构、依赖、加速后端、数据落位情况。

    python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WEEK1 = ["numpy", "pandas", "polars", "pyarrow", "duckdb", "yaml", "sklearn", "matplotlib", "tqdm"]
WEEK2 = ["torch"]
OPTIONAL = ["lightgbm", "faiss"]  # P1


def _probe(names: list[str]) -> list[str]:
    missing = []
    for name in names:
        try:
            mod = importlib.import_module(name)
            print(f"  ok    {name:12s} {getattr(mod, '__version__', '?')}")
        except ImportError:
            print(f"  --    {name:12s} 未安装")
            missing.append(name)
    return missing


def main() -> int:
    print(f"Python {sys.version.split()[0]}  {platform.machine()}  ({sys.prefix})")
    if sys.version_info[:2] != (3, 11):
        print("  warn  预期 3.11；torch/lightgbm/faiss 在更新的版本上可能没有轮子")
    if platform.machine() != "arm64" and platform.system() == "Darwin":
        print("  warn  不是 arm64，可能在 Rosetta 下运行，性能会明显下降")

    print("\n[Week 1 依赖]")
    missing = _probe(WEEK1)

    print("\n[Week 2 依赖]")
    _probe(WEEK2)

    print("\n[P1 可选]")
    _probe(OPTIONAL)

    print("\n[加速后端]")
    try:
        import torch

        if torch.backends.mps.is_available():
            print("  ok    MPS 可用")
        else:
            print("  --    MPS 不可用，将回退到 CPU")
    except ImportError:
        print("  --    torch 未安装，Week 2 开始前再装")

    print("\n[数据]")
    raw = ROOT / "data" / "raw"
    files = sorted(p for p in raw.rglob("*") if p.is_file() and p.name != ".gitkeep")
    if not files:
        print(f"  --    {raw.relative_to(ROOT)}/ 为空，请先下载 KuaiRand-1K")
    else:
        total = sum(p.stat().st_size for p in files)
        print(f"  ok    {len(files)} 个文件，合计 {total / 1024**3:.2f} GB")
        for p in files[:10]:
            print(f"        {p.relative_to(raw)}  ({p.stat().st_size / 1024**2:.1f} MB)")
        if len(files) > 10:
            print(f"        ... 另有 {len(files) - 10} 个")

    if missing:
        print(f"\n缺少 Week 1 依赖: {' '.join(missing)}")
        return 1
    print("\n环境就绪。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
