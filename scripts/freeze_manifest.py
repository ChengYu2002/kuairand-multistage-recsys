"""冻结清单：记录既有产物的哈希，之后断言它们一个字节都没变。

排序侧要新建自己的元数据空间（item / author / tag），而召回的词表、静态表、
checkpoint 必须原样不动 —— 一旦 author 词表的「第 N 行是谁」变了，已经训好的双塔
checkpoint 就静默作废：不报错，只是指标莫名变差，而你会去怀疑模型。

因此在新增任何文件**之前**先记一份哈希，之后每次验算都比对。新增文件是允许的
（排序表就是新增），被修改或删除的旧文件一律报错。

用法：
    python scripts/freeze_manifest.py --record      # 建基线（只允许跑一次）
    python scripts/freeze_manifest.py --check       # 断言旧文件未变
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "processed" / "retrieval_freeze_manifest.json"
# 清单自身与纯缓存不纳入
SKIP_NAMES = {MANIFEST.name, ".DS_Store", ".gitkeep"}
WATCH = ["data/processed", "experiments", "results"]


def sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def scan() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for d in WATCH:
        base = ROOT / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.name in SKIP_NAMES or "__pycache__" in p.parts:
                continue
            out[str(p.relative_to(ROOT))] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的基线（谨慎）")
    a = ap.parse_args()
    if a.record == a.check:
        ap.error("必须且只能指定 --record 或 --check")

    if a.record:
        if MANIFEST.exists() and not a.force:
            print(f"基线已存在：{MANIFEST.relative_to(ROOT)}。"
                  "重新记录会把「当前状态」当成基线，从而掩盖已发生的改动；"
                  "确认要这么做请加 --force。")
            return 1
        files = scan()
        MANIFEST.write_text(json.dumps(
            {"n_files": len(files), "files": files}, indent=2, sort_keys=True), encoding="utf-8")
        total = sum(v["bytes"] for v in files.values())
        print(f"已记录 {len(files)} 个文件（{total / 2**30:.2f} GB）-> "
              f"{MANIFEST.relative_to(ROOT)}")
        return 0

    if not MANIFEST.exists():
        print(f"找不到基线 {MANIFEST.relative_to(ROOT)}，先跑 --record")
        return 1
    base = json.loads(MANIFEST.read_text("utf-8"))["files"]
    now = scan()
    changed = [k for k in base if k in now and now[k]["sha256"] != base[k]["sha256"]]
    missing = [k for k in base if k not in now]
    added = sorted(set(now) - set(base))
    for k in changed:
        print(f"  FAIL 被修改  {k}")
    for k in missing:
        print(f"  FAIL 已删除  {k}")
    for k in added:
        print(f"  新增      {k}")
    ok = not changed and not missing
    print(f"\n基线 {len(base)} 个文件：修改 {len(changed)}，删除 {len(missing)}，"
          f"新增 {len(added)}。{'召回侧产物未变。' if ok else '有旧产物被改动！'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
