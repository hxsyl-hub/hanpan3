# -*- coding: utf-8 -*-
"""把划分清单物化成标准数据集目录，方便打包上传。

平时训练只需要两样东西：data\raw\右焊盘（图）+ splits\*.txt（清单），
脚本按清单取图，一张图都不复制 —— 这是"始终跑同一份数据"的保证。

但要打包上传到云端时，一个已经按 train/val/test 分好类的目录更省事：
云端解压就能直接用 ImageFolder 读，也不依赖清单里的相对路径。
所以这个脚本负责把清单"物化"成目录结构，raw 里的原始图始终不动。

产出：
    data\dataset\
      train\<工业类名>\*.bmp
      val\<工业类名>\*.bmp
      test\<工业类名>\*.bmp

用法：
    python make_dataset_dir.py                     # 从 raw 复制
    python make_dataset_dir.py --mode hardlink     # 硬链接，省空间、速度快
    python make_dataset_dir.py --out data\pack     # 换输出目录
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
SPLITS = ROOT / "splits"


def read_list(path):
    """读取清单：每行格式为 "相对路径\\t标签"。"""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        rel = parts[0].strip()
        label = int(parts[1]) if len(parts) > 1 else -1
        out.append((rel, label))
    return out


def place(src: Path, dst: Path, mode: str) -> str:
    """按指定方式把图放到目标位置，返回 ok / skip。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "skip"
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)        # 跨盘或文件系统不支持时退回复制
        except Exception:
            shutil.copy2(src, dst)
    return "ok"


def main():
    p = argparse.ArgumentParser(description="把划分清单物化成数据集目录")
    p.add_argument("--data-dir", default=str(ROOT / "data" / "raw" / "右焊盘"),
                   help="原始图根目录")
    p.add_argument("--out", default=str(ROOT / "data" / "dataset"),
                   help="输出目录，默认 data/dataset")
    p.add_argument("--mode", default="copy",
                   choices=["copy", "symlink", "hardlink"],
                   help="copy=复制(最稳); hardlink=硬链接(省空间,需同盘); "
                        "symlink=软链接(最省,个别环境不支持)")
    p.add_argument("--splits", default="train,val,test",
                   help="只物化指定集合，逗号分隔。数据量大时可分多次跑，"
                        "例：--splits train")
    args = p.parse_args()

    data = Path(args.data_dir)
    out = Path(args.out)
    if not data.is_dir():
        print("原始图目录不存在:", data)
        return

    classes = json.loads((SPLITS / "classes.json").read_text(encoding="utf-8"))
    idx2name = {i: c for c, i in classes.items()}

    print("原始图目录:", data)
    print("输出目录:  ", out)
    print("模式:      ", args.mode)
    print()

    total, bad = 0, 0
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        lst = read_list(SPLITS / f"{split}.txt")
        cnt, missing = Counter(), 0
        for rel, label in lst:
            src = data / rel
            if not src.is_file():
                missing += 1
                continue
            cls_name = idx2name.get(label, Path(rel).parts[0])
            place(src, out / split / cls_name / Path(rel).name, args.mode)
            cnt[cls_name] += 1
        n = sum(cnt.values())
        total += n
        bad += missing
        print("%-6s %6d 张   (缺失 %d)" % (split, n, missing))
        for c in sorted(cnt):
            print("    %-24s %5d" % (c, cnt[c]))
        print()

    size_mb = sum(f.stat().st_size for f in out.rglob("*")
                  if f.is_file()) / (1024 * 1024)
    print("合计 %d 张 / %.1f MB（缺失 %d 张）" % (total, size_mb, bad))
    print("目录结构: %s/{train,val,test}/<类别>/*.bmp" % out)
