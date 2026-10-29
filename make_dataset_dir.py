# -*- coding: utf-8 -*-
"""把划分清单物化成打包用的数据集目录。

平时训练只需要两样东西：data\\raw\\右焊盘（图）+ splits\\*.txt（清单），
脚本按清单取图，一张图都不复制 —— 这是"始终跑同一份数据"的保证。

但打包上传到云端时，自带的 raw 里有 1546 张被裁掉的 OK 冗余图，白占体积。
这个脚本只把清单里真正用到的图复制出来（train+val+test 的并集），
目录结构保持和 raw 一致（<类别名>/*.bmp），所以
    python train.py --data-dir data\\dataset ...
可以直接跑，splits 里的相对路径不用改。

产出：
    data\\dataset\\
      CX\\*.bmp      （29 张：train 20 + val 6 + test 3）
      HDZ\\*.bmp
      OK\\*.bmp      （2806 张：1500 + 870 + 436）
      PW\\*.bmp
      XH1\\*.bmp
      XH2\\*.bmp
      XH3\\*.bmp

用法：
    python make_dataset_dir.py                      # 默认输出到 data\\dataset
    python make_dataset_dir.py --out data\\pack      # 换输出目录
    python make_dataset_dir.py --mode hardlink      # 硬链接，省空间，需同盘
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
SPLITS = ROOT / "splits"


def read_list(path: Path):
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
    """把图放到目标位置，返回 ok / skip。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "skip"
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)          # 跨盘或不支持时退回复制
        except Exception:
            shutil.copy2(src, dst)
    return "ok"


def main():
    p = argparse.ArgumentParser(description="把划分清单物化成数据集目录")
    p.add_argument("--data-dir", default=str(ROOT / "data" / "raw" / "右焊盘"),
                   help="原始图根目录")
    p.add_argument("--out", default=str(ROOT / "data" / "dataset"),
                   help="输出目录，默认 data/dataset（结构与 raw 一致）")
    p.add_argument("--mode", default="copy",
                   choices=["copy", "symlink", "hardlink"],
                   help="copy=复制(最稳); hardlink=硬链接(省空间,需同盘); "
                        "symlink=软链接(最省,个别环境不支持)")
    p.add_argument("--splits", default="train,val,test",
                   help="只物化指定集合，逗号分隔（数据量大时可分多次跑），"
                        "例：--splits train")
    p.add_argument("--layout", default="flat", choices=["flat", "split"],
                   help="flat = out/<类别>/*.bmp，配合 splits/*.txt 清单使用；"
                        "split = out/{train,val,test}/<类别>/*.bmp，"
                        "物理分好目录，训练时加 --layout split")
    args = p.parse_args()

    data = Path(args.data_dir)
    out = Path(args.out)
    if not data.is_dir():
        print("原始图目录不存在:", data)
        return

    classes = json.loads((SPLITS / "classes.json").read_text(encoding="utf-8"))
    idx2name = {i: c for c, i in classes.items()}

    print("原始图目录:", data)
    layout_desc = ("<类别名>/*.bmp（配 splits 清单）" if args.layout == "flat"
                   else "{train,val,test}/<类别名>/*.bmp（物理分目录）")
    print("输出目录:  ", out, "（", layout_desc, "）")
    print("模式:      ", args.mode)
    print()

    total, missing_all = 0, 0
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        lst = read_list(SPLITS / f"{split}.txt")
        cnt, missing = Counter(), 0
        for rel, label in lst:
            src = data / rel
            if not src.is_file():
                missing += 1
                continue
            cls_name = idx2name.get(label, Path(rel).parts[0])
            # flat  : out/<类别>/<文件>          配 splits 清单
            # split : out/<split>/<类别>/<文件>  物理分目录，可直接扫描
            dst_dir = ((out / split / cls_name) if args.layout == "split"
                       else (out / cls_name))
            place(src, dst_dir / Path(rel).name, args.mode)
            cnt[cls_name] += 1
        n = sum(cnt.values())
        total += n
        missing_all += missing
        print("%-6s %6d 张   (缺失 %d)" % (split, n, missing))
        print()

    size_mb = sum(f.stat().st_size for f in out.rglob("*")
                  if f.is_file()) / (1024 * 1024)
    print("合计 %d 张 / %.1f MB（缺失 %d 张）" % (total, size_mb, missing_all))
    cmd = "python train.py --data-dir %s" % out
    if args.layout == "split":
        cmd += " --layout split"
    print("用法:", cmd)
