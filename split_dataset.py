"""Stratified 7:2:1 train/val/test split. Pure stdlib, run locally.

hanpan3 适配说明（与原项目不同之处）：
  1. 原始图在 data\\raw\\右焊盘\\<中文类别>\\，不是 data\\<类别>\\
  2. 中文类别名按 scripts\\类别映射.json 重映射为工业标准名（CX -> NG_PadDiscoloration 等），
     输出的 classes.json 与 splits/*.txt 里用的都是工业名
  3. 不做类别均衡、不做增强 —— 划分结果保持原始真实分布
"""
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
DATA_DIR = Path(__file__).parent / "data" / "raw" / "右焊盘"
OUT_DIR = Path(__file__).parent / "splits"
RATIOS = (0.7, 0.2, 0.1)
IMG_EXTS = {".bmp", ".jpg", ".jpeg", ".png"}
MAPPING_JSON = Path(__file__).parent / "scripts" / "类别映射.json"


def load_class_map():
    """从 scripts/类别映射.json 反查：中文原类别名 -> 工业标准类别名"""
    if not MAPPING_JSON.is_file():
        print(f"[警告] 未找到 {MAPPING_JSON}，将直接使用原始目录名作为类别名")
        return {}
    data = json.loads(MAPPING_JSON.read_text(encoding="utf-8"))
    rev = {}
    for new_cls, info in data.get("映射", {}).items():
        for src in info.get("来源", []):
            rev[src] = new_cls
    return rev


def main():
    rev = load_class_map()
    raw_classes = sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())
    classes = sorted(set(rev.get(c, c) for c in raw_classes))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    if rev:
        print("类别映射（中文原名 -> 工业标准名）：")
        for c in raw_classes:
            print(f"    {c:6s} -> {rev.get(c, c)}")
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "classes.json").write_text(
        json.dumps(class_to_idx, ensure_ascii=False, indent=2), encoding="utf-8")

    rng = random.Random(SEED)
    by_split = {"train": [], "val": [], "test": []}

    for cls in raw_classes:
        new_cls = rev.get(cls, cls)          # 中文原名 -> 工业标准名
        idx = class_to_idx[new_cls]
        files = sorted(p for p in (DATA_DIR / cls).iterdir()
                       if p.suffix.lower() in IMG_EXTS)
        rng.shuffle(files)
        n = len(files)
        n_train = round(n * RATIOS[0])
        n_val = round(n * RATIOS[1])
        n_test = n - n_train - n_val
        splits = [("train", files[:n_train]),
                  ("val", files[n_train:n_train + n_val]),
                  ("test", files[n_train + n_val:])]
        print(f"{cls:6s}->{new_cls:22s} total={n:5d}  train={n_train} val={n_val} test={n_test}")
        for split, items in splits:
            for p in items:
                by_split[split].append((p.relative_to(DATA_DIR).as_posix(), idx))

    for split, rows in by_split.items():
        rows.sort()
        with open(OUT_DIR / f"{split}.txt", "w", encoding="utf-8") as f:
            for rel, idx in rows:
                f.write(f"{rel}\t{idx}\n")
        print(f"{split}: {len(rows)} images -> splits/{split}.txt")

    total = sum(len(v) for v in by_split.values())
    print(f"合计 {total} 张 -> splits/")
    print("说明：本次划分未做类别均衡、未做数据增强，各类保持真实分布。")
    print("Done.")


if __name__ == "__main__":
    main()
