# -*- coding: utf-8 -*-
"""离线数据增强：只给训练集里的小类补图。

文档依据（hanpan/docs/重要说明.docx）：
    "再次尝试离线数据增强"
    "针对类别少的数据先增强（比如原来训练只有 30 张，增强至 50%-80%）"
    "逐步进行如区域直方图，旋转，gamma 光线调整，或者创造性增强。"

关于"增强至 50%-80%"怎么落地：
    按"在原数量上增加 50%~80%"处理（--ratio 0.5 起，可设 0.8）。
    另一种读法是"补到最大类的 50%~80%"，但那对只有 20 张的
    NG_PadDiscoloration 意味着派生到 750~1200 张（39 倍以上），
    一张原图要变出几十个副本，过拟合风险太大，所以不采用。

三条保证实验可比的原则：
    1. 结果只写入 splits/train_aug.txt，基线的 splits/train.txt 一个字不改；
    2. val.txt / test.txt 完全不碰；
    3. 增强图文件名带 _aug 标记，原始图不覆盖、不改名、可随时删除。
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps

ROOT = Path(__file__).parent


def load_splits(path: Path):
    """读清单文件，每行格式：相对路径\t标签"""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rel, _, idx = line.rpartition("\t")
        rows.append((rel, int(idx)))
    return rows


# ---------------- 增强算子：都在 PIL 图像上做 ----------------

def op_hist(img, rng):
    """直方图均衡。文档说最大的困难是"曝光过度、有一些图偏暗"，
    均衡能拉开对比度，对这类图通常有效。"""
    return ImageOps.equalize(img)


def op_rotate(img, rng):
    """角度旋转：小角度模拟摆放偏差；三成概率用 90 的整数倍（无插值损失）"""
    deg = rng.choice([-90, 90, 180]) if rng.random() < 0.3 else rng.uniform(-15, 15)
    return img.rotate(deg, resample=Image.BICUBIC, fillcolor=0)


def op_gamma(img, rng):
    """gamma 光线调整：模拟整体偏亮或偏暗"""
    g = rng.uniform(0.6, 1.7)
    lut = [min(255, max(0, int(255 * ((i / 255.0) ** (1.0 / g)))))
           for i in range(256)]
    return img.point(lut * len(img.getbands()))


def op_contrast(img, rng):
    """对比度 + 锐化（文档所说"创造性增强"里最稳妥的一种）"""
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(1.1, 1.6))
    return ImageEnhance.Sharpness(img).enhance(rng.uniform(1.0, 1.8))


OPS = [op_hist, op_rotate, op_gamma, op_contrast]


def make_variant(src: Path, dst: Path, k: int, rng: random.Random):
    """从一张原图派生第 k 个变体：算子轮换，同一张原图派生的副本互不重样"""
    img = Image.open(src).convert("RGB")
    img = OPS[k % len(OPS)](img, rng)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(ROOT / "data" / "raw" / "右焊盘"),
                   help="原始图根目录，与 train.py 的 --data-dir 必须一致")
    p.add_argument("--train-list", default=str(ROOT / "splits" / "train.txt"),
                   help="基线训练集清单（只读，不会被修改）")
    p.add_argument("--out-list", default=str(ROOT / "splits" / "train_aug.txt"),
                   help="增强后的新清单（基线清单保持不变）")
    p.add_argument("--ratio", type=float, default=0.5,
                   help="小类增加多少：0.5=增加 50%（默认），0.8=增加 80%")
    p.add_argument("--small-ratio", type=float, default=0.2,
                   help="低于最大类的这个比例才算小类（默认 0.2）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="只算数量，不写盘")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    rows = load_splits(Path(args.train_list))
    classes = json.loads(
        (ROOT / "splits" / "classes.json").read_text(encoding="utf-8"))
    idx2cls = {v: k for k, v in classes.items()}

    cnt = Counter(idx for _, idx in rows)
    biggest = max(cnt.values())
    small = sorted(i for i, n in cnt.items() if n < biggest * args.small_ratio)

    print(f"基线训练集 {len(rows)} 张，最大类 {biggest} 张")
    print(f"小类判定线：少于 {biggest * args.small_ratio:.0f} 张")
    print(f"被增强的类别（{len(small)} 个）：")
    for i in small:
        print(f"    {idx2cls[i]:24s} {cnt[i]:5d} 张")

    rng = random.Random(args.seed)
    by_cls = defaultdict(list)
    for rel, idx in rows:
        by_cls[idx].append(rel)

    plan = []
    for i in small:
        need = round(cnt[i] * args.ratio)
        srcs = sorted(by_cls[i])
        for k in range(need):
            src_rel = srcs[k % len(srcs)]          # 循环取原图，保证每张都用到
            src = data_dir / src_rel
            sp = Path(src_rel)
            op_name = OPS[k % len(OPS)].__name__[3:]      # op_hist -> hist
            round_no = k // len(srcs) + 1
            name = f"{sp.stem}_aug{round_no}_{op_name}{sp.suffix}"
            dst_rel = (sp.parent / name).as_posix()
            plan.append((i, src, data_dir / dst_rel, dst_rel))

    print(f"\n计划生成 {len(plan)} 张增强图（每类增加 {args.ratio:.0%}）")
    if args.dry_run:
        print("[dry-run] 未写入任何文件，去掉 --dry-run 才真正生成")
        return

    new_rows = list(rows)
    done = 0
    for k, (i, src, dst, dst_rel) in enumerate(plan):
        if not src.is_file():
            print(f"  ⚠ 原图不存在，跳过：{src}")
            continue
        make_variant(src, dst, k, rng)
        new_rows.append((dst_rel, i))
        done += 1
        if (done % 100) == 0:
            print(f"    已生成 {done}/{len(plan)} ...")

    rng.shuffle(new_rows)
    out = Path(args.out_list)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for rel, idx in new_rows:
            f.write(f"{rel}\t{idx}\n")
    print(f"\n新清单 {out}：{len(new_rows)} 行"
          f"（基线 {len(rows)} + 新增 {done}）")

    stat = ROOT / "splits" / "离线增强统计.csv"
    after = Counter(idx for _, idx in new_rows)
    with open(stat, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["类别", "原训练集", "新增", "增强后", "增幅"])
        for i in sorted(cnt, key=lambda x: -cnt[x]):
            w.writerow([idx2cls[i], cnt[i], after[i] - cnt[i], after[i],
                        f"{(after[i] - cnt[i]) / cnt[i]:.0%}"])
    print(f"统计表 {stat}")

    print("\n说明：val.txt / test.txt 未改动，基线 train.txt 也未改动，")
    print("      因此基线实验与增强实验的评估口径仍然一致，可直接对比。")


if __name__ == "__main__":
    main()
