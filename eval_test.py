"""用训练好的 best.pth 在指定 split 上做单独评估。

为什么要单独一个脚本：
  train.py 每轮都会算 test ACC，但那只是"顺带看一眼"。这份脚本是训练结束后，
  拿 best.pth（验证集最好的那一轮权重）在测试集上正式跑一次，给出：
    总体 ACC / 宏平均 ACC / 每个类别的 ACC 与支持数 / 混淆矩阵
  委托方文档要求"除了总体上的 ACC，还需要每个类别的 ACC"，且小类的支持数
  必须一起看（某个类只有 3 张时，它的 ACC 跳一个样本就是 33 个百分点）。

用法：
  python eval_test.py                          # 评估全部 5 个模型的 test 集
  python eval_test.py --models googlenet       # 只评一个
  python eval_test.py --split val              # 评验证集（与 test 对照看）
  python eval_test.py --split train            # 评训练集（配合看是否过拟合）
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import train                      # 复用 SolderDataset / build_model / 归一化常量

ROOT = Path(__file__).parent


def evaluate_one(name, split, data_dir, batch_size, device, out_dir):
    ck_path = ROOT / out_dir / name / "best.pth"
    if not ck_path.is_file():
        print(f"  [跳过] {name}: 没有 {ck_path}")
        return None

    ck = torch.load(ck_path, map_location="cpu")
    classes = ck["classes"]                       # {类别名: 索引}
    order = sorted(classes, key=lambda c: classes[c])
    n = len(order)

    model = train.build_model(name, n)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(train.IMAGENET_MEAN, train.IMAGENET_STD),
    ])
    ds = train.SolderDataset(ROOT / "splits" / f"{split}.txt", Path(data_dir), tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2,
                    pin_memory=(device == "cuda"))

    cm = np.zeros((n, n), dtype=np.int64)
    with torch.no_grad():
        for images, targets in dl:
            out = model(images.to(device))
            if not torch.is_tensor(out):          # googlenet 可能返回辅助输出
                out = out.logits
            pred = out.argmax(1).cpu().numpy()
            for t, p in zip(targets.numpy(), pred):
                cm[t, p] += 1

    total = cm.sum()
    acc = float(np.trace(cm) / total) if total else 0.0
    support = cm.sum(axis=1)
    per = np.array([cm[i, i] / support[i] if support[i] else 0.0 for i in range(n)])

    # 保存结果
    out_path = ROOT / out_dir / name
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / f"eval_{split}.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["类别", "支持数", "ACC"])
        for i, c in enumerate(order):
            w.writerow([c, int(support[i]), f"{per[i]:.4f}"])
        w.writerow(["__总体ACC__", int(total), f"{acc:.4f}"])
        w.writerow(["__宏平均ACC__", "", f"{per.mean():.4f}"])

    lines = [
        f"# {name} 在 {split} 集上的评估",
        "",
        f"- 权重：`{ck_path}`（best 轮 = epoch {ck.get('best_epoch', '?')}，"
        f"当时 val_acc = {ck.get('best_val_acc', 0):.4f}）",
        f"- 数据：`splits/{split}.txt`，共 {total} 张",
        f"- 总体 ACC：**{acc:.4f}**   宏平均 ACC：**{per.mean():.4f}**",
        "",
        "| 类别 | 支持数 | ACC |",
        "| --- | --- | --- |",
    ]
    for i, c in enumerate(order):
        lines.append(f"| {c} | {int(support[i])} | {per[i]:.4f} |")
    lines += ["", "## 混淆矩阵（行=真实，列=预测）", "",
              "| 真实\\预测 | " + " | ".join(order) + " |",
              "| --- | " + " | ".join(["---"] * n) + " |"]
    for i, c in enumerate(order):
        lines.append(f"| {c} | " + " | ".join(str(int(v)) for v in cm[i]) + " |")
    (out_path / f"eval_{split}.md").write_text("\n".join(lines), encoding="utf-8")

    return {"model": name, "acc": acc, "macro": float(per.mean()),
            "per": per, "support": support, "order": order, "total": int(total)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=",".join(train.MODEL_NAMES),
                   help="要评估的模型，逗号分隔")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--data-dir", default=str(ROOT / "data" / "raw" / "右焊盘"))
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，改用 CPU")
        device = "cpu"

    results = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"评估 {name} ...")
        r = evaluate_one(name, args.split, args.data_dir, args.batch_size,
                         device, args.out_dir)
        if r:
            results.append(r)
            print(f"  总体 ACC {r['acc']:.4f} | 宏平均 {r['macro']:.4f} "
                  f"→ {args.out_dir}/{name}/eval_{args.split}.md")

    if len(results) > 1:
        print("\n" + "=" * 74)
        print(f"{args.split} 集对比（按总体 ACC 降序）")
        print("=" * 74)
        print("%-16s %12s %14s" % ("模型", "总体 ACC", "宏平均 ACC"))
        print("-" * 74)
        for r in sorted(results, key=lambda x: -x["acc"]):
            print("%-16s %12.4f %14.4f" % (r["model"], r["acc"], r["macro"]))
        print("=" * 74)


if __name__ == "__main__":
    main()
