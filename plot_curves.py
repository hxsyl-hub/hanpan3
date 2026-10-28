"""Plot per-split ACC and LOSS curves (Plan A: one figure per split, all models merged).

Usage:
    python plot_curves.py [--out-dir outputs]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
COLORS = {"vgg16": "#1f77b4", "googlenet": "#ff7f0e", "resnet34": "#2ca02c",
          "resnet50": "#d62728", "mobilenet_v2": "#9467bd"}
MARKERS = {"vgg16": "o", "googlenet": "s", "resnet34": "^",
           "resnet50": "D", "mobilenet_v2": "v"}


def plot_all(out_dir: Path):
    curves_dir = out_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    for h in sorted(out_dir.glob("*/history.json")):
        histories[h.parent.name] = json.loads(h.read_text(encoding="utf-8"))
    if not histories:
        print("no history.json found, run train.py first")
        return

    for metric in ["acc", "loss"]:
        for split in ["train", "val", "test"]:
            fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
            for name, hist in histories.items():
                epochs = [e["epoch"] for e in hist["epochs"]]
                values = [e[f"{split}_{metric}"] for e in hist["epochs"]]
                ax.plot(epochs, values, label=name, linewidth=1.6,
                        color=COLORS.get(name), marker=MARKERS.get(name),
                        markersize=3, markevery=max(1, len(epochs) // 10))
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy" if metric == "acc" else "Loss")
            ax.set_title(f"{split.capitalize()} {metric.capitalize()}")
            if metric == "acc":
                ax.set_ylim(0, 1.02)
                ax.grid(True, linestyle="--", alpha=0.4)
            else:
                ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="best", framealpha=0.9)
            fig.tight_layout()
            path = curves_dir / f"{metric}_{split}.png"
            fig.savefig(path)
            plt.close(fig)
            print(f"saved {path}")


def plot_per_model(out_dir: Path, histories: dict):
    """每个模型单独一张图，把 train/val/test 三条曲线画在一起。

    用途：判断单个模型是否过拟合 —— train 持续上升而 val/test 走平或掉头，
    就是典型的过拟合，在一张图里看三条线的间距最直观。
    """
    curves_dir = out_dir / "curves" / "by_model"
    curves_dir.mkdir(parents=True, exist_ok=True)
    # 训练集实线、验证集虚线、测试集点划线，即使黑白打印也能区分
    STYLE = {"train": ("#1f77b4", "-", "o"),
             "val": ("#ff7f0e", "--", "s"),
             "test": ("#2ca02c", "-.", "^")}

    for name, hist in sorted(histories.items()):
        epochs = [e["epoch"] for e in hist["epochs"]]
        for metric in ["acc", "loss"]:
            fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
            for split in ["train", "val", "test"]:
                color, ls, mk = STYLE[split]
                values = [e[f"{split}_{metric}"] for e in hist["epochs"]]
                ax.plot(epochs, values, label=split, color=color, linestyle=ls,
                        marker=mk, linewidth=1.8, markersize=3.5,
                        markevery=max(1, len(epochs) // 10))
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy" if metric == "acc" else "Loss")
            ax.set_title(f"{name}  {metric.upper()}: train vs val vs test")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="best", framealpha=0.9)
            if metric == "acc":
                ax.set_ylim(0, 1.02)
            fig.tight_layout()
            path = curves_dir / f"{name}_{metric}.png"
            fig.savefig(path)
            plt.close(fig)
            print(f"saved {path}")


def plot_overfit_gap(out_dir: Path, histories: dict):
    """过拟合差距图：train ACC - val ACC 随 epoch 的变化，并给出排名。

    这条曲线越往上，说明模型越是把训练集背下来而对验证集没有帮助。
    """
    curves_dir = out_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    summary = []
    for name, hist in sorted(histories.items()):
        epochs = [e["epoch"] for e in hist["epochs"]]
        gap = [e["train_acc"] - e["val_acc"] for e in hist["epochs"]]
        ax.plot(epochs, gap, label=name, color=COLORS.get(name),
                marker=MARKERS.get(name), linewidth=1.6, markersize=3,
                markevery=max(1, len(epochs) // 10))
        summary.append((name, gap[-1], max(gap)))
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train ACC - Val ACC")
    ax.set_title("Overfitting gap (higher = more overfitting)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    path = curves_dir / "overfit_gap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")

    print("\n过拟合程度排名（末轮 train-val 差距，越大越严重）：")
    for name, last, mx in sorted(summary, key=lambda x: -x[1]):
        print(f"  {name:14s} 末轮 {last:+.4f}   峰值 {mx:+.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args()
    out_dir = ROOT / args.out_dir
    histories = {}
    for h in sorted(out_dir.glob("*/history.json")):
        histories[h.parent.name] = json.loads(h.read_text(encoding="utf-8"))
    if not histories:
        print("no history.json found, run train.py first")
    else:
        plot_all(out_dir)                       # 每个 split 一张，5 模型合并（横向比模型）
        plot_per_model(out_dir, histories)      # 每个模型一张，三条线（看单个模型过拟合）
        plot_overfit_gap(out_dir, histories)    # 过拟合差距排名（找最严重的）
