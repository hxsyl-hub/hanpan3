"""Compare GoogLeNet with vs without ImageNet pretrained weights (early stopping on).

Same everything as the baseline run (batch=64, AdamW lr=1e-4, no augmentation);
the ONLY difference between the two runs is pretrained weights, so the comparison
is controlled. Early stopping: stop if val_acc shows no improvement for
`--patience` consecutive epochs. Best weights saved per variant as best.pth.

Usage:
    python compare_pretrained.py                    # both variants, then plots
    python compare_pretrained.py --variant pretrained
    python compare_pretrained.py --patience 15
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import train
from train import (forward_logits, make_loaders, train_one_epoch, evaluate,
                   count_params)

ROOT = Path(__file__).parent
VARIANTS = ["scratch", "pretrained"]
STYLE = {"scratch": ("#7f7f7f", "o"), "pretrained": ("#d62728", "s")}


def build_googlenet(pretrained: bool, num_classes: int) -> nn.Module:
    if pretrained:
        m = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
    else:
        m = models.googlenet(weights=None, init_weights=True)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def train_variant(variant, args, loaders, classes, device):
    out_dir = ROOT / args.out_dir / f"googlenet_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)  # identical batch order for both variants
    model = build_googlenet(pretrained=(variant == "pretrained"),
                            num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = {"model": f"googlenet_{variant}",
               "config": {"batch_size": args.batch_size, "lr": args.lr,
                          "max_epochs": args.epochs, "optimizer": "AdamW",
                          "seed": args.seed, "img_size": 224,
                          "early_stopping_patience": args.patience,
                          "pretrained": variant == "pretrained"},
               "classes": classes, "epochs": []}

    csv_path = out_dir / "log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
             "test_loss", "test_acc", "seconds"])

    best_val_acc, best_epoch = -1.0, 0
    epochs_no_improve = 0
    stopped_epoch = args.epochs
    print(f"\n===== googlenet_{variant}  params={count_params(model):.2f}M =====")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)
        test_loss, test_acc = evaluate(model, loaders["test"], criterion, device)
        dt = time.time() - t0

        history["epochs"].append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "test_loss": test_loss, "test_acc": test_acc, "seconds": dt})
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                 f"{val_loss:.6f}", f"{val_acc:.6f}",
                 f"{test_loss:.6f}", f"{test_acc:.6f}", f"{dt:.1f}"])

        star = ""
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            epochs_no_improve = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "model_name": f"googlenet_{variant}", "classes": classes,
                        "best_val_acc": val_acc, "best_epoch": epoch},
                       out_dir / "best.pth")
            star = "  *best*"
        else:
            epochs_no_improve += 1
        print(f"[googlenet_{variant}] epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}  "
              f"test_acc={test_acc:.4f}  {dt:.1f}s{star}"
              f"  patience={epochs_no_improve}/{args.patience}")
        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if epochs_no_improve >= args.patience:
            stopped_epoch = epoch
            print(f"[googlenet_{variant}] early stop at epoch {epoch} "
                  f"(no val_acc improvement for {args.patience} epochs)")
            break

    best = max(history["epochs"], key=lambda e: e["val_acc"])
    return {"variant": variant,
            "params_M": round(count_params(model), 2),
            "best_val_acc": best_val_acc, "best_epoch": best_epoch,
            "test_acc_at_best": best["test_acc"],
            "stopped_epoch": stopped_epoch,
            "total_seconds": round(sum(e["seconds"] for e in history["epochs"]), 1)}


def plot_compare(out_dir: Path):
    curves_dir = out_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    histories = {}
    for v in VARIANTS:
        h = out_dir / f"googlenet_{v}" / "history.json"
        if h.exists():
            histories[v] = json.loads(h.read_text(encoding="utf-8"))
    if not histories:
        print("no history found, nothing to plot")
        return
    for metric in ["acc", "loss"]:
        for split in ["train", "val", "test"]:
            fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
            for v, hist in histories.items():
                color, marker = STYLE[v]
                epochs = [e["epoch"] for e in hist["epochs"]]
                values = [e[f"{split}_{metric}"] for e in hist["epochs"]]
                ax.plot(epochs, values, label=f"googlenet_{v}", linewidth=1.8,
                        color=color, marker=marker, markersize=3,
                        markevery=max(1, len(epochs) // 10))
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy" if metric == "acc" else "Loss")
            ax.set_title(f"GoogLeNet {split.capitalize()} {metric.capitalize()}: "
                         f"pretrained vs scratch")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="best", framealpha=0.9)
            fig.tight_layout()
            path = curves_dir / f"googlenet_{metric}_{split}.png"
            fig.savefig(path)
            plt.close(fig)
            print(f"saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--out-dir", default="outputs_compare")
    p.add_argument("--variant", default="both", choices=["both", "scratch", "pretrained"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"device={device}  max_epochs={args.epochs}  patience={args.patience}  "
          f"batch={args.batch_size}  lr={args.lr}")
    loaders, classes = make_loaders(args)
    print("classes:", list(classes))

    if args.variant == "both":
        todo = VARIANTS
    else:
        todo = [args.variant]
    results = [train_variant(v, args, loaders, classes, device) for v in todo]

    summary_path = ROOT / args.out_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nsummary saved to {summary_path}")
    for r in results:
        print(f"  googlenet_{r['variant']:10s} best_val_acc={r['best_val_acc']:.4f} "
              f"(ep{r['best_epoch']})  test_acc_at_best={r['test_acc_at_best']:.4f}  "
              f"stopped@{r['stopped_epoch']}  {r['total_seconds']}s")
    plot_compare(ROOT / args.out_dir)


if __name__ == "__main__":
    main()
