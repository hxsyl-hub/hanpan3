"""Train baseline models (no pretrained weights) on the solder-joint dataset.

Usage:
    python train.py                      # train all 5 models sequentially
    python train.py --model resnet34     # train one model
    python train.py --device cuda        # force device
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

ROOT = Path(__file__).parent
MODEL_NAMES = ["vgg16", "googlenet", "resnet34", "resnet50", "mobilenet_v2"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SolderDataset(Dataset):
    def __init__(self, list_file: Path, data_dir: Path, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        with open(list_file, encoding="utf-8") as f:
            for line in f:
                rel, idx = line.strip().split("\t")
                self.samples.append((rel, int(idx)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        rel, idx = self.samples[i]
        img = Image.open(self.data_dir / rel).convert("RGB")
        return self.transform(img), idx


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "vgg16":
        m = models.vgg16(weights=None)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    elif name == "googlenet":
        m = models.googlenet(weights=None, init_weights=True)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet34":
        m = models.resnet34(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"unknown model: {name}")
    return m


def forward_logits(model, images):
    out = model(images)
    if not torch.is_tensor(out):  # googlenet in train mode returns aux outputs
        out = out.logits
    return out


def exp_out_root(args) -> Path:
    """实验输出根目录：按"训练集怎么来的"自动加后缀。

    不加这个的话，跑增强实验会直接写进 outputs/<模型>/ 把基线的
    history.json、best.pth 覆盖掉，之后想对比都没了。
      基线      -> outputs/
      在线增强  -> outputs_online/
      离线增强  -> outputs_train_aug/   （--train-list 指向 train_aug.txt 时）
    """
    suffix = "" if args.augment == "none" else f"_{args.augment}"
    if args.train_list:
        suffix += "_" + Path(args.train_list).stem
    return ROOT / (args.out_dir.rstrip("/\\") + suffix)


def make_loaders(args):
    """构建三个 DataLoader。

    对比实验的公平性由这里保证，有两条铁律：
      1. val / test 的清单永远取 splits/ 下那份，预处理永远不含任何随机变换；
      2. 只有训练集可以被换掉或加增强（--train-list / --augment）。
    这样无论训练集怎么增强，评估口径始终一致，实验结果才能横向对比。
    """
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    if args.augment == "online":
        # 在线增强：只在训练时随机变换，每个 epoch 看到的图都不一样（不写盘）
        train_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                   saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        train_tf = eval_tf            # 基线：与评估口径完全一致，不做任何增强

    splits_dir = ROOT / "splits"
    data_dir = Path(args.data_dir)
    train_list = (Path(args.train_list) if args.train_list
                  else splits_dir / "train.txt")
    loaders = {}
    for split in ["train", "val", "test"]:
        list_file = train_list if split == "train" else splits_dir / f"{split}.txt"
        tf = train_tf if split == "train" else eval_tf
        ds = SolderDataset(list_file, data_dir, tf)
        loaders[split] = DataLoader(
            ds, batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers, pin_memory=True)
    with open(splits_dir / "classes.json", encoding="utf-8") as f:
        classes = json.load(f)
    return loaders, classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = forward_logits(model, images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * targets.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=None):
    """返回 (loss, 总体acc, 每类acc列表, 每类支持数列表)。

    传入 num_classes 时额外统计每个类别的 ACC —— 委托方文档要求：
    “评价指标：除了总体上的 ACC，还需要每个类别的 ACC”。
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    per_correct = torch.zeros(num_classes, dtype=torch.long) if num_classes else None
    per_total = torch.zeros(num_classes, dtype=torch.long) if num_classes else None
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * targets.size(0)
        pred = logits.argmax(1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
        if per_correct is not None:
            t = targets.cpu()
            p = pred.cpu()
            per_correct += torch.bincount(t[p == t], minlength=num_classes)
            per_total += torch.bincount(t, minlength=num_classes)
    if per_correct is None:
        return total_loss / total, correct / total, None, None
    support = per_total.tolist()
    per_acc = [per_correct[i].item() / support[i] if support[i] else 0.0
               for i in range(num_classes)]
    return total_loss / total, correct / total, per_acc, support


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def train_model(name, args, loaders, classes, device):
    out_dir = exp_out_root(args) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(classes)

    torch.manual_seed(args.seed)
    model = build_model(name, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = {"model": name,
               "config": {"batch_size": args.batch_size, "lr": args.lr,
                          "epochs": args.epochs, "optimizer": "AdamW",
                          "seed": args.seed, "img_size": 224},
               "classes": classes, "epochs": []}

    csv_path = out_dir / "log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
             "test_loss", "test_acc", "seconds"])

    best_val_acc = -1.0
    best_epoch = 0
    best_per = None          # best 轮次的每类 ACC（val 与 test 各一份）
    print(f"\n===== {name}  params={count_params(model):.2f}M =====")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc, val_per, val_sup = evaluate(
            model, loaders["val"], criterion, device, num_classes)
        test_loss, test_acc, test_per, test_sup = evaluate(
            model, loaders["test"], criterion, device, num_classes)
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
            best_per = {"epoch": epoch,
                        "val": val_per, "val_support": val_sup,
                        "test": test_per, "test_support": test_sup}
            torch.save({"model_state_dict": model.state_dict(),
                        "model_name": name, "classes": classes,
                        "best_val_acc": val_acc, "best_epoch": epoch,
                        "per_class_acc": best_per},
                       out_dir / "best.pth")
            star = "  *best*"
        print(f"[{name}] epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}  "
              f"test_acc={test_acc:.4f}  {dt:.1f}s{star}")
        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"[{name}] done. best_val_acc={best_val_acc:.4f} @ epoch {best_epoch}")

    # ---- 每类 ACC（委托方文档要求：除了总体 ACC，还要每个类别的 ACC）----
    macro_test = 0.0
    if best_per:
        order = sorted(classes, key=lambda c: classes[c])   # classes: {类名: 索引}
        with open(out_dir / "per_class_acc.csv", "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["类别", "val支持数", "val_ACC", "test支持数", "test_ACC"])
            for c in order:
                i = classes[c]
                w.writerow([c, best_per["val_support"][i], f"{best_per['val'][i]:.4f}",
                            best_per["test_support"][i], f"{best_per['test'][i]:.4f}"])
        print(f"[{name}] 每类 ACC（best 轮 = epoch {best_per['epoch']}）：")
        print("    %-24s %7s %9s %8s %9s"
              % ("类别", "val数", "val_ACC", "test数", "test_ACC"))
        for c in order:
            i = classes[c]
            print("    %-24s %7d %9.4f %8d %9.4f"
                  % (c, best_per["val_support"][i], best_per["val"][i],
                     best_per["test_support"][i], best_per["test"][i]))
        macro_test = sum(best_per["test"][classes[c]] for c in order) / len(order)
        print(f"    宏平均 ACC(test) = {macro_test:.4f}"
              f"   （总体 ACC(test) = {history['epochs'][best_per['epoch'] - 1]['test_acc']:.4f}）")
        history["best_per_class_acc"] = best_per

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return {"model": name, "params_M": round(count_params(model), 2),
            "best_val_acc": best_val_acc, "best_epoch": best_epoch,
            "final_test_acc": history["epochs"][-1]["test_acc"],
            "macro_test_acc": round(macro_test, 4),
            "total_seconds": round(sum(e["seconds"] for e in history["epochs"]), 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(ROOT / "data" / "raw" / "右焊盘"),
                   help="原始图根目录（与 split_dataset.py 的 DATA_DIR 必须一致，"
                        "因为 splits/*.txt 里存的是相对它的路径）")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--models", default=",".join(MODEL_NAMES),
                   help="comma-separated model names")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--augment", default="none", choices=["none", "online"],
                   help="none=基线，不做任何增强（默认）；"
                        "online=训练集在线随机增强。两者都不影响 val/test 的口径")
    p.add_argument("--train-list", default=None,
                   help="训练集清单，默认 splits/train.txt。"
                        "离线增强实验可指向 train_aug.txt（只加训练图），"
                        "val/test 仍固定取 splits/ 下那份，保证结果可比")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"device={device}  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print(f"augment={args.augment}  训练集清单={args.train_list or 'splits/train.txt'}")
    print(f"实验输出目录: {exp_out_root(args)}"
          f"  （增强方式不同会自动分开存，不会覆盖之前的结果）")
    loaders, classes = make_loaders(args)
    print("classes:", list(classes))

    results = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        results.append(train_model(name, args, loaders, classes, device))

    summary_path = exp_out_root(args) / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nsummary saved to {summary_path}")
    print("\n" + "=" * 74)
    print("%-16s %13s %14s %14s %10s"
          % ("模型", "best_val_ACC", "末轮test_ACC", "宏平均test_ACC", "用时(s)"))
    print("-" * 74)
    for r in sorted(results, key=lambda x: -x["best_val_acc"]):
        print("%-16s %13.4f %14.4f %14.4f %10.1f"
              % (r["model"], r["best_val_acc"], r["final_test_acc"],
                 r.get("macro_test_acc", 0.0), r["total_seconds"]))
    print("=" * 74)


if __name__ == "__main__":
    main()
