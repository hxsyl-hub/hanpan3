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
from PIL import Image, ImageOps

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


def build_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """构建模型。

    pretrained: 是否加载 ImageNet 预训练权重。
      文档步骤2 —— "拿步骤1最好的模型，加上预训练权重去对比，
      有预训练和无预训练权重的差异"。
      注意无论是否预训练，分类头都重建成本任务类别数（预训练头是 1000 类）。
    """
    w = "DEFAULT" if pretrained else None
    if name == "vgg16":
        m = models.vgg16(weights=w)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    elif name == "googlenet":
        # init_weights 只在无预训练时需要（有预训练时它会覆盖加载到的权重）
        m = models.googlenet(weights=w, init_weights=not pretrained)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        # 辅助分类器的输出层也要换，否则训练模式下 aux 分支输出仍是 1000 类
        if m.aux_logits:
            m.aux1.fc2 = nn.Linear(m.aux1.fc2.in_features, num_classes)
            m.aux2.fc2 = nn.Linear(m.aux2.fc2.in_features, num_classes)
    elif name == "resnet34":
        m = models.resnet34(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet50":
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"unknown model: {name}")
    return m


def head_modules(model, name):
    """返回该模型的"分类头"模块列表 —— 冻结骨干时只有这些层参与训练。"""
    if name == "vgg16":
        return [model.classifier[6]]
    if name == "googlenet":
        heads = [model.fc]
        if getattr(model, "aux_logits", False):
            heads += [model.aux1.fc2, model.aux2.fc2]
        return heads
    if name.startswith("resnet"):
        return [model.fc]
    if name == "mobilenet_v2":
        return [model.classifier[1]]
    raise ValueError(f"unknown model: {name}")


def freeze_backbone(model, name):
    """冻结骨干，只训练分类头（文档步骤 3）。

    前提：必须配合 --pretrained 使用。若骨干是随机初始化再冻住，
    等于只训一个随机分类头，学不到东西。
    返回可训练参数量，用于日志确认确实冻住了。
    """
    for p in model.parameters():
        p.requires_grad = False
    for h in head_modules(model, name):
        for p in h.parameters():
            p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def forward_logits(model, images):
    out = model(images)
    if not torch.is_tensor(out):  # googlenet in train mode returns aux outputs
        out = out.logits
    return out


def exp_out_root(args) -> Path:
    """实验输出根目录：按"训练集怎么来的"自动加后缀。

    不加这个的话，后续实验会直接写进 outputs/<模型>/ 把基线的
    history.json、best.pth 覆盖掉，之后想对比都没了。
      基线        -> outputs/
      增强消融    -> outputs_hist/ （或 _flip / _rotate / _blur）
      在线叠加    -> outputs_online/
      离线增强    -> outputs_train_aug/  （--train-list 指向 train_aug.txt 时）
      预训练      -> outputs_pretrained/
      预训练+冻结 -> outputs_pretrained_freeze/
    """
    suffix = "" if args.augment == "none" else f"_{args.augment}"
    if args.train_list:
        suffix += "_" + Path(args.train_list).stem
    if args.pretrained:
        suffix += "_pretrained"
    if args.freeze:
        suffix += "_freeze"
    return ROOT / (args.out_dir.rstrip("/\\") + suffix)


def _equalize(img):
    """直方图均衡，必须在 PIL 图像上做（放在 ToTensor 之前）。

    文档提到数据"最大的困难是曝光过度，有一些图偏暗"，
    均衡能拉开对比度，对这类图通常有效。
    """
    return ImageOps.equalize(img)


def build_train_transform(augment: str):
    """训练集的在线增强：只在训练时随机作用，不写盘、不改变图片总数。

    文档原话："根据黑白图的特点，每次只选其一，直方图、水平垂直翻转、
    角度旋转、模糊（常规）。"
    所以 hist / flip / rotate / blur 是一组**消融实验**，每个选项只启用
    一种变换，这样才能单独看出哪一种增强真的有用；
    online 是它们叠加起来的对照组。
    """
    hist = transforms.RandomApply([transforms.Lambda(_equalize)], p=0.5)
    flip = [transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)]
    rotate = transforms.RandomRotation(degrees=15)
    blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
    color = transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                   saturation=0.2)
    table = {
        "hist": [hist],
        "flip": flip,
        "rotate": [rotate],
        "blur": [blur],
        "online": [hist] + flip + [rotate, color],
    }
    ops = table.get(augment, [])
    return transforms.Compose(
        [transforms.Resize((224, 224))] + ops
        + [transforms.ToTensor(),
           transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


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
    if args.augment == "none":
        train_tf = eval_tf        # 基线：与评估口径完全一致，不做任何增强
    else:
        train_tf = build_train_transform(args.augment)

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


# 严重缺陷：参照同类焊盘项目的划分（CX / HDZ / PW / XH2）。
# 这几类漏检代价最高，单独统计其召回率，比看总体 ACC 更能反映能否上线。
SEVERE_CLASSES = [
    "NG_PadDiscoloration",   # CX   焊盘变色
    "NG_PadContamination",   # HDZ  焊盘污染
    "NG_Misalignment",       # PW   偏移
    "NG_ColdJoint",          # XH2  虚焊
]


def compute_class_metrics(preds, targets, num_classes):
    """按类算 精确率 / 召回率 / 漏检率 / F1 / 支持数，并给出混淆矩阵。

    漏检率 = 1 - 召回率 = 该类里被判成别的比例。
    工业场景最关心这个：把缺陷判成合格（漏检）的代价，
    远高于把合格判成缺陷（后者只是多复检一次）。
    """
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets.tolist(), preds.tolist()):
        cm[t][p] += 1
    rows = []
    for i in range(num_classes):
        tp = cm[i, i].item()
        support = cm[i, :].sum().item()        # 真实是 i 的总数
        pred_as_i = cm[:, i].sum().item()      # 被预测成 i 的总数
        fn = support - tp                      # 漏检：真实 i 判成了别的
        fp = pred_as_i - tp                    # 误报：别的判成了 i
        precision = tp / pred_as_i if pred_as_i else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        rows.append({"index": i, "support": support, "correct": tp,
                     "missed": fn, "false_alarm": fp,
                     "acc": recall,            # 每类 ACC 就是该类召回率
                     "precision": precision, "recall": recall,
                     "miss_rate": 1.0 - recall, "f1": f1})
    return rows, cm


def evaluate(model, loader, criterion, device, num_classes=None):
    """返回 (loss, 总体acc, 每类指标列表 or None, 混淆矩阵 or None)。

    文档要求"除了总体 ACC，还需要每个类别的 ACC"。这里在此基础上
    把精确率 / 召回率 / 漏检率 / F1 一起算了 —— 只是多统计一次混淆矩阵，
    但工业交付时漏检率才是真正的验收指标。
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_pred, all_true = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = forward_logits(model, images)
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            pred = logits.argmax(1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
            if num_classes:
                all_pred.append(pred.cpu())
                all_true.append(targets.cpu())
    if not num_classes:
        return total_loss / total, correct / total, None, None
    rows, cm = compute_class_metrics(torch.cat(all_pred), torch.cat(all_true),
                                     num_classes)
    return total_loss / total, correct / total, rows, cm


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def train_model(name, args, loaders, classes, device):
    out_dir = exp_out_root(args) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(classes)

    torch.manual_seed(args.seed)
    model = build_model(name, num_classes, pretrained=args.pretrained).to(device)
    trainable_M = count_params(model)
    if args.freeze:
        if not args.pretrained:
            print("  ⚠ --freeze 未配合 --pretrained：骨干是随机初始化的，"
                  "冻住之后只剩一个随机分类头在学，结果没有意义。")
        trainable_M = freeze_backbone(model, name) / 1e6
    criterion = nn.CrossEntropyLoss()
    # 冻结时只把需要梯度的参数交给优化器，否则 AdamW 也会为冻结参数建立动量状态
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    history = {"model": name,
               "config": {"batch_size": args.batch_size, "lr": args.lr,
                          "epochs": args.epochs, "optimizer": "AdamW",
                          "seed": args.seed, "img_size": 224,
                          "pretrained": args.pretrained,
                          "freeze": args.freeze,
                          "augment": args.augment,
                          "train_list": args.train_list or "splits/train.txt",
                          "trainable_M": round(trainable_M, 2)},
               "classes": classes, "epochs": []}

    csv_path = out_dir / "log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
             "test_loss", "test_acc", "seconds"])

    best_val_acc = -1.0
    best_epoch = 0
    best_per = None          # best 轮次的每类 ACC（val 与 test 各一份）
    tags = []
    if args.pretrained:
        tags.append("预训练")
    if args.freeze:
        tags.append("冻结骨干")
    if args.augment != "none":
        tags.append(f"增强:{args.augment}")
    if args.train_list:
        tags.append(f"清单:{Path(args.train_list).stem}")
    head = name + (f" [{'/'.join(tags)}]" if tags else "")
    extra = f"  可训练={trainable_M:.2f}M" if args.freeze else ""
    print(f"\n===== {head}  params={count_params(model):.2f}M{extra} =====")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc, val_rows, val_cm = evaluate(
            model, loaders["val"], criterion, device, num_classes)
        test_loss, test_acc, test_rows, test_cm = evaluate(
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
                        "val": val_rows, "test": test_rows,
                        "test_cm": test_cm.tolist()}
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

    # ---- 每类指标（文档要求每类 ACC；工业验收还要召回率 / 漏检率）----
    macro_test = 0.0
    if best_per:
        order = sorted(classes, key=lambda c: classes[c])   # classes: {类名: 索引}
        with open(out_dir / "per_class_metrics.csv", "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["类别", "严重缺陷", "test支持数", "test_ACC",
                        "test_精确率", "test_召回率", "test_漏检率", "test_F1",
                        "val支持数", "val_ACC", "val_漏检率"])
            for c in order:
                i = classes[c]
                t, v = best_per["test"][i], best_per["val"][i]
                w.writerow([c, "★" if c in SEVERE_CLASSES else "",
                            t["support"], f"{t['acc']:.4f}",
                            f"{t['precision']:.4f}", f"{t['recall']:.4f}",
                            f"{t['miss_rate']:.4f}", f"{t['f1']:.4f}",
                            v["support"], f"{v['acc']:.4f}",
                            f"{v['miss_rate']:.4f}"])

        # 混淆矩阵：行=真实类别，列=预测类别（best 轮的 test）
        with open(out_dir / "confusion_matrix.csv", "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["真实\\预测"] + order)
            for c in order:
                w.writerow([c] + list(best_per["test_cm"][classes[c]]))

        print(f"\n[{name}] 每类指标（best 轮 = epoch {best_per['epoch']}）：")
        print("    %-24s %4s %7s %9s %9s %9s %9s %8s"
              % ("类别", "严重", "test数", "ACC", "精确率", "召回率", "漏检率", "F1"))
        for c in order:
            i = classes[c]
            t = best_per["test"][i]
            mark = "★" if c in SEVERE_CLASSES else ""
            thin = "   ⚠ 样本太少，比例不可靠" if t["support"] < 10 else ""
            print("    %-24s %4s %7d %9.4f %9.4f %9.4f %9.4f %8.4f%s"
                  % (c, mark, t["support"], t["acc"], t["precision"],
                     t["recall"], t["miss_rate"], t["f1"], thin))
        macro_test = sum(best_per["test"][classes[c]]["acc"]
                         for c in order) / len(order)
        print(f"    宏平均 ACC(test) = {macro_test:.4f}"
              f"   （总体 ACC(test) = "
              f"{history['epochs'][best_per['epoch'] - 1]['test_acc']:.4f}）")

        # 严重缺陷单独统计：漏检代价最高，验收盯的是这个而不是总体 ACC
        severe = [c for c in order if c in SEVERE_CLASSES]
        if severe:
            rec = [best_per["test"][classes[c]]["recall"] for c in severe]
            miss = [best_per["test"][classes[c]]["miss_rate"] for c in severe]
            print("    " + "-" * 62)
            print("    严重缺陷汇总（%s）：" % " / ".join(severe))
            print("      平均召回率 %.4f    平均漏检率 %.4f"
                  % (sum(rec) / len(rec), sum(miss) / len(miss)))
            for c in severe:
                t = best_per["test"][classes[c]]
                flag = "   ⚠ 漏检率过高" if t["miss_rate"] > 0.15 else ""
                thin = ("   注意 test 仅 %d 张，该比例不稳定"
                        % t["support"]) if t["support"] < 10 else ""
                print("      %-24s 漏检 %d/%d = %5.1f%%%s%s"
                      % (c, t["missed"], t["support"],
                         t["miss_rate"] * 100, flag, thin))
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
    p.add_argument("--augment", default="none",
                   choices=["none", "hist", "flip", "rotate", "blur", "online"],
                   help="训练集在线增强。文档要求“根据黑白图的特点，每次只选其一”，"
                        "所以 hist/flip/rotate/blur 各自只启用一种变换，用于消融对比；"
                        "online 是几种叠加的对照组。都不影响 val/test 的口径")
    p.add_argument("--pretrained", action="store_true",
                   help="加载 ImageNet 预训练权重（基线对比不要加；"
                        "文档步骤2 用它对比有无预训练的差异）")
    p.add_argument("--freeze", action="store_true",
                   help="冻结骨干只训练分类头，须配合 --pretrained；"
                        "文档步骤3 用它对比冻结与全量微调的差异")
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
