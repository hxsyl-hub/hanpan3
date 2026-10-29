# -*- coding: utf-8 -*-
"""一键跑完后续实验（文档步骤 2、3 与数据增强 ①②③）。

基线的五模型对比已经跑完，最优是 googlenet，所以后续实验默认只跑这一个
模型 —— 五个模型全跑一遍既费时又没必要。

每组实验一个独立输出目录（由 train.exp_out_root 按开关自动加后缀），
彼此不会互相覆盖；串行执行，中断后再跑会自动跳过已完成的组。

用法：
    python run_next.py --data-dir images                 # 跑全部
    python run_next.py --data-dir images --dry-run       # 只看计划
    python run_next.py --data-dir images --only C1,C2    # 只跑翻转/旋转消融
    python run_next.py --data-dir images --skip D1,E1    # 跳过离线相关
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import train  # noqa: E402  复用 SEVERE_CLASSES 与 MODEL_NAMES

# 每组实验 = 基线超参 + 一组开关。augment / train_list / pretrained / freeze
# 四个开关共同决定输出目录名，所以不同组天然分开，不会互相覆盖。
EXPERIMENTS = [
    # 文档步骤 2：拿步骤1最好的模型，对比有 / 无预训练
    {"id": "A2", "group": "步骤2 预训练对比", "desc": "有预训练权重",
     "augment": "none", "train_list": None, "pretrained": True,
     "freeze": False},
    # 文档步骤 3：在预训练基础上，对比冻结骨干 vs 全量微调
    {"id": "B2", "group": "步骤3 冻结对比", "desc": "预训练+冻结骨干(只训分类头)",
     "augment": "none", "train_list": None, "pretrained": True,
     "freeze": True},
    # 增强①：在线增强消融，文档要求"每次只选其一"
    {"id": "C1", "group": "增强① 在线消融", "desc": "直方图均衡",
     "augment": "hist", "train_list": None, "pretrained": False,
     "freeze": False},
    {"id": "C2", "group": "增强① 在线消融", "desc": "水平+垂直翻转",
     "augment": "flip", "train_list": None, "pretrained": False,
     "freeze": False},
    {"id": "C3", "group": "增强① 在线消融", "desc": "角度旋转 ±15°",
     "augment": "rotate", "train_list": None, "pretrained": False,
     "freeze": False},
    {"id": "C4", "group": "增强① 在线消融", "desc": "高斯模糊",
     "augment": "blur", "train_list": None, "pretrained": False,
     "freeze": False},
    {"id": "C5", "group": "增强① 在线消融", "desc": "四种叠加(对照组)",
     "augment": "online", "train_list": None, "pretrained": False,
     "freeze": False},
    # 增强②：离线增强，小类 +50%~80%
    {"id": "D1", "group": "增强② 离线扩充", "desc": "小类离线增强 +50%",
     "augment": "none", "train_list": "splits/train_aug.txt",
     "pretrained": False, "freeze": False},
    # 增强③：在线 + 离线叠加
    {"id": "E1", "group": "增强③ 叠加", "desc": "离线扩充 + 在线增强",
     "augment": "online", "train_list": "splits/train_aug.txt",
     "pretrained": False, "freeze": False},
]

# 用到离线增强清单的实验编号
NEEDS_AUG_LIST = {e["id"] for e in EXPERIMENTS if e["train_list"]}


def out_root_for(exp, base_out: str) -> Path:
    """预测该实验的输出根目录 —— 与 train.exp_out_root 的规则保持一致。

    如果以后改了 train.py 里的命名规则，这里也要跟着改，否则汇总时找不到目录。
    """
    suffix = "" if exp["augment"] == "none" else "_" + exp["augment"]
    if exp["train_list"]:
        suffix += "_" + Path(exp["train_list"]).stem
    if exp["pretrained"]:
        suffix += "_pretrained"
    if exp["freeze"]:
        suffix += "_freeze"
    return ROOT / (base_out.rstrip("/\\") + suffix)


def is_finished(root: Path, model: str, epochs: int) -> bool:
    """该组是否已跑满 epochs 轮（用于中断后接着跑）。"""
    h = root / model / "history.json"
    if not h.is_file():
        return False
    try:
        d = json.loads(h.read_text(encoding="utf-8"))
    except Exception:
        return False
    return len(d.get("epochs", [])) >= epochs


def ensure_aug_list(args) -> bool:
    """离线增强实验需要先生成 splits/train_aug.txt。"""
    aug_list = ROOT / "splits" / "train_aug.txt"
    if aug_list.is_file():
        return True
    if args.dry_run:
        print("  [待生成] splits/train_aug.txt（正式跑之前会先调用 "
              "make_offline_aug.py 生成）")
        return False
    print("  → 未找到 splits/train_aug.txt，先生成离线增强图 ...")
    cmd = [sys.executable, "make_offline_aug.py",
           "--data-dir", args.data_dir, "--ratio", str(args.aug_ratio)]
    r = subprocess.run(cmd, cwd=str(ROOT))
    return r.returncode == 0 and aug_list.is_file()


def build_cmd(exp, args):
    cmd = [sys.executable, "train.py",
           "--data-dir", args.data_dir,
           "--models", args.models,
           "--epochs", str(args.epochs),
           "--batch-size", str(args.batch_size),
           "--lr", str(args.lr),
           "--out-dir", args.out_dir,
           "--augment", exp["augment"]]
    if exp["train_list"]:
        cmd += ["--train-list", exp["train_list"]]
    if exp["pretrained"]:
        cmd.append("--pretrained")
    if exp["freeze"]:
        cmd.append("--freeze")
    return cmd


def run_exp(exp, args, log_dir):
    cmd = build_cmd(exp, args)
    if args.dry_run:
        print("    " + " ".join(cmd))
        return "dry"
    log_path = log_dir / ("%s_%s.log" % (exp["id"], exp["augment"]))
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        code = p.wait()
    dt = time.time() - t0
    print("    → %s（%.1f 分钟，日志 logs/%s）"
          % ("完成" if code == 0 else "失败 code=%d" % code, dt / 60,
             log_path.name))
    return "ok" if code == 0 else "fail"


def summarize(args, model):
    """读各组 history.json 的 best 轮指标，汇总成一张对比表。"""
    rows = []
    for exp in EXPERIMENTS:
        root = out_root_for(exp, args.out_dir)
        h = root / model / "history.json"
        if not h.is_file():
            rows.append((exp["id"], exp["group"], exp["desc"], None, None,
                         None))
            continue
        try:
            d = json.loads(h.read_text(encoding="utf-8"))
        except Exception:
            continue
        ep = d.get("epochs", [])
        if not ep:
            continue
        bp = d.get("best_per_class_acc") or {}
        # best 轮的 epoch 从 1 开始，对应 epochs 列表的下标要减 1
        bi = min((bp.get("epoch") or len(ep)) - 1, len(ep) - 1)
        acc = ep[bi]["test_acc"]
        test_rows = bp.get("test") or []
        macro = (sum(r["acc"] for r in test_rows) / len(test_rows)
                 if test_rows else 0.0)
        cls = d.get("classes", {})
        idx2name = {i: c for c, i in cls.items()}
        sev = [r for r in test_rows
               if idx2name.get(r["index"], "") in train.SEVERE_CLASSES]
        smiss = sum(r["miss_rate"] for r in sev) / len(sev) if sev else 0.0
        rows.append((exp["id"], exp["group"], exp["desc"], acc, macro,
                     smiss))

    print("\n" + "=" * 98)
    print("%s 各实验对比（best 轮指标，按总体 ACC 降序）" % model)
    print("=" * 98)
    print("%-5s %-16s %-26s %10s %10s %14s"
          % ("编号", "组", "说明", "总体ACC", "宏平均", "严重缺陷漏检"))
    print("-" * 98)
    done = [r for r in rows if r[3] is not None]
    for r in sorted(done, key=lambda x: -x[3]):
        print("%-5s %-16s %-26s %10.4f %10.4f %13.1f%%"
              % (r[0], r[1], r[2], r[3], r[4], r[5] * 100))
    for r in rows:
        if r[3] is None:
            print("%-5s %-16s %-26s %10s %10s %14s"
                  % (r[0], r[1], r[2], "未跑", "-", "-"))
    print("=" * 98)
    print("严重缺陷 = " + " / ".join(train.SEVERE_CLASSES))
    print("选模型别只看总体 ACC：工业场景优先看最后一列，漏检率越低越安全")
    if done:
        best = min(done, key=lambda x: x[5])
        print("本轮漏检率最低：%s（%s，%.1f%%）" % (best[0], best[2],
                                                 best[5] * 100))


def main():
    p = argparse.ArgumentParser(
        description="一键跑完后续实验（预训练 / 冻结 / 增强消融）")
    p.add_argument("--data-dir", default=str(ROOT / "data" / "raw" / "右焊盘"),
                   help="原始图根目录（云端解压到英文目录时用 images）")
    p.add_argument("--models", default="googlenet",
                   help="要跑的模型，逗号分隔。基线已定 googlenet，默认只跑它")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out-dir", default="outputs",
                   help="输出根目录名，每组实验会自动加后缀区分")
    p.add_argument("--aug-ratio", type=float, default=0.5,
                   help="离线增强时小类扩充比例（文档要求 50%~80%）")
    p.add_argument("--only", default=None,
                   help="只跑指定编号，逗号分隔，如 A2,C1")
    p.add_argument("--skip", default=None, help="跳过指定编号")
    p.add_argument("--dry-run", action="store_true", help="只打印计划不训练")
    p.add_argument("--no-summary", action="store_true", help="跳过最后汇总")
    args = p.parse_args()

    print("=" * 98)
    print("后续实验一键执行")
    print("=" * 98)
    print("数据目录: %s" % args.data_dir)
    print("模型:     %s" % args.models)
    print("超参:     epochs=%d  batch=%d  lr=%g"
          % (args.epochs, args.batch_size, args.lr))
    if args.dry_run:
        print("模式:     dry-run（只看计划，不训练）")

    todo = list(EXPERIMENTS)
    if args.only:
        want = {s.strip().upper() for s in args.only.split(",") if s.strip()}
        todo = [e for e in todo if e["id"] in want]
    if args.skip:
        drop = {s.strip().upper() for s in args.skip.split(",") if s.strip()}
        todo = [e for e in todo if e["id"] not in drop]

    if not todo:
        print("\n没有要跑的实验（--only / --skip 过滤后为空）")
        return

    log_dir = ROOT / "logs"
    if not args.dry_run:
        log_dir.mkdir(exist_ok=True)

    # 离线增强相关实验需要先有 train_aug.txt
    if any(e["id"] in NEEDS_AUG_LIST for e in todo) and not args.dry_run:
        if not ensure_aug_list(args):
            print("  ⚠ 离线增强清单未生成，D1 / E1 会失败（图读不到）")

    print("\n共 %d 组实验：" % len(todo))
    for i, exp in enumerate(todo, 1):
        root = out_root_for(exp, args.out_dir)
        print("  %d. %-4s %-16s %-26s → %s/" % (i, exp["id"], exp["group"],
                                                 exp["desc"], root.name))

    results = []
    for i, exp in enumerate(todo, 1):
        root = out_root_for(exp, args.out_dir)
        first_model = [m.strip() for m in args.models.split(",")
                       if m.strip()][0]
        print("\n[%d/%d] %s  %s —— %s"
              % (i, len(todo), exp["id"], exp["group"], exp["desc"]))
        if is_finished(root, first_model, args.epochs):
            print("    已跑满 %d 轮，跳过（要重跑请先删掉 %s/%s）"
                  % (args.epochs, root.name, first_model))
            results.append((exp["id"], "skip"))
            continue
        results.append((exp["id"], run_exp(exp, args, log_dir)))

    print("\n" + "=" * 98)
    ok = sum(1 for _, s in results if s == "ok")
    skip = sum(1 for _, s in results if s == "skip")
    fail = [i for i, s in results if s == "fail"]
    print("执行结果：完成 %d，跳过 %d，失败 %d" % (ok, skip, len(fail)))
    if fail:
        print("失败编号：%s（日志见 logs/）" % ", ".join(fail))
    print("=" * 98)

    if not args.no_summary:
        for m in [s.strip() for s in args.models.split(",") if s.strip()]:
            summarize(args, m)


if __name__ == "__main__":
    main()
