"""训练集多数类裁减（只裁不补），作用于 splits/train.txt。

为什么单独一个脚本：
  train.py 那一套是按 splits/*.txt 读图的，没有"均衡"这一步。
  而原始分布是 OK 3046 : NG_PadDiscoloration 20 = 152:1，多数类压倒性优势，
  模型会偏向 OK。所以按需求把多数类裁到上限 1500。

原则（与数据处理的口径一致）：
  1. 只裁多数类，不补足少数类 —— 补要靠增强派生，那是后面的实验，基线阶段不做
  2. 只动 train.txt，val / test 一行不改（必须保持真实分布）
  3. 被裁掉的行存到 runs\\balance\\裁减清单\\ 下，随时可以还原

用法：
  python balance_splits.py --max 1500
  python balance_splits.py --max 1500 --dry-run     # 只看数字不落盘
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent
SPLITS = ROOT / "splits"
OUT = ROOT / "runs" / "balance" / "裁减清单"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=1500,
                    help="多数类保留上限（超过就随机裁到这个数）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="只预览，不写文件")
    args = ap.parse_args()

    classes = json.loads((SPLITS / "classes.json").read_text(encoding="utf-8"))
    idx2name = {v: k for k, v in classes.items()}

    rows = [l for l in (SPLITS / "train.txt").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    by_cls = {}
    for r in rows:
        rel, idx = r.split("\t")
        by_cls.setdefault(int(idx), []).append(r)

    rng = random.Random(args.seed)
    kept, removed = [], {}
    print("=" * 62)
    print("训练集多数类裁减%s  上限 = %d" % ("（预览）" if args.dry_run else "", args.max))
    print("=" * 62)
    print("%-24s %9s %9s %9s" % ("类别", "原张数", "裁减", "保留"))
    print("-" * 62)
    for idx in sorted(by_cls):
        lst = by_cls[idx][:]
        rng.shuffle(lst)
        n = len(lst)
        name = idx2name[idx]
        if n > args.max:
            cut = lst[args.max:]
            lst = lst[:args.max]
            removed[name] = cut
        kept.extend(lst)
        print("%-24s %9d %9d %9d" % (name, n, len(removed.get(name, [])), len(lst)))
    print("-" * 62)
    print("%-24s %9d %9d %9d"
          % ("合计", len(rows), sum(len(v) for v in removed.values()), len(kept)))
    mn = min(len(by_cls[i]) for i in by_cls)
    mx = max(min(len(by_cls[i]), args.max) for i in by_cls)
    print("裁后类别间最大差距 %d : %d = %.1f : 1" % (mx, mn, mx / mn))

    if args.dry_run:
        print("\n[预览模式] 未写入任何文件")
        return

    if removed:
        OUT.mkdir(parents=True, exist_ok=True)
        for name, cut in removed.items():
            (OUT / f"train_{name}.txt").write_text("\n".join(cut) + "\n",
                                                   encoding="utf-8")
        print(f"\n被裁清单已保存: {OUT}（未删除原图，可据此还原）")

    kept.sort()
    (SPLITS / "train.txt").write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"train.txt 已更新: {len(rows)} -> {len(kept)} 行")


if __name__ == "__main__":
    main()
