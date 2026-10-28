# -*- coding: utf-8 -*-
"""类别均衡：缩小各类别样本量差距，降低过拟合风险

类别数量相差过大时，模型倾向于把样本全部预测为多数类（本数据集多数类 OK 与最小类
相差约 40 倍），表现为总体准确率虚高而少数类完全识别不出。本脚本通过两种方式均衡：

  1. 多数类裁减：样本数超过 --max 的类别按分层抽样保留，保留不同清晰度/亮度层次的
     样本，避免直接随机抽样丢掉某一类外观；被裁掉的样本移动到 runs\balance\裁减样本\，
     不删除，随时可恢复
  2. 少数类补足：样本数不足 --min 的类别用离线增强补足，算子复用 03_离线数据增强.py

默认只处理训练集。验证集保持真实分布，用于反映模型在实际产线上的表现；
如需验证集也均衡，加 --balance-val。

用法：
  python 06_类别均衡.py                        # 默认：多数类裁到 800，少数类补到 300
  python 06_类别均衡.py --max 1000 --min 400
  python 06_类别均衡.py --strategy best        # 多数类只保留最清晰的样本
  python 06_类别均衡.py --balance-val          # 验证集一并均衡
  python 06_类别均衡.py --dry-run              # 只看会怎么变，不实际执行
"""
import argparse
import csv
import importlib.util
import os
import random
import shutil
import sys
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'dataset')
BAL = os.path.join(ROOT, 'runs', 'balance')
IMG_EXT = {'.bmp', '.jpg', '.jpeg', '.png'}


def load_aug_ops():
    """复用 03_离线数据增强.py 里已实现的增强算子"""
    p = os.path.join(ROOT, 'scripts', '03_离线数据增强.py')
    spec = importlib.util.spec_from_file_location('aug', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.OPS, ['gamma', 'rotate', 'flip', 'blur', 'bright']


def imread_cn(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def metrics_of(p):
    """计算清晰度与亮度，用于分层抽样"""
    img = imread_cn(p)
    if img is None:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(g.astype(np.float32), cv2.CV_32F).astype(np.float64).var())
    return sharp, float(g.mean())


def stratified_keep(files, keep_n, rng):
    """分层抽样：按清晰度(3档) × 亮度(3档) 分 9 格，格内按比例取样，保留外观多样性"""
    if keep_n >= len(files):
        return files, []
    rec = []
    for f, s, b in files:
        rec.append((f, s, b))
    ss = sorted(r[1] for r in rec)
    bb = sorted(r[2] for r in rec)
    q1, q2 = (np.quantile(ss, 0.33), np.quantile(ss, 0.67))
    b1, b2 = (np.quantile(bb, 0.33), np.quantile(bb, 0.67))
    cells = defaultdict(list)
    for f, s, b in rec:
        si = 0 if s <= q1 else (1 if s <= q2 else 2)
        bi = 0 if b <= b1 else (1 if b <= b2 else 2)
        cells[(si, bi)].append(f)
    kept, dropped = [], []
    for k in sorted(cells):
        lst = cells[k][:]
        rng.shuffle(lst)
        n = max(1, round(keep_n * len(cells[k]) / len(rec)))
        kept += lst[:n]
        dropped += lst[n:]
    rng.shuffle(kept)
    if len(kept) > keep_n:
        dropped += kept[keep_n:]
        kept = kept[:keep_n]
    elif len(kept) < keep_n:                      # 补齐差额
        rng.shuffle(dropped)
        kept += dropped[:keep_n - len(kept)]
        dropped = dropped[keep_n - len(kept):]
    return kept, dropped


def process(split, args, ops, need_rng_ops):
    d = os.path.join(args.data, split)
    if not os.path.isdir(d):
        return []
    rng = random.Random(args.seed)
    rows = []
    cls_dirs = sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))
    stat = {c: [f for f in os.listdir(os.path.join(d, c))
                if os.path.splitext(f)[1].lower() in IMG_EXT] for c in cls_dirs}

    for c in cls_dirs:
        files = stat[c]
        n0 = len(files)
        keep, drop, add = n0, 0, 0
        if n0 > args.max:                                   # 多数类：裁减
            with_m = []
            for f in files:
                m = metrics_of(os.path.join(d, c, f))
                if m is None:
                    continue
                with_m.append((f, m[0], m[1]))
            if args.strategy == 'best':
                with_m.sort(key=lambda x: -x[1])
                kept = [x[0] for x in with_m[:args.max]]
                dropped = [x[0] for x in with_m[args.max:]]
            else:
                kept, dropped = stratified_keep(with_m, args.max, rng)
            drop = len(dropped)
            if not args.dry_run:
                out = os.path.join(BAL, '裁减样本', split, c)
                os.makedirs(out, exist_ok=True)
                for f in dropped:
                    shutil.move(os.path.join(d, c, f), os.path.join(out, f))
            keep = len(kept)
        elif n0 < args.min:                                 # 少数类：补足
            need = args.min - n0
            per = min(args.max_factor, max(1, int(np.ceil(need / max(n0, 1)))))
            if not args.dry_run:
                for f in files:
                    img = imread_cn(os.path.join(d, c, f))
                    if img is None:
                        continue
                    base, ext = os.path.splitext(f)
                    for k in range(per):
                        names = rng.sample(need_rng_ops, rng.choice([1, 2]))
                        out = img
                        for nm in names:
                            fn = ops[nm]
                            out = fn(out, rng) if nm in ('gamma', 'rotate', 'flip',
                                                         'blur', 'bright') else fn(out)
                        dst = os.path.join(d, c, '%s_bal%d_%s%s' % (base, k, '_'.join(names), ext))
                        ok, buf = cv2.imencode(ext, out)
                        if ok:
                            buf.tofile(dst)
                            add += 1
            else:
                add = per * n0
            keep = n0 + add
        rows.append({'划分': split, '类别': c, '处理前': n0, '裁减': drop, '新增': add,
                     '处理后': keep, '操作': ('裁减' if drop else ('增强' if add else '不变'))})
    return rows


def main():
    ap = argparse.ArgumentParser(description='类别均衡（多数类裁减 + 少数类增强）')
    ap.add_argument('--max', type=int, default=800, help='单个类别样本数上限')
    ap.add_argument('--min', type=int, default=300, help='单个类别样本数下限')
    ap.add_argument('--max-factor', type=int, default=4, help='少数类单张原图最多派生几张')
    ap.add_argument('--strategy', default='stratified', choices=['stratified', 'best', 'random'],
                    help='多数类裁减策略：分层抽样 / 保留最清晰 / 随机')
    ap.add_argument('--balance-val', action='store_true', help='验证集一并均衡（默认不动）')
    ap.add_argument('--dry-run', action='store_true', help='只预览，不实际执行')
    ap.add_argument('--data', default=DATA,
                    help='要均衡的数据集目录（默认 data\\dataset；'
                         '与 04_训练.py 的 --data 命名一致，可指向新建的 dataset_v2 等）')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    if args.strategy == 'random':
        args.strategy = 'stratified'      # random 由 stratified 内部按格随机实现

    print('数据集目录：%s' % args.data)
    if not os.path.isdir(args.data):
        print('目录不存在：%s' % args.data)
        sys.exit(1)

    ops, need_rng_ops = load_aug_ops()
    splits = ['train'] + (['val'] if args.balance_val else [])
    rows = []
    for s in splits:
        rows += process(s, args, ops, need_rng_ops)

    out_csv = os.path.join(args.data, '均衡统计.csv')
    if not args.dry_run:
        with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=['划分', '类别', '处理前', '裁减', '新增', '处理后', '操作'])
            w.writeheader()
            w.writerows(rows)

    print('=' * 70)
    print('类别均衡%s' % ('（预览模式，未实际执行）' if args.dry_run else '完成'))
    print('策略：多数类 > %d 裁减（%s），少数类 < %d 增强补足' % (args.max, args.strategy, args.min))
    print('%-22s %8s %8s %8s %8s   %s' % ('划分/类别', '处理前', '裁减', '新增', '处理后', '操作'))
    for r in rows:
        print('%-22s %8d %8d %8d %8d   %s' % (r['划分'] + '/' + r['类别'], r['处理前'],
                                              r['裁减'], r['新增'], r['处理后'], r['操作']))
    tot0 = sum(r['处理前'] for r in rows)
    tot1 = sum(r['处理后'] for r in rows)
    cnt = [r['处理后'] for r in rows if r['划分'] == 'train']
    print('-' * 70)
    print('训练集 %d → %d，类别间最大差距 %d:%.0f:1 → %.1f:1'
          % (sum(r['处理前'] for r in rows if r['划分'] == 'train'), sum(cnt),
             max(r['处理前'] for r in rows if r['划分'] == 'train'),
             max(1, min(r['处理前'] for r in rows if r['划分'] == 'train')),
             max(cnt) / max(1, min(cnt))))
    print('总计 %d → %d' % (tot0, tot1))
    if not args.dry_run:
        print('被裁减样本已移至: %s（未删除）' % os.path.join(BAL, '裁减样本'))
        print('统计表: %s' % out_csv)
    print('=' * 70)


if __name__ == '__main__':
    main()
