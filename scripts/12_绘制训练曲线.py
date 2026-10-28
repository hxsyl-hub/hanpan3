#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""12_绘制训练曲线.py —— 为每个实验生成单独的 ACC 图与 LOSS 图

数据来源
--------
各实验目录下的 metrics.csv（04_训练.py 每轮写入），字段：
    epoch, train_loss, train_acc, val_loss, val_acc, macro_acc, acc_<各类别>...

输出
----
    <实验名>_ACC.png     训练 ACC / 验证 ACC / 宏平均 ACC
    <实验名>_LOSS.png    训练 loss / 验证 loss
    总览_验证ACC对比.png  （--combine）所有实验画在一张图上

用法
----
    python 12_绘制训练曲线.py
    python 12_绘制训练曲线.py --runs runs/train --out runs/train/训练曲线
    python 12_绘制训练曲线.py --combine          # 额外出一张所有实验的对比图
    python 12_绘制训练曲线.py --inplace          # 同时在各自实验目录里存一份

说明
----
- 环境无中文字体时自动改用英文标签，不会出现方块乱码
- 首轮 loss 常常异常大（比如 200+），会自动跳过，避免纵轴被拉爆看不清细节
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')                      # 无图形界面，必须放 import pyplot 之前
import matplotlib.pyplot as plt

from matplotlib import font_manager

CN_FONTS = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Noto Sans CJK SC',
            'WenQuanYi Zen Hei', 'Heiti SC', 'Arial Unicode MS']


def pick_font():
    """返回可用的中文字体名；没有就返回 None（走英文标签）。"""
    avail = set(f.name for f in font_manager.fontManager.ttflist)
    for name in CN_FONTS:
        if name in avail:
            return name
    return None


FONT = pick_font()
CN = FONT is not None

if FONT:                                    # 必须真正应用到 rcParams，否则中文会显示成方块
    plt.rcParams['font.sans-serif'] = [FONT]
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示


def L(zh, en):
    """有中文字体就用中文，否则用英文。"""
    return zh if CN else en


def label(zh, en):
    return zh if CN else en


def read_metrics(path):
    """读 metrics.csv，返回 (epochs, {列名: [值]})。"""
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        return [], {}
    cols = {}
    for k in rows[0]:
        vals = []
        for r in rows:
            v = r.get(k, '')
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(None)
        cols[k] = vals
    ep = [int(float(r['epoch'])) for r in rows if r.get('epoch')]
    return ep, cols


def drop_huge_first(losses, epochs):
    """首轮 loss 异常大时，从第 2 轮开始画。返回 (起始下标, 是否跳过)。"""
    if len(losses) < 2:
        return 0, False
    a, b = losses[0], losses[1]
    if a is None or b is None:
        return 0, False
    if b != 0 and abs(a) > 3 * abs(b) and abs(a) > 5:
        return 1, True
    return 0, False


def plot_acc(exp, epochs, cols, out_png, dpi):
    tr = cols.get('train_acc', [])
    va = cols.get('val_acc', [])
    ma = cols.get('macro_acc', [])
    if not va:
        return False

    fig, ax = plt.subplots(figsize=(9, 5.5))
    if tr:
        ax.plot(epochs[:len(tr)], tr, color='#1f77b4', lw=1.8,
                label=label('训练 ACC', 'Train ACC'))
    ax.plot(epochs[:len(va)], va, color='#d62728', lw=2.0,
            label=label('验证 ACC', 'Val ACC'))
    if ma:
        ax.plot(epochs[:len(ma)], ma, color='#2ca02c', lw=1.6, ls='--',
                label=label('宏平均 ACC', 'Macro ACC'))

    # 标出最佳轮
    best_i, best_v = None, -1.0
    for i, v in enumerate(va):
        if v is not None and v > best_v:
            best_v, best_i = v, i
    if best_i is not None:
        ax.plot(epochs[best_i], best_v, 'o', color='#d62728', ms=7)
        ax.annotate('%s %d: %.4f' % (label('最佳轮', 'Best ep'), epochs[best_i], best_v),
                    xy=(epochs[best_i], best_v),
                    xytext=(8, -18), textcoords='offset points',
                    fontsize=9, color='#d62728')

    ax.set_xlabel(label('轮次', 'Epoch'))
    ax.set_ylabel(label('准确率', 'Accuracy'))
    ax.set_title('%s  %s' % (exp, label('准确率曲线', 'Accuracy Curve')))
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return True


def plot_loss(exp, epochs, cols, out_png, dpi):
    tr = cols.get('train_loss', [])
    va = cols.get('val_loss', [])
    if not tr and not va:
        return False

    start, skipped = drop_huge_first(tr or va, epochs)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    if tr:
        ax.plot(epochs[start:len(tr)], tr[start:], color='#1f77b4', lw=1.8,
                label=label('训练 loss', 'Train loss'))
    if va:
        ax.plot(epochs[start:len(va)], va[start:], color='#ff7f0e', lw=2.0,
                label=label('验证 loss', 'Val loss'))

    ax.set_xlabel(label('轮次', 'Epoch'))
    ax.set_ylabel('Loss')
    title = '%s  %s' % (exp, label('损失曲线', 'Loss Curve'))
    if skipped:
        title += '  (%s)' % label('已跳过异常首轮', 'first epoch skipped')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return True


def find_experiments(runs_dir):
    """找出所有含 metrics.csv 的实验目录。"""
    out = []
    for root, dirs, files in os.walk(runs_dir):
        if 'metrics.csv' in files:
            out.append(root)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description='为每个实验生成单独的 ACC / LOSS 曲线图')
    ap.add_argument('--runs', default=os.path.join('runs', 'train'),
                    help='实验根目录（默认 runs/train）')
    ap.add_argument('--out', default=None,
                    help='图片输出目录（默认 <runs>/训练曲线）')
    ap.add_argument('--combine', action='store_true',
                    help='额外生成一张所有实验的验证 ACC 对比图')
    ap.add_argument('--inplace', action='store_true',
                    help='同时在各自实验目录里也存一份')
    ap.add_argument('--dpi', type=int, default=130, help='图片清晰度（默认 130）')
    args = ap.parse_args()

    runs_dir = os.path.abspath(args.runs)
    if not os.path.isdir(runs_dir):
        sys.exit('实验根目录不存在：%s' % runs_dir)

    out_dir = os.path.abspath(args.out) if args.out else os.path.join(runs_dir, '训练曲线')
    os.makedirs(out_dir, exist_ok=True)

    exps = find_experiments(runs_dir)
    if not exps:
        sys.exit('没找到任何含 metrics.csv 的实验目录')

    print('=' * 66)
    print(label('绘制训练曲线', 'Plot training curves'))
    print('=' * 66)
    print('%s %s' % (label('实验根目录', 'Runs dir '), runs_dir))
    print('%s %s' % (label('输出目录  ', 'Out dir   '), out_dir))
    print('%s %s' % (label('中文字体  ', 'CN font   '), FONT or label('不可用，改用英文标签', 'N/A, using English labels')))
    print('%s %d' % (label('实验数量  ', 'Experiments'), len(exps)))
    print()

    n_ok = 0
    for d in exps:
        exp = os.path.basename(d.rstrip(os.sep))
        csv_path = os.path.join(d, 'metrics.csv')
        try:
            epochs, cols = read_metrics(csv_path)
        except Exception as e:
            print('  [跳过] %s  读取失败：%s' % (exp, e))
            continue
        if not epochs:
            print('  [跳过] %s  metrics.csv 为空' % exp)
            continue

        a = plot_acc(exp, epochs, cols, os.path.join(out_dir, exp + '_ACC.png'), args.dpi)
        b = plot_loss(exp, epochs, cols, os.path.join(out_dir, exp + '_LOSS.png'), args.dpi)
        if args.inplace:
            if a:
                plot_acc(exp, epochs, cols, os.path.join(d, 'ACC曲线.png'), args.dpi)
            if b:
                plot_loss(exp, epochs, cols, os.path.join(d, 'LOSS曲线.png'), args.dpi)
        if a or b:
            n_ok += 1
            print('  %-42s ACC:%s LOSS:%s (%d 轮)' % (
                exp, '√' if a else '-', '√' if b else '-', len(epochs)))

    print()
    print('%s %d/%d' % (label('已生成', 'Generated'), n_ok, len(exps)))

    if args.combine:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for d in exps:
            exp = os.path.basename(d.rstrip(os.sep))
            try:
                epochs, cols = read_metrics(os.path.join(d, 'metrics.csv'))
            except Exception:
                continue
            va = cols.get('val_acc', [])
            if not va:
                continue
            ax.plot(epochs[:len(va)], va, lw=1.5, label=exp)
        ax.set_xlabel(label('轮次', 'Epoch'))
        ax.set_ylabel(label('验证 ACC', 'Val ACC'))
        ax.set_title(label('各实验验证准确率对比', 'Val ACC comparison across experiments'))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        png = os.path.join(out_dir, '总览_验证ACC对比.png')
        fig.savefig(png, dpi=args.dpi)
        plt.close(fig)
        print('%s %s' % (label('对比图', 'Comparison'), png))

    print()
    print('%s %s' % (label('图片目录', 'Images in'), out_dir))


if __name__ == '__main__':
    main()
