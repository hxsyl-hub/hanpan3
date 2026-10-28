# -*- coding: utf-8 -*-
"""基线对比：不使用预训练权重，统一超参训练多个模型并汇总

《重要说明》要求：相同 batch、相同数据、相同学习率、相同 epoch（50），
在不加载预训练权重的条件下至少对比 vgg16、googlenet、resnet34、resnet50、
mobilenet，以及其它模型。

本脚本强制关闭预训练权重，并对所有模型使用完全一致的设置，确保结果可比。
每个模型通过独立子进程调用 04_训练.py 完成训练，训练结束释放显存，
避免连续训练多个模型时显存累积占用。

用法：
  # 标准基线对比（GPU，默认 5 个必需模型 + 224 输入）
  python 08_基线对比.py --device cuda

  # 追加其它模型一起对比
  python 08_基线对比.py --device cuda --extra resnet18 efficientnet_b0 densenet121

  # 本机 CPU 试跑（少量样本、1 轮，仅确认脚本可用）
  python 08_基线对比.py --device cpu --epochs 1 --subset 200 --batch 8 --img-size 128

  # 只看将要执行的命令，不训练
  python 08_基线对比.py --dry-run

产物：runs\\train\\基线对比结果.md / .csv（含参数量、总体 ACC、宏平均 ACC、
各类别 ACC、训练时长），可与 05_批量训练.py 的结果分开保存。
"""
import argparse
import csv
import importlib.util
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
TRAIN = os.path.join(ROOT, 'scripts', '04_训练.py')
RUNS = os.path.join(ROOT, 'runs', 'train')
DATA = os.path.join(ROOT, 'data', 'dataset')

# 《重要说明》点名要求的 5 个模型
REQUIRED = ['vgg16', 'googlenet', 'resnet34', 'resnet50', 'mobilenet']


def load_train_module():
    """加载 04_训练.py，用于统计模型参数量（不参与训练）"""
    spec = importlib.util.spec_from_file_location('trainer', TRAIN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def count_params(arch, num_classes):
    """构建一次模型统计参数量（在 CPU 上执行，耗时很短）"""
    try:
        tr = load_train_module()
        model = tr.build_model(arch, num_classes, pretrained=False)
        n = sum(p.numel() for p in model.parameters())
        del model
        return n
    except Exception as e:
        print('  [提示] %s 参数量统计失败：%s' % (arch, e))
        return 0


def read_classes():
    d = os.path.join(DATA, 'train')
    return sorted([x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))])


def train_one(arch, args):
    """调用 04_训练.py 训练单个模型"""
    cmd = [PY, '-u', TRAIN,
           '--model', arch,
           '--epochs', str(args.epochs),
           '--batch', str(args.batch),
           '--lr', str(args.lr),
           '--img-size', str(args.img_size),
           '--device', args.device,
           '--workers', str(args.workers),
           '--seed', str(args.seed)]
    if args.subset:
        cmd += ['--subset', str(args.subset)]
    # 基线对比不使用预训练权重，此处不传 --pretrained

    print('=' * 72)
    print('[%s] %s 开始训练（无预训练权重）' % (datetime.now().strftime('%H:%M:%S'), arch))
    print('命令: python 04_训练.py %s' % ' '.join(cmd[3:]))
    print('=' * 72, flush=True)
    if args.dry_run:
        return None
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    if r.returncode != 0:
        print('[失败] %s 返回码 %d' % (arch, r.returncode), flush=True)
        return None
    return dt / 60


def exp_dir(arch, args):
    """与 04_训练.py 的实验目录命名规则一致"""
    return os.path.join(RUNS, '%s_e%d_p0_f0_ce_none' % (arch, args.epochs))


def collect(arch, args, classes):
    d = exp_dir(arch, args)
    csv_path = os.path.join(d, 'metrics.csv')
    if not os.path.exists(csv_path):
        return None
    rows = list(csv.DictReader(open(csv_path, encoding='utf-8-sig')))
    best = max(rows, key=lambda r: float(r['val_acc']))
    out = {'模型': arch,
           '总体ACC': round(float(best['val_acc']), 4),
           '宏平均ACC': round(float(best['macro_acc']), 4),
           '训练ACC': round(float(best['train_acc']), 4),
           '最佳轮': int(best['epoch'])}
    per = {k[4:]: round(float(v), 4) for k, v in best.items() if k.startswith('acc_')}
    out['各类ACC'] = per
    return out


def main():
    ap = argparse.ArgumentParser(description='基线对比（无预训练权重）')
    ap.add_argument('--models', nargs='*', default=REQUIRED, help='参与对比的模型')
    ap.add_argument('--extra', nargs='*', default=[], help='在 --models 之外追加的模型')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4,
                    help='学习率（与 04_训练.py 保持一致；注意会覆盖 04 的默认值）')
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--subset', type=int, default=0, help='仅用于试跑，每类抽样数量')
    ap.add_argument('--out-name', default='基线对比结果')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    models = list(dict.fromkeys(list(args.models) + list(args.extra)))
    classes = read_classes()
    print('基线对比：%d 个模型 | 无预训练权重 | epoch %d | batch %d | lr %g | 输入 %d | 设备 %s'
          % (len(models), args.epochs, args.batch, args.lr, args.img_size, args.device))
    print('类别（%d）: %s' % (len(classes), ', '.join(classes)))
    if args.dry_run:
        for a in models:
            train_one(a, args)
        print('预览结束，未执行训练')
        return

    os.makedirs(RUNS, exist_ok=True)
    results = []
    for a in models:
        minutes = train_one(a, args)
        if minutes is None:
            continue
        m = collect(a, args, classes)
        if m is None:
            print('[警告] %s 未找到指标文件，跳过汇总' % a, flush=True)
            continue
        m['参数量'] = count_params(a, len(classes))
        m['用时分钟'] = round(minutes, 1)
        results.append(m)
        print('[%s] %s 完成：总体 ACC %.4f | 宏平均 %.4f | 用时 %.1f 分钟'
              % (datetime.now().strftime('%H:%M:%S'), a, m['总体ACC'],
                 m['宏平均ACC'], minutes), flush=True)

    if not results:
        print('没有可汇总的结果')
        return

    keys = ['模型', '参数量', '总体ACC', '宏平均ACC', '训练ACC', '最佳轮', '用时分钟'] + \
           ['ACC_' + c for c in classes]
    csv_path = os.path.join(RUNS, args.out_name + '.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in results:
            w.writerow([r.get(k, '') for k in keys[:7]] +
                       [r.get('各类ACC', {}).get(c, '') for c in classes])

    md = ['# 基线对比结果（无预训练权重）', '',
          '生成时间：%s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '',
          '统一设置：epoch %d | batch %d | 学习率 %g | 输入尺寸 %d | 优化器 AdamW | '
          '损失 交叉熵 | 设备 %s | 随机种子 %d'
          % (args.epochs, args.batch, args.lr, args.img_size, args.device, args.seed),
          '', '所有模型均**不加载预训练权重**，数据、批量大小、学习率、训练轮数完全一致。', '',
          '## 总体结果', '',
          '| 模型 | 参数量 | 总体ACC | 宏平均ACC | 训练ACC | 最佳轮 | 用时(分) |',
          '|---|---|---|---|---|---|---|']
    for r in results:
        md.append('| %s | %s | %.4f | %.4f | %.4f | %d | %.1f |'
                  % (r['模型'], format(r.get('参数量', 0), ','), r['总体ACC'],
                     r['宏平均ACC'], r['训练ACC'], r['最佳轮'], r['用时分钟']))
    md += ['', '## 各类别 ACC', '',
           '| 模型 | ' + ' | '.join(classes) + ' |', '|---|' + '---|' * len(classes)]
    for r in results:
        md.append('| %s | %s |' % (r['模型'],
                  ' | '.join('%.3f' % r['各类ACC'].get(c, 0) for c in classes)))
    best = max(results, key=lambda x: x['总体ACC'])
    best_macro = max(results, key=lambda x: x['宏平均ACC'])
    md += ['', '## 结论', '',
           '总体准确率最高：**%s**（%.4f）' % (best['模型'], best['总体ACC']),
           '', '宏平均准确率最高：**%s**（%.4f）' % (best_macro['模型'], best_macro['宏平均ACC']),
           '', '准确率目标 95%%：%s' % ('达成' if best['总体ACC'] >= 0.95 else '未达成')]
    md_path = os.path.join(RUNS, args.out_name + '.md')
    open(md_path, 'w', encoding='utf-8').write('\n'.join(md))

    print('=' * 72)
    print('基线对比完成，共 %d 个模型' % len(results))
    for r in results:
        print('  %-20s 总体 %.4f | 宏平均 %.4f | 参数量 %s'
              % (r['模型'], r['总体ACC'], r['宏平均ACC'], format(r.get('参数量', 0), ',')))
    print('对比表: %s' % md_path)
    print('数据: %s' % csv_path)
    print('=' * 72)


if __name__ == '__main__':
    main()
