# -*- coding: utf-8 -*-
"""批量训练：依次训练多个模型，自动汇总成对比表

与单独执行 04_训练.py 的区别：
  - 一次配置跑完多个模型，中间不用手动改命令
  - 每个模型用独立子进程训练，训练完释放显存/内存，避免长时间运行累积占用
  - 全部跑完后自动生成对比表（总体 ACC、宏平均 ACC、各类别 ACC、用时）

用法：
  # 本机 CPU：轻量模型 + 128 尺寸（推荐配置）
  python 05_批量训练.py --models mobilenet resnet34 --epochs 50 --batch 16 --img-size 128 --device cpu

  # 算力云 GPU：五个模型完整基线对比
  python 05_批量训练.py --models vgg16 googlenet resnet34 resnet50 mobilenet --epochs 50 --batch 32 --device cuda

  # 快速验证流程（少量样本、1 轮）
  python 05_批量训练.py --models mobilenet --epochs 1 --subset 200 --device cpu
"""
import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
TRAIN = os.path.join(ROOT, 'scripts', '04_训练.py')
RUNS = os.path.join(ROOT, 'runs', 'train')


ALIAS = {'mobilenet': 'mobilenet_v3_small'}


def exp_tag(model, epochs, pretrained, freeze, loss, aug, attn):
    model = ALIAS.get(model, model)
    """与 04_训练.py 中的命名规则保持一致"""
    return '%s_e%d_p%d_f%d_%s_%s%s' % (
        model, epochs, int(pretrained), int(freeze), loss, aug,
        '_' + attn if attn else '')


def run_one(model, args):
    tag = exp_tag(model, args.epochs, args.pretrained, args.freeze,
                  args.loss, args.aug, args.attn)
    cmd = [PY, '-u', TRAIN,
           '--model', model, '--epochs', str(args.epochs),
           '--batch', str(args.batch), '--lr', str(args.lr),
           '--img-size', str(args.img_size), '--device', args.device,
           '--workers', str(args.workers), '--loss', args.loss, '--aug', args.aug]
    if args.pretrained:
        cmd.append('--pretrained')
    if args.freeze:
        cmd.append('--freeze')
    if args.attn:
        cmd += ['--attn', args.attn]
    if args.subset:
        cmd += ['--subset', str(args.subset)]

    # 新增开关透传：--extra "--keep-ratio --label-smoothing 0.1 --patience 10"
    extra = shlex.split(args.extra) if args.extra else []
    if extra:
        cmd += extra
        # 用参数缩写拼成后缀，避免不同配置写进同一个实验目录互相覆盖
        short = []
        for t in extra:
            if t.startswith('--'):
                short.append(t[2:].replace('-', ''))
            elif t.replace('.', '').isdigit():
                short.append(t.replace('.', ''))
        if short:
            tag += '_' + ''.join(short)[:40]
    if args.tag_suffix:
        tag += '_' + args.tag_suffix
    cmd += ['--name', tag]

    print('=' * 70)
    print('[%s] 开始训练 %s' % (datetime.now().strftime('%H:%M:%S'), model))
    print('命令: %s' % ' '.join(cmd[2:]))
    print('=' * 70, flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    if r.returncode != 0:
        print('[失败] %s 返回码 %d' % (model, r.returncode), flush=True)
        return None
    return {'模型': model, '实验': tag, '用时分钟': round(dt / 60, 1)}


def collect(tag, classes):
    """从单次实验产物中读取最终指标"""
    d = os.path.join(RUNS, tag)
    csv_path = os.path.join(d, 'metrics.csv')
    if not os.path.exists(csv_path):
        return None
    rows = list(csv.DictReader(open(csv_path, encoding='utf-8-sig')))
    last = rows[-1]
    out = {'总体ACC': round(float(last['val_acc']), 4),
           '宏平均ACC': round(float(last['macro_acc']), 4),
           '训练ACC': round(float(last['train_acc']), 4),
           '轮数': int(last['epoch'])}
    per = {}
    for k, v in last.items():
        if k.startswith('acc_'):
            per[k[4:]] = round(float(v), 4)
    out['各类ACC'] = per
    rep = os.path.join(d, '最终报告.md')
    if os.path.exists(rep):
        t = open(rep, encoding='utf-8').read()
        m = re.search(r'用时：(\d+) 秒', t)
        if m:
            out['用时分钟'] = round(int(m.group(1)) / 60, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description='批量训练与结果汇总')
    ap.add_argument('--models', nargs='+', default=['mobilenet', 'resnet34'],
                    choices=['vgg11', 'vgg13', 'vgg16', 'vgg19', 'googlenet',
                             'resnet18', 'resnet34', 'resnet50', 'resnet101',
                             'resnext50_32x4d', 'mobilenet_v2', 'mobilenet_v3_small',
                             'mobilenet_v3_large', 'efficientnet_b0', 'efficientnet_b1',
                             'densenet121', 'shufflenet_v2_x1_0', 'squeezenet1_0',
                             'mnasnet0_5', 'convnext_tiny', 'regnet_x_400mf',
                             'mobilenet'])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--img-size', type=int, default=128)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--workers', type=int, default=2, help='CPU 训练建议 2，GPU 建议 4')
    ap.add_argument('--pretrained', action='store_true')
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--loss', default='ce', choices=['ce', 'weighted', 'focal'])
    ap.add_argument('--aug', default='none', choices=['none', 'online'])
    ap.add_argument('--attn', default=None, choices=['se', 'ca', 'cbam'])
    ap.add_argument('--subset', type=int, default=0, help='仅用于快速验证，每类抽样数量')
    ap.add_argument('--out-name', default='对比结果', help='汇总文件名')
    ap.add_argument('--extra', default='',
                    help='透传给 04_训练.py 的附加参数，如 "--keep-ratio --patience 10"')
    ap.add_argument('--tag-suffix', default='', help='实验目录名后缀，便于区分不同配置')
    args = ap.parse_args()

    os.makedirs(RUNS, exist_ok=True)
    classes = None
    results = []
    for m in args.models:
        info = run_one(m, args)
        if info is None:
            continue
        tag = info.pop('实验')
        metrics = collect(tag, classes)
        if metrics is None:
            print('[警告] %s 未找到指标文件，跳过汇总' % tag, flush=True)
            continue
        if classes is None and metrics.get('各类ACC'):
            classes = sorted(metrics['各类ACC'].keys())
        info.update(metrics)
        results.append(info)
        print('[%s] %s 完成：总体 ACC %.4f | 宏平均 %.4f | 用时 %.1f 分钟'
              % (datetime.now().strftime('%H:%M:%S'), m,
                 metrics['总体ACC'], metrics['宏平均ACC'],
                 metrics.get('用时分钟', 0)), flush=True)

    if not results:
        print('没有可汇总的结果')
        return

    keys = ['模型', '总体ACC', '宏平均ACC', '训练ACC', '轮数', '用时分钟'] + \
           ['ACC_' + c for c in (classes or [])]
    csv_path = os.path.join(RUNS, args.out_name + '.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in results:
            w.writerow([r.get(k, '') for k in keys[:6]] +
                       [r.get('各类ACC', {}).get(c, '') for c in (classes or [])])

    md = ['# 模型对比结果', '',
          '生成时间：%s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '',
          '统一设置：epochs %d | batch %d | lr %g | 输入尺寸 %d | 设备 %s | 损失 %s | 增强 %s%s'
          % (args.epochs, args.batch, args.lr, args.img_size, args.device,
             args.loss, args.aug, ' | 注意力 ' + args.attn if args.attn else ''),
          '', '## 总体结果', '',
          '| 模型 | 总体ACC | 宏平均ACC | 训练ACC | 用时(分) |', '|---|---|---|---|---|']
    for r in results:
        md.append('| %s | %.4f | %.4f | %.4f | %.1f |'
                  % (r['模型'], r['总体ACC'], r['宏平均ACC'], r['训练ACC'],
                     r.get('用时分钟', 0)))
    if classes:
        md += ['', '## 各类别 ACC', '',
               '| 模型 | ' + ' | '.join(classes) + ' |',
               '|---|' + '---|' * len(classes)]
        for r in results:
            per = r.get('各类ACC', {})
            md.append('| %s | %s |' % (r['模型'],
                      ' | '.join('%.3f' % per.get(c, 0) for c in classes)))
    best = max(results, key=lambda x: x['总体ACC'])
    md += ['', '## 结论', '',
           '总体准确率最高的模型为 **%s**，总体 ACC %.4f，宏平均 ACC %.4f。'
           % (best['模型'], best['总体ACC'], best['宏平均ACC']),
           '准确率目标 95%%：%s' % ('达成' if best['总体ACC'] >= 0.95 else '未达成')]
    md_path = os.path.join(RUNS, args.out_name + '.md')
    open(md_path, 'w', encoding='utf-8').write('\n'.join(md))

    print('=' * 70)
    print('全部完成，共 %d 个模型' % len(results))
    for r in results:
        print('  %-12s 总体 ACC %.4f  宏平均 %.4f' % (r['模型'], r['总体ACC'], r['宏平均ACC']))
    print('对比表: %s' % md_path)
    print('数据: %s' % csv_path)
    print('=' * 70)


if __name__ == '__main__':
    main()
