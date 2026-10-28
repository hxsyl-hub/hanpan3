# -*- coding: utf-8 -*-
"""测试集评估脚本（三分数据集专用）

背景：04_训练.py 只读取 train 和 val，test 目录在训练过程中完全不被触碰。
因此三分之后，测试集需要一个独立的评估入口 —— 本脚本就是干这个的。

它做的事：
  加载某个实验目录下训练好的 best.pt，在指定 split（默认 test）上跑一次推理，
  输出总体 ACC、宏平均 ACC、每类 ACC（含支持数）与混淆矩阵，并保存报告。

关键点（保证与训练时口径一致）：
  1. 直接复用 04_训练.py 里的 build_model / ResizePad / MEAN / STD，
     不另写一份，避免预处理或模型结构出现偏差
  2. 类别顺序以 best.pt 里记录的为准，而不是 ImageFolder 扫描出来的顺序
  3. 权重用 strict=False 加载，与训练时 --init-weights 的做法一致

用法：
  python 11_测试集评估.py --exp runs/train/googlenet_e50_p0_f0_ce_none
  python 11_测试集评估.py --exp runs/train/模型A,runs/train/模型B     # 多个一起评，最后出对比表
  python 11_测试集评估.py --exp <目录> --split val                    # 也可以评验证集做对照
"""
import argparse
import csv
import importlib.util
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'runs', 'train')


def load_train_module():
    """动态加载 04_训练.py，复用它的 build_model / ResizePad / MEAN / STD。

    文件名含中文和前导数字，不能直接 import，故用 importlib 按路径加载。
    加载时模块的 __name__ 不是 '__main__'，所以 04 的 main() 不会被执行。
    """
    p = os.path.join(ROOT, 'scripts', '04_训练.py')
    if not os.path.isfile(p):
        sys.exit('找不到 %s' % p)
    spec = importlib.util.spec_from_file_location('_train_mod', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluate(T, exp_dir, arch, data_root, split, img_size, batch, device, keep_ratio):
    """评估单个实验，返回结果字典"""
    wpath = os.path.join(exp_dir, 'best.pt')
    if not os.path.isfile(wpath):
        return {'error': '没有 best.pt（实验可能没跑完）', 'exp': exp_dir}

    ckpt = torch.load(wpath, map_location='cpu')
    classes = list(ckpt.get('classes') or [])
    if not classes:
        return {'error': 'best.pt 里没有 classes 字段', 'exp': exp_dir}

    model = T.build_model(arch, len(classes), pretrained=False, freeze=False, attn=None)
    miss, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    model.to(device).eval()

    # 预处理必须与训练时一致
    first = (T.ResizePad(img_size) if keep_ratio
             else transforms.Resize((img_size, img_size)))
    tf = transforms.Compose([first, transforms.ToTensor(),
                             transforms.Normalize(T.MEAN, T.STD)])

    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        return {'error': '数据集里没有 %s 目录' % split, 'exp': exp_dir}
    ds = datasets.ImageFolder(split_dir, transform=tf)

    # 类别顺序对齐：以 best.pt 记录的为准
    if list(ds.classes) != classes:
        idx = [ds.classes.index(c) if c in ds.classes else -1 for c in classes]
        if any(i < 0 for i in idx):
            return {'error': '数据集类别与权重不匹配：缺 %s'
                    % [c for c in classes if c not in ds.classes], 'exp': exp_dir}
        ds.samples = [(p, idx[y]) for p, y in ds.samples]
        ds.classes = classes

    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2,
                    pin_memory=(device == 'cuda'))

    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    with torch.no_grad():
        for x, y in dl:
            out = model(x.to(device))
            if isinstance(out, (tuple, list)):   # googlenet 训练时的 aux 分支
                out = out[0]
            pred = out.argmax(1).cpu().numpy()
            for t, p_ in zip(y.numpy(), pred):
                cm[t, p_] += 1

    total = cm.sum()
    acc = float(np.trace(cm) / total) if total else 0.0
    support = cm.sum(axis=1)
    per = np.array([cm[i, i] / support[i] if support[i] else 0.0 for i in range(n)])
    return {
        'exp': exp_dir, 'arch': arch, 'split': split, 'n': int(total),
        'acc': acc, 'macro': float(per.mean()),
        'per': per, 'support': support, 'cm': cm, 'classes': classes,
        'epoch': ckpt.get('epoch', '-'), 'val_acc': ckpt.get('val_acc', '-'),
        'miss': len(miss), 'unexpected': len(unexpected),
    }


def print_report(r, f=None):
    """打印并可选写入报告文件"""
    out = f or sys.stdout
    print('=' * 70, file=out)
    print('测试集评估：%s' % os.path.basename(os.path.normpath(r['exp'])), file=out)
    print('=' * 70, file=out)
    print('模型 %s | 权重 best.pt（第 %s 轮，当时 val_acc %s）'
          % (r['arch'], r['epoch'], r['val_acc']), file=out)
    print('数据 %s（%d 张）' % (r['split'], r['n']), file=out)
    print('-' * 70, file=out)
    print('%-24s %8s %10s' % ('类别', '支持数', 'ACC'), file=out)
    for i, c in enumerate(r['classes']):
        print('%-24s %8d %10.4f' % (c, r['support'][i], r['per'][i]), file=out)
    print('-' * 70, file=out)
    print('总体 ACC  %.4f' % r['acc'], file=out)
    print('宏平均 ACC %.4f' % r['macro'], file=out)
    print('（权重加载：缺失 %d 个键，多余 %d 个键）' % (r['miss'], r['unexpected']), file=out)
    print(file=out)
    print('混淆矩阵（行=真实类别，列=预测类别）', file=out)
    head = '%-22s' % '真实\\预测' + ''.join('%8s' % c[:7] for c in r['classes'])
    print(head, file=out)
    for i, c in enumerate(r['classes']):
        print('%-22s' % c[:21] + ''.join('%8d' % v for v in r['cm'][i]), file=out)
    print(file=out)


def main():
    ap = argparse.ArgumentParser(description='用 best.pt 在测试集上做最终评估')
    ap.add_argument('--exp', required=True,
                    help='实验目录（含 best.pt）；多个用逗号分隔，最后出对比表')
    ap.add_argument('--model', default=None,
                    help='模型架构；默认从目录名解析（如 googlenet_e50_... → googlenet）')
    ap.add_argument('--split', default='test', help='评估哪个集合（默认 test）')
    ap.add_argument('--data', default=os.path.join(ROOT, 'data', 'dataset'),
                    help='数据集根目录')
    ap.add_argument('--img-size', type=int, default=128)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--device', default='cuda', help='cuda / cpu')
    ap.add_argument('--keep-ratio', action='store_true',
                    help='等比缩放保留长宽比（训练时若加了此参数，这里也要加）')
    ap.add_argument('--out', default=None, help='报告输出目录（默认写到各自实验目录下）')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA 不可用，改用 CPU')
        device = 'cpu'

    T = load_train_module()
    exps = [e.strip() for e in args.exp.split(',') if e.strip()]
    results = []
    for e in exps:
        tag = os.path.basename(os.path.normpath(e))
        arch = args.model or tag.split('_')[0]
        print('评估 %s（架构 %s）...' % (tag, arch))
        r = evaluate(T, e, arch, args.data, args.split, args.img_size,
                     args.batch, device, args.keep_ratio)
        if 'error' in r:
            print('  [跳过] %s' % r['error'])
            continue
        results.append(r)
        out_dir = args.out or e
        os.makedirs(out_dir, exist_ok=True)
        md = os.path.join(out_dir, '%s评估.md' % args.split)
        with open(md, 'w', encoding='utf-8') as f:
            print('生成时间：%s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'), file=f)
            print_report(r, f)
        with open(os.path.join(out_dir, '%s评估.csv' % args.split), 'w',
                  newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['类别', '支持数', 'ACC'])
            for i, c in enumerate(r['classes']):
                w.writerow([c, int(r['support'][i]), '%.4f' % r['per'][i]])
            w.writerow(['总体ACC', r['n'], '%.4f' % r['acc']])
            w.writerow(['宏平均ACC', '', '%.4f' % r['macro']])
        print('  总体 ACC %.4f | 宏平均 %.4f → %s' % (r['acc'], r['macro'], md))

    if len(results) > 1:            # 多实验时给一张横向对比表
        print()
        print('=' * 70)
        print('%s 集对比（按总体 ACC 降序）' % args.split)
        print('=' * 70)
        print('%-34s %10s %10s' % ('实验', '总体 ACC', '宏平均'))
        print('-' * 70)
        for r in sorted(results, key=lambda x: -x['acc']):
            print('%-34s %10.4f %10.4f'
                  % (os.path.basename(os.path.normpath(r['exp']))[:33], r['acc'], r['macro']))
        print('=' * 70)


if __name__ == '__main__':
    main()
