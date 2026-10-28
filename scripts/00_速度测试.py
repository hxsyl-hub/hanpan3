# -*- coding: utf-8 -*-
"""训练速度测试：跑之前先量一下这台机器能有多快，再决定 batch / 图片尺寸 / 跑不跑得起

用法：
  python 00_速度测试.py                    # 测 resnet34 在 128/224 下的速度
  python 00_速度测试.py --models resnet34 mobilenet --sizes 128 160 224
"""
import argparse
import time

import torch
import torch.nn as nn
from torchvision import models


ALIAS = {'mobilenet': 'mobilenet_v3_small'}   # torchvision 里没有 mobilenet 这个属性名


def bench(arch, size, bs, iters=3):
    arch = ALIAS.get(arch, arch)
    m = getattr(models, arch)(weights=None, num_classes=7)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    x, y = torch.randn(bs, 3, size, size), torch.randint(0, 7, (bs,))
    crit = nn.CrossEntropyLoss()
    for _ in range(1):                       # 预热
        opt.zero_grad()
        crit(m(x), y).backward()
        opt.step()
    t = time.time()
    for _ in range(iters):
        opt.zero_grad()
        crit(m(x), y).backward()
        opt.step()
    dt = (time.time() - t) / iters
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=['resnet34'])
    ap.add_argument('--sizes', nargs='*', type=int, default=[128, 224])
    ap.add_argument('--batches', nargs='*', type=int, default=[8, 16])
    ap.add_argument('--train-size', type=int, default=5608, help='训练集张数，用于估算单轮耗时')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    torch.set_num_threads(torch.get_num_threads())
    print('设备: %s | CPU 线程 %d' % (args.device, torch.get_num_threads()))
    print('按 Ctrl+C 可随时中断')
    print('%-12s %5s %5s %12s %12s %14s' % ('模型', '尺寸', 'batch', '秒/批', '张/秒', '1轮/50轮(分)'))
    for arch in args.models:
        for size in args.sizes:
            for bs in args.batches:
                try:
                    dt = bench(arch, size, bs)
                except Exception as e:
                    print('%-12s %5d %5d   失败: %s' % (arch, size, bs, e))
                    continue
                ips = bs / dt
                ep = args.train_size / ips / 60
                print('%-12s %5d %5d %12.2f %12.1f %6.1f / %.0f'
                      % (arch, size, bs, dt, ips, ep, ep * 50), flush=True)


if __name__ == '__main__':
    main()
