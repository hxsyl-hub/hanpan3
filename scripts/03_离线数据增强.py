# -*- coding: utf-8 -*-
"""离线数据增强：对样本量不足的类别补充样本（对应《重要说明》第 6 条）

原则：
  - 仅对训练集增强，验证集保持原始状态，以保证评估结果的真实性
  - 每个类别增强至 --target 张（默认 200），单张原始图像最多派生 --max-factor 张
    （委托方建议增幅为 50%~80%，避免过度复制导致过拟合）
  - 增强算子按《重要说明》列举项实现：直方图/CLAHE、旋转、gamma、翻转、模糊，
    每张随机选取 1~2 种叠加

用法：
  python 03_离线数据增强.py                       # 对小类(<200)增强到 200 张
  python 03_离线数据增强.py --target 300 --max-factor 3
  python 03_离线数据增强.py --classes NG_焊盘暗蚀 NG_偏位 --factor 5   # 指定类别、固定倍数
"""
import argparse
import csv
import os
import random
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, 'data', 'dataset', 'train')
IMG_EXT = {'.bmp', '.jpg', '.jpeg', '.png'}


# ---------- 增强算子（都做成 (img)->img，img 为 BGR ndarray） ----------
def aug_clahe(img):
    """自适应直方图均衡（区域直方图）—— 用于改善部分图像偏暗或局部曝光过度"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def aug_hist(img):
    """全局直方图均衡（逐通道）"""
    out = img.copy()
    for c in range(3):
        out[:, :, c] = cv2.equalizeHist(out[:, :, c])
    return out


def aug_gamma(img, rng):
    """gamma 校正，0.6~1.6 之间随机"""
    g = rng.uniform(0.6, 1.6)
    inv = 1.0 / g
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def aug_rotate(img, rng):
    """小角度旋转（焊点方向敏感，只做 ±15°，並用边缘像素填充）"""
    ang = rng.uniform(-15, 15)
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def aug_flip(img, rng):
    """水平或垂直翻转"""
    return cv2.flip(img, rng.choice([-1, 0, 1]))


def aug_blur(img, rng):
    """轻微模糊（模拟对焦抖动，半径 1~3）"""
    k = rng.choice([1, 2, 3])
    return cv2.GaussianBlur(img, (2 * k + 1, 2 * k + 1), 0)


def aug_bright(img, rng):
    """亮度对比度扰动（应对曝光过度）"""
    alpha = rng.uniform(0.75, 1.25)     # 对比度
    beta = rng.uniform(-25, 25)         # 亮度
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


OPS = {
    'clahe': aug_clahe, 'hist': aug_hist, 'gamma': aug_gamma,
    'rotate': aug_rotate, 'flip': aug_flip, 'blur': aug_blur,
    'bright': aug_bright,
}


def main():
    ap = argparse.ArgumentParser(description='离线数据增强（只增强 train）')
    ap.add_argument('--train', default=TRAIN, help='train 目录')
    ap.add_argument('--target', type=int, default=200, help='每个小类增强到多少张')
    ap.add_argument('--max-factor', type=int, default=3, help='单张原图最多派生几张')
    ap.add_argument('--classes', nargs='*', default=None, help='只增强这些类别（默认：少于 target 的类别）')
    ap.add_argument('--factor', type=int, default=None, help='固定倍数（覆盖 target 逻辑）')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    if not os.path.isdir(args.train):
        sys.exit('train 目录不存在: %s（先跑 02_划分数据集.py）' % args.train)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    classes = sorted(d for d in os.listdir(args.train) if os.path.isdir(os.path.join(args.train, d)))
    before = {c: len([f for f in os.listdir(os.path.join(args.train, c))
                      if os.path.splitext(f)[1].lower() in IMG_EXT]) for c in classes}

    if args.classes:
        targets = [c for c in args.classes if c in classes]
    else:
        targets = [c for c in classes if before[c] < args.target]

    report, total_added = [], 0
    for cls in classes:
        d = os.path.join(args.train, cls)
        files = sorted(f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in IMG_EXT)
        n0 = len(files)
        if cls not in targets:
            report.append({'类别': cls, '增强前': n0, '新增': 0, '增强后': n0, '方式': '样本充足，不增强'})
            continue

        if args.factor:
            per_img = args.factor
        else:
            need = max(0, args.target - n0)
            per_img = min(args.max_factor, max(1, int(np.ceil(need / max(n0, 1)))))

        added = 0
        for f in files:
            # Windows 下 cv2.imread 读不了中文路径，用 imdecode 绕开
            img = cv2.imdecode(np.fromfile(os.path.join(d, f), dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                continue
            base, ext = os.path.splitext(f)
            for k in range(per_img):
                # 每张随机选 1~2 种算子叠加
                names = rng.sample(list(OPS.keys()), rng.choice([1, 2]))
                out = img
                for nm in names:
                    out = OPS[nm](out, rng) if nm in ('gamma', 'rotate', 'flip', 'blur', 'bright') else OPS[nm](out)
                dst = os.path.join(d, '%s_aug%d_%s%s' % (base, k, '_'.join(names), ext))
                # 同理，写中文路径要用 imencode + tofile
                ok, buf = cv2.imencode(ext, out)
                if ok:
                    buf.tofile(dst)
                    added += 1
        total_added += added
        report.append({'类别': cls, '增强前': n0, '新增': added, '增强后': n0 + added,
                       '方式': '每图派生 %d 张（%s）' % (per_img, '、'.join(OPS.keys())[:40])})

    out_csv = os.path.join(os.path.dirname(args.train), '增强统计.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['类别', '增强前', '新增', '增强后', '方式'])
        w.writeheader()
        w.writerows(report)

    print('=' * 64)
    print('离线增强完成（只改了 train，val 未动）')
    print('%-16s %8s %8s %8s' % ('类别', '增强前', '新增', '增强后'))
    for r in report:
        print('%-16s %8d %8d %8d' % (r['类别'], r['增强前'], r['新增'], r['增强后']))
    print('-' * 64)
    print('合计新增 %d 张 → train 总计 %d 张' % (total_added, sum(before.values()) + total_added))
    print('统计表: %s' % out_csv)
    print('=' * 64)


if __name__ == '__main__':
    main()
