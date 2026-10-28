# -*- coding: utf-8 -*-
"""分类错误复核：检出同一张图被放进两个类别的情况，并生成人工复核材料

对应《重要说明》第 3 项：删除分类明显错误的数据，并记录删除数量。

同一张图像同时出现在两个类别中，其中必有一处标注错误。脚本完成三件事：
  1. 列出所有跨类重复组，标明每个成员所属类别、清晰度、亮度
  2. 生成并排对照图（左右两张 + 文件名标注），供人工判断哪个类别正确
  3. 给出建议：保留样本数较少的类别中的那一份（稀有类别样本更宝贵），
     另一份移入 runs\clean\错分移除\，不删除，可恢复

同时报告这些样本当前是否已进入训练集（若已被清洗步骤排除则无需处理）。

用法：
  python 07_错分复核.py               # 生成对照图与清单（不改动数据）
  python 07_错分复核.py --apply       # 按建议移除，并生成处理记录
"""
import argparse
import csv
import json
import os
import shutil
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(ROOT, 'runs', 'clean')
DATASET = os.path.join(ROOT, 'data', 'dataset')
RAW = os.path.join(ROOT, 'data', 'raw', '右焊盘')
OUT = os.path.join(CLEAN, '错分复核')
REMOVE = os.path.join(CLEAN, '错分移除')


def imread_cn(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def metrics_of(p):
    img = imread_cn(p)
    if img is None:
        return 0.0, 0.0
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(g.astype(np.float32), cv2.CV_32F).astype(np.float64).var())
    return sharp, float(g.mean())


def load_groups():
    """跨类重复的结构为 {类别对: [[图A, 图B], ...]}，这里展平成组列表"""
    p = os.path.join(CLEAN, '重复分组.json')
    j = json.load(open(p, encoding='utf-8'))
    g = j.get('跨类重复', {})
    groups = []
    if isinstance(g, dict):
        for pair, items in g.items():
            for it in items:
                if isinstance(it, (list, tuple)) and len(it) >= 2:
                    groups.append(list(it))
    else:
        groups = [x for x in g if isinstance(x, (list, tuple))]
    return groups


def in_dataset(fname, cls=None):
    """检查该文件当前是否已进入训练集或验证集。
    注意：数据集目录使用工业类别名，与原始类别名不同，因此按文件名全局查找。"""
    for split in ('train', 'val'):
        base = os.path.join(DATASET, split)
        if not os.path.isdir(base):
            continue
        for c in os.listdir(base):
            if os.path.exists(os.path.join(base, c, fname)):
                return '%s/%s' % (split, c)
    return ''


def make_pair_image(group, idx, out_dir):
    """生成左右并排对照图"""
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 13)
    except Exception:
        font = ImageFont.load_default()
    ims = []
    for m in group:
        im = imread_cn(m['路径'])
        if im is None:
            return None
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (280, 346), interpolation=cv2.INTER_AREA)
        ims.append(Image.fromarray(im))
    w = 280 * len(ims) + 20 * (len(ims) + 1)
    canvas = Image.new('RGB', (w, 346 + 60), (250, 250, 250))
    d = ImageDraw.Draw(canvas)
    for i, im in enumerate(ims):
        x = 20 + i * 300
        canvas.paste(im, (x, 20))
        m = group[i]
        txt = '%s  %s\n清晰度 %.0f  亮度 %.0f%s' % (
            m['类别'], m['文件名'], m['清晰度'], m['亮度'],
            '\n[建议保留]' if m.get('建议') == '保留' else '\n[建议移除]')
        d.text((x, 366), txt, fill=(20, 100, 20) if m.get('建议') == '保留'
               else (160, 40, 40), font=font)
    path = os.path.join(out_dir, '组%02d_%s.jpg' % (idx, '_vs_'.join(m['类别'] for m in group)))
    canvas.save(path, quality=92)
    return path


def main():
    ap = argparse.ArgumentParser(description='分类错误复核')
    ap.add_argument('--apply', action='store_true', help='按建议移除错误样本（默认只生成材料）')
    args = ap.parse_args()

    groups = load_groups()
    if not groups:
        print('未发现跨类重复组')
        return

    os.makedirs(OUT, exist_ok=True)
    # 各类别样本数，用于给出保留建议
    cnt = {c: len([f for f in os.listdir(os.path.join(RAW, c))
                   if f.lower().endswith(('.bmp', '.jpg', '.png'))])
           for c in os.listdir(RAW) if os.path.isdir(os.path.join(RAW, c))}

    rows, details = [], []
    for gi, g in enumerate(groups, 1):
        members = []
        for p in g:
            cls = os.path.basename(os.path.dirname(p))
            sharp, bright = metrics_of(p)
            members.append({'类别': cls, '文件名': os.path.basename(p), '路径': p,
                            '清晰度': round(sharp, 1), '亮度': round(bright, 1),
                            '该类样本数': cnt.get(cls, 0),
                            '在数据集': in_dataset(os.path.basename(p)) or '未进入'})
        # 建议：保留样本数较少的类别（稀有类更宝贵）
        keep = min(members, key=lambda m: m['该类样本数'])
        for m in members:
            m['建议'] = '保留' if m is keep else '移除'
        img_path = make_pair_image(members, gi, OUT)
        details.append({'组': gi, '对照图': img_path, '成员': members})
        for m in members:
            rows.append({'组': gi, **m})

    csv_path = os.path.join(CLEAN, '疑似错分清单.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_remove = sum(1 for r in rows if r['建议'] == '移除')
    print('=' * 72)
    print('跨类重复（分类明显错误）共 %d 组，涉及 %d 个文件' % (len(groups), len(rows)))
    print('%-4s %-22s %-8s %8s %8s  %-8s %s' % ('组', '文件', '类别', '清晰度', '亮度', '数据集', '建议'))
    for r in rows:
        print('%-4d %-22s %-8s %8.1f %8.1f  %-8s %s'
              % (r['组'], r['文件名'], r['类别'], r['清晰度'], r['亮度'], r['在数据集'], r['建议']))
    print('-' * 72)
    print('建议移除 %d 个文件（每组保留样本数较少类别中的那一份）' % n_remove)
    print('对照图目录: %s' % OUT)
    print('清单: %s' % csv_path)

    if not args.apply:
        print('当前为预览模式，未改动任何数据；人工确认后加 --apply 执行移除')
        print('=' * 72)
        return

    os.makedirs(REMOVE, exist_ok=True)
    moved = 0
    for r in rows:
        if r['建议'] != '移除':
            continue
        dst_dir = os.path.join(REMOVE, r['类别'])
        os.makedirs(dst_dir, exist_ok=True)
        if os.path.exists(r['路径']):
            shutil.move(r['路径'], os.path.join(dst_dir, r['文件名']))
            moved += 1
        # 若该样本已进入训练集/验证集，一并移出
        for split in ('train', 'val'):
            p2 = os.path.join(DATASET, split, r['类别'], r['文件名'])
            if os.path.exists(p2):
                shutil.move(p2, os.path.join(dst_dir, '%s_%s' % (split, r['文件名'])))
                moved += 1

    rec = ['# 分类错误数据处理记录', '',
           '处理时间：%s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '',
           '检出跨类重复 %d 组，涉及 %d 个文件；每组仅保留一处标注，' % (len(groups), len(rows)),
           '保留原则为保留所属类别样本数较少的那一份（稀有类别样本更宝贵）。', '',
           '**共移除 %d 个文件**，移动至 `runs\\clean\\错分移除\\`，未删除，可恢复。' % moved, '',
           '| 组 | 移除的文件 | 原类别 | 保留的文件 | 保留类别 |', '|---|---|---|---|---|']
    for d in details:
        rm = [m for m in d['成员'] if m['建议'] == '移除']
        kp = [m for m in d['成员'] if m['建议'] == '保留']
        for m in rm:
            rec.append('| %d | %s | %s | %s | %s |'
                       % (d['组'], m['文件名'], m['类别'],
                          kp[0]['文件名'] if kp else '-', kp[0]['类别'] if kp else '-'))
    rec += ['', '对照图见 `runs\\clean\\错分复核\\`。']
    rec_path = os.path.join(CLEAN, '错分处理记录.md')
    open(rec_path, 'w', encoding='utf-8').write('\n'.join(rec))
    print('已移除 %d 个文件 -> %s' % (moved, REMOVE))
    print('处理记录: %s' % rec_path)
    print('=' * 72)


if __name__ == '__main__':
    main()
