# -*- coding: utf-8 -*-
"""焊点数据集划分脚本：按工业标准类别重映射 + 分层划分 train/val

功能：
  ① 读取 scripts/类别映射.json，把原始 7 个文件夹重映射为工业标准类别
     （同时生成 OK/NG 二级标签，工业上先判合格再归因）
  ② 自动排除清洗报告里的待复核图（精确重复/相似重复/质量异常/疑似错分）
  ③ 分层抽样划分 train/val（默认 8:2），保证每个类别都按比例进 val
  ④ 输出 torchvision.datasets.ImageFolder 兼容的目录结构：
       data/dataset/train/<工业类别>/xxx.bmp
       data/dataset/val/<工业类别>/xxx.bmp
  ⑤ 生成 manifest.csv（溯源：原图路径 -> 新路径 -> split -> 原类别 -> 新类别 -> OK/NG）
  ⑥ 生成类别抽样拼图（每类 16 张缩略图），供人工快速确认映射是否正确

用法：
  python 02_划分数据集.py                    # 默认：映射 + 排除问题图 + 8:2 划分
  python 02_划分数据集.py --val-ratio 0.15
  python 02_划分数据集.py --keep-original    # 不重映射，按原始 7 类划分（做对比实验用）
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'raw', '右焊盘')
MAPPING_JSON = os.path.join(ROOT, 'scripts', '类别映射.json')
CLEAN_CSV = os.path.join(ROOT, 'runs', 'clean', '待复核清单.csv')

IMG_EXT = {'.bmp', '.jpg', '.jpeg', '.png'}


def load_mapping():
    with open(MAPPING_JSON, encoding='utf-8') as f:
        m = json.load(f)
    folder2new, folder2conf = {}, {}
    for new_cls, info in m['映射'].items():
        for src in info['来源']:
            folder2new[src] = new_cls
            folder2conf[src] = info.get('confidence', '?')
    return m, folder2new, folder2conf


def load_excluded():
    """清洗报告里标记为待复核的图（按路径精确匹配）"""
    excluded = set()
    if os.path.exists(CLEAN_CSV):
        with open(CLEAN_CSV, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                excluded.add(os.path.normpath(row['路径']))
    return excluded


def contact_sheet(classes_files, out_dir, per_class=16, cell=140):
    """每个类别拼一张 4x4 缩略图，文件名标注在图上，人工核对用"""
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 11)
    except Exception:
        font = ImageFont.load_default()
    os.makedirs(out_dir, exist_ok=True)
    cols = 4
    for cls, files in classes_files.items():
        files = files[:per_class]
        rows = (len(files) + cols - 1) // cols
        sheet = Image.new('RGB', (cols * cell, rows * (cell + 16)), (24, 24, 24))
        d = ImageDraw.Draw(sheet)
        for k, (fp, name) in enumerate(files):
            try:
                im = Image.open(fp).convert('RGB')
                im.thumbnail((cell, cell))
            except Exception:
                continue
            x = (k % cols) * cell
            y = (k // cols) * (cell + 16)
            sheet.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
            d.text((x + 3, y + cell + 2), name[:22], fill=(255, 255, 120), font=font)
        sheet.save(os.path.join(out_dir, '抽样_%s.jpg' % cls), quality=90)


def main():
    ap = argparse.ArgumentParser(description='焊点数据集划分（工业标准类别 + 分层 train/val）')
    ap.add_argument('--val-ratio', type=float, default=0.2, help='验证集比例（默认 0.2）')
    ap.add_argument('--test-ratio', type=float, default=0.0,
                    help='测试集比例（默认 0=不划分测试集，只分 train/val；'
                         '要三分就传 0.1 之类，比例按 train:val:test 从原图切）')
    ap.add_argument('--seed', type=int, default=42, help='随机种子（保证可复现）')
    ap.add_argument('--keep-original', action='store_true', help='不重映射，按原始 7 类划分')
    ap.add_argument('--no-exclude', action='store_true', help='不排除清洗报告标记的问题图')
    ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'dataset'), help='输出目录')
    ap.add_argument('--no-sheet', action='store_true', help='不生成类别抽样拼图')
    args = ap.parse_args()

    if not os.path.isdir(RAW):
        sys.exit('原始数据集不存在: %s' % RAW)

    mapping, folder2new, folder2conf = load_mapping()
    excluded = set() if args.no_exclude else load_excluded()

    # 收集文件：folder -> [(path, filename), ...]
    classes_files = {}
    for folder in sorted(os.listdir(RAW)):
        d = os.path.join(RAW, folder)
        if not os.path.isdir(d):
            continue
        fs = sorted(f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in IMG_EXT)
        classes_files[folder] = [(os.path.join(d, f), f) for f in fs]

    # 抽样拼图（人工确认映射用）
    if not args.no_sheet:
        sheet_dir = os.path.join(ROOT, 'runs', 'clean', '类别抽样拼图')
        contact_sheet(classes_files, sheet_dir)
        print('已生成类别抽样拼图 -> %s' % sheet_dir)

    # 组装清单：原类别 -> 新类别
    items = []           # (src_path, fname, old_cls, new_cls)
    n_excluded = 0
    for folder, files in classes_files.items():
        if args.keep_original:
            new_cls = folder
        elif folder in folder2new:
            new_cls = folder2new[folder]
        else:
            new_cls = 'UNCERTAIN_' + folder   # 映射表里没有的类别
        for p, f in files:
            if os.path.normpath(p) in excluded:
                n_excluded += 1
                continue
            items.append((p, f, folder, new_cls))

    # 分层划分
    rng = random.Random(args.seed)
    by_new = defaultdict(list)
    for it in items:
        by_new[it[3]].append(it)

    train, val, test = [], [], []
    three = args.test_ratio > 0                            # 是否划分测试集
    for new_cls in sorted(by_new):
        lst = by_new[new_cls][:]
        rng.shuffle(lst)
        n = len(lst)
        if not three:                                      # 原逻辑：只分 train / val
            n_val = max(1, round(n * args.val_ratio))      # 小类也至少 1 张进 val
            if n < 5:                                      # 少于 5 张的类：全部进 train 并告警
                print('  [警告] 类别 %s 仅 %d 张，全部放入 train（val 无覆盖）' % (new_cls, n))
                train.extend(lst)
                continue
            val.extend(lst[:n_val])
            train.extend(lst[n_val:])
            continue
        # 三分：先切 test，再切 val，剩下归 train
        if n < 3:
            print('  [警告] 类别 %s 仅 %d 张，全部放入 train（val/test 无覆盖）' % (new_cls, n))
            train.extend(lst)
            continue
        n_test = max(1, round(n * args.test_ratio))
        n_val = max(1, round(n * args.val_ratio))
        if n_test + n_val > n - 1:                         # 至少给 train 留 1 张
            n_test = 1
            n_val = max(1, min(n_val, n - 2))
        test.extend(lst[:n_test])
        val.extend(lst[n_test:n_test + n_val])
        train.extend(lst[n_test + n_val:])

    # 落盘：复制到 ImageFolder 结构
    if os.path.isdir(args.out):
        print('输出目录已存在，先清空: %s' % args.out)
        shutil.rmtree(args.out)
    n_copied = 0
    for split, lst in (('train', train), ('val', val), ('test', test)):
        for p, f, old_cls, new_cls in lst:
            dst_dir = os.path.join(args.out, split, new_cls)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, f)
            n = 1
            while os.path.exists(dst):
                b, e = os.path.splitext(f)
                dst = os.path.join(dst_dir, '%s_%s_%d%s' % (b, old_cls, n, e))
                n += 1
            shutil.copy2(p, dst)
            n_copied += 1

    # manifest 溯源表
    man = os.path.join(args.out, 'manifest.csv')
    with open(man, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['split', '新类别', 'OK/NG', '原类别', '原路径', '文件名', '映射置信度'])
        for split, lst in (('train', train), ('val', val), ('test', test)):
            for p, fname, old_cls, new_cls in lst:
                is_ok = 'OK' if new_cls.startswith('OK') else 'NG'
                conf = folder2conf.get(old_cls, '-')
                w.writerow([split, new_cls, is_ok, old_cls, p, fname, conf])

    # 统计
    def stat(lst):
        c = defaultdict(lambda: defaultdict(int))
        for _, _, old, new in lst:
            c[new][old] += 1
        return c

    ts, vs, es = stat(train), stat(val), stat(test)
    print('=' * 66)
    if three:
        print('划分完成（seed=%d, train:val:test）' % args.seed)
    else:
        print('划分完成（seed=%d, val_ratio=%.2f）' % (args.seed, args.val_ratio))
    print('排除问题图 %d 张（来自清洗报告待复核清单）' % n_excluded)
    if three:
        print('%-16s %6s %6s %6s %6s   %s' % ('工业类别', 'train', 'val', 'test', '合计', '来源(原类别)'))
    else:
        print('%-16s %6s %6s %6s   %s' % ('工业类别', 'train', 'val', '合计', '来源(原类别)'))
    for new_cls in sorted(set(ts) | set(vs) | set(es)):
        t, v, e = ts[new_cls], vs[new_cls], es[new_cls]
        src = ', '.join('%s:%d' % (k, t[k]) for k in sorted(t))
        nt, nv, ne = sum(t.values()), sum(v.values()), sum(e.values())
        if three:
            print('%-16s %6d %6d %6d %6d   %s' % (new_cls, nt, nv, ne, nt + nv + ne, src))
        else:
            print('%-16s %6d %6d %6d   %s' % (new_cls, nt, nv, nt + nv, src))
    if three:
        print('合计 train %d | val %d | test %d | 复制 %d 张'
              % (len(train), len(val), len(test), n_copied))
    else:
        print('合计 train %d | val %d | 复制 %d 张' % (len(train), len(val), n_copied))
    print('输出: %s' % args.out)
    print('溯源表: %s' % man)
    print('生成时间: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('=' * 66)


if __name__ == '__main__':
    main()
