# -*- coding: utf-8 -*-
"""焊点数据清洗脚本（按《重要说明.docx》步骤 1~3 编写）

做四件事：
  ① 摸清数据现状：每个类别数量、图像尺寸、亮度、清晰度
  ② 精确去重：MD5 完全相同的文件（同一张图存了多份）
  ③ 相似去重：感知哈希 dHash，汉明距离 <= 阈值 视为重复（老师脚本用的是 pHash+SSIM，
               这里改用 dHash + 向量化比对，7225 张从"几天"缩短到"几十秒"）
  ④ 质量筛查：过曝 / 过暗 / 模糊 / 尺寸异常，并找出"跨类重复"（疑似分类错误）

安全策略：
  默认【只出报告不动画】。确认无误后加 --move 才会把问题图移到 <out>/待复核/ 下。
  原图永远不删，随时可撤回。

用法：
  python 01_数据清洗.py                      # 只体检出报告
  python 01_数据清洗.py --move               # 确认后把问题图移走
  python 01_数据清洗.py --hamming 3          # 相似判定更严格（默认 5）
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import cv2
except ImportError:
    cv2 = None

IMG_EXT = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'}


# ---------- 基础指标计算 ----------
def md5_of_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def dhash(img_gray):
    """差值哈希：缩到 9x8，比较相邻列，得到 64 bit（返回 Python int，便于位运算）"""
    small = np.asarray(img_gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.float32)
    diff = small[:, 1:] > small[:, :-1]
    bits = diff.flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def laplacian_var(arr):
    """清晰度：拉普拉斯方差，越小越模糊"""
    if cv2 is not None:
        try:
            # OpenCV 5.x 不再支持 float32 输入 -> float64 输出，需保持同类型
            return float(cv2.Laplacian(arr, cv2.CV_32F).astype(np.float64).var())
        except cv2.error:
            pass
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = arr.shape
    out = np.zeros((h - 2, w - 2), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            out += k[i, j] * arr[i:i + h - 2, j:j + w - 2]
    return float(out.var())


def scan_images(src):
    """遍历所有类别目录，逐图计算指标"""
    rows = []
    classes = sorted(d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)))
    for cls in classes:
        d = os.path.join(src, cls)
        files = sorted(f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in IMG_EXT)
        print('  扫描 %-6s %5d 张 ...' % (cls, len(files)), flush=True)
        for fn in files:
            p = os.path.join(d, fn)
            rec = {'类别': cls, '文件名': fn, '路径': p,
                   '大小KB': round(os.path.getsize(p) / 1024, 1)}
            try:
                img = Image.open(p)
                img.load()
                gray = img.convert('L')
                arr = np.asarray(gray, dtype=np.float32)
                rec['宽'], rec['高'] = img.size
                rec['通道'] = len(img.getbands())
                rec['亮度均值'] = round(float(arr.mean()), 1)
                rec['对比度std'] = round(float(arr.std()), 1)
                rec['饱和占比%'] = round(float((arr >= 250).mean() * 100), 2)
                rec['清晰度'] = round(laplacian_var(arr), 1)
                rec['dHash'] = '%016x' % dhash(gray)
                rec['MD5'] = md5_of_file(p)
                rec['读取错误'] = ''
            except Exception as e:
                rec.update({'宽': 0, '高': 0, '通道': 0, '亮度均值': -1,
                            '对比度std': -1, '清晰度': -1, 'dHash': '', 'MD5': '',
                            '读取错误': str(e)[:80]})
            rows.append(rec)
    return rows


# ---------- 去重 ----------
def hamming_pairs(hashes, threshold, chunk=512):
    """返回 (i, j) 索引对：hash 汉明距离 <= threshold（只算上三角，不重复）"""
    n = len(hashes)
    if n < 2:
        return []
    a = np.frombuffer(np.array(hashes, dtype='>u8').tobytes(), dtype=np.uint8).reshape(n, 8)
    pairs = []
    use_np_count = hasattr(np, 'bitwise_count')
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        blk = a[s:e][:, None, :] ^ a[None, :, :]          # (m, n, 8)
        if use_np_count:
            dist = np.bitwise_count(blk).sum(axis=2)       # (m, n)
        else:
            dist = np.unpackbits(blk, axis=2).sum(axis=2)
        ii, jj = np.nonzero((dist <= threshold) & (np.arange(n)[None, :] > np.arange(s, e)[:, None]))
        pairs.extend(zip((ii + s).tolist(), jj.tolist()))
    return pairs


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def group_duplicates(rows, threshold):
    """按类别分组做相似去重，返回 {类别: [[组内索引...], ...]}"""
    result = {}
    for cls in sorted({r['类别'] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r['类别'] == cls and r['dHash']]
        hashes = [int(rows[i]['dHash'], 16) for i in idx]
        uf = UnionFind(len(idx))
        for a, b in hamming_pairs(hashes, threshold):
            uf.union(a, b)
        groups = {}
        for k in range(len(idx)):
            groups.setdefault(uf.find(k), []).append(idx[k])
        result[cls] = [g for g in groups.values() if len(g) > 1]
    return result


# ---------- 报告 ----------
def write_csv(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description='焊点数据清洗：体检 + 去重 + 质量筛查')
    ap.add_argument('--src', default=os.path.join(ROOT, 'data', 'raw', '右焊盘'),
                    help='数据集根目录（各类别子目录）')
    ap.add_argument('--out', default=os.path.join(ROOT, 'runs', 'clean'), help='报告输出目录')
    ap.add_argument('--hamming', type=int, default=1,
                    help='相似判定汉明距离阈值（0~64，越小越严格；实测本数据集 >=3 会误合并成巨型组，推荐 1~2）')
    ap.add_argument('--sat', type=float, default=15.0,
                    help='局部过曝阈值：像素饱和(>=250)占比高于此百分比'
                         '（实测本数据集饱和占比中位数仅 3%%，焊点本身有高光，默认 15%% 只挑极端）')
    ap.add_argument('--bright-hi', type=float, default=210.0, help='过曝阈值：亮度均值高于此值')
    ap.add_argument('--bright-lo', type=float, default=35.0, help='过暗阈值：亮度均值低于此值')
    ap.add_argument('--blur', type=float, default=30.0, help='模糊阈值：拉普拉斯方差低于此值')
    ap.add_argument('--move', action='store_true', help='把问题图移到 <out>/待复核/（默认只报告）')
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        sys.exit('数据集目录不存在: %s' % args.src)
    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print('=' * 62)
    print('① 扫描图像，计算指标 ...')
    rows = scan_images(args.src)
    ok = [r for r in rows if not r['读取错误']]
    print('   完成，共 %d 张（读取失败 %d 张）' % (len(rows), len(rows) - len(ok)))

    print('② 精确去重（MD5 完全相同）...')
    md5_map = {}
    for i, r in enumerate(ok):
        md5_map.setdefault(r['MD5'], []).append(i)
    exact_groups = [g for g in md5_map.values() if len(g) > 1]

    print('③ 相似去重（dHash 距离 <= %d）...' % args.hamming)
    sim_groups = group_duplicates(ok, args.hamming)

    print('④ 质量筛查 ...')
    for r in ok:
        f = []
        if r['亮度均值'] > args.bright_hi:
            f.append('过曝')
        if r['饱和占比%'] > args.sat:
            f.append('局部过曝')
        if r['亮度均值'] < args.bright_lo:
            f.append('过暗')
        if r['清晰度'] < args.blur:
            f.append('模糊')
        if min(r['宽'], r['高']) < 32:
            f.append('尺寸过小')
        r['质量标记'] = '+'.join(f)

    # 跨类重复：同一张图出现在两个不同类别 -> 疑似分类错误
    cross = {}
    for g in exact_groups:
        cls_set = {ok[i]['类别'] for i in g}
        if len(cls_set) > 1:
            cross.setdefault(tuple(sorted(cls_set)), []).append([ok[i]['路径'] for i in g])

    # 汇总每个问题图（保留每组第一张，其余标记为待处理）
    dup_keep = set()
    todo = {}   # 索引 -> 原因
    for g in exact_groups:
        dup_keep.add(g[0])
        for i in g[1:]:
            todo.setdefault(i, []).append('精确重复')
    for cls, groups in sim_groups.items():
        for g in groups:
            dup_keep.add(g[0])
            for i in g[1:]:
                if i not in dup_keep:
                    todo.setdefault(i, []).append('相似重复')
    for i, r in enumerate(ok):
        if r['质量标记']:
            todo.setdefault(i, []).append(r['质量标记'])
    for i, r in enumerate(rows):
        if r['读取错误']:
            rows[i]['质量标记'] = '读取失败'

    # ---------- 写报告 ----------
    fields = ['类别', '文件名', '宽', '高', '通道', '大小KB', '亮度均值', '对比度std',
              '饱和占比%', '清晰度', '质量标记', 'MD5', 'dHash', '路径']
    write_csv(os.path.join(args.out, '逐图指标.csv'), ok, fields)

    todo_rows = []
    for i, reasons in sorted(todo.items()):
        r = dict(ok[i])
        r['待处理原因'] = '+'.join(sorted(set(reasons)))
        r['处置'] = '待复核（--move 后会移到 待复核 目录）'
        todo_rows.append(r)
    write_csv(os.path.join(args.out, '待复核清单.csv'), todo_rows,
              ['类别', '文件名', '待处理原因', '处置', '亮度均值', '清晰度', '路径'])

    # 各类别统计
    stats = []
    for cls in sorted({r['类别'] for r in rows}):
        sub = [r for r in ok if r['类别'] == cls]
        if not sub:
            continue
        br = np.array([r['亮度均值'] for r in sub])
        lp = np.array([r['清晰度'] for r in sub])
        sizes = {}
        for r in sub:
            k = '%dx%d' % (r['宽'], r['高'])
            sizes[k] = sizes.get(k, 0) + 1
        top = sorted(sizes.items(), key=lambda x: -x[1])[:3]
        n_exact = sum(len(g) - 1 for g in exact_groups if ok[g[0]]['类别'] == cls)
        n_sim = sum(len(g) - 1 for g in sim_groups.get(cls, []))
        n_q = sum(1 for r in sub if r['质量标记'])
        stats.append({
            '类别': cls, '总数': len(sub),
            '精确重复': n_exact, '相似重复': n_sim, '质量异常': n_q,
            '预计保留': len(sub) - n_exact - n_sim,
            '亮度均值': round(float(br.mean()), 1), '亮度min': round(float(br.min()), 1),
            '亮度max': round(float(br.max()), 1),
            '清晰度中位': round(float(np.median(lp)), 1),
            '常见尺寸': ', '.join('%s(%d)' % (k, v) for k, v in top),
        })

    n_exact = sum(len(g) - 1 for g in exact_groups)
    n_sim = sum(len(g) - 1 for g in sum(sim_groups.values(), []))
    n_q = sum(1 for r in ok if r['质量标记'])
    n_cross = sum(len(v) for v in cross.values())

    md = [
        '# 焊点数据清洗报告',
        '',
        '- 生成时间：%s' % stamp,
        '- 数据集：%s' % args.src,
        '- 相似阈值：dHash 汉明距离 <= %d；过曝 > %.0f；过暗 < %.0f；模糊 < %.0f' % (
            args.hamming, args.bright_hi, args.bright_lo, args.blur),
        '',
        '## 一、数据现状（步骤 1）',
        '',
        '| 类别 | 总数 | 精确重复 | 相似重复 | 质量异常 | 预计保留 | 亮度均值 | 亮度范围 | 清晰度中位 | 常见尺寸 |',
        '|---|---|---|---|---|---|---|---|---|---|',
    ]
    for s in stats:
        md.append('| %(类别)s | %(总数)d | %(精确重复)d | %(相似重复)d | %(质量异常)d | %(预计保留)d '
                  '| %(亮度均值)s | %(亮度min)s~%(亮度max)s | %(清晰度中位)s | %(常见尺寸)s |' % s)
    md += [
        '',
        '**总览**：原始 %d 张 → 精确重复 %d 张、相似重复 %d 张、质量异常 %d 张 → '
        '**预计保留 %d 张**（唯一 %d 张）' % (
            len(ok), n_exact, n_sim, n_q, len(ok) - n_exact - n_sim, len(md5_map)),
        '',
        '## 二、去重结果（步骤 2）',
        '',
        '- 精确重复（MD5 相同）：%d 组，可清理 %d 张' % (len(exact_groups), n_exact),
        '- 相似重复（感知哈希）：%d 组，可清理 %d 张' % (
            sum(len(v) for v in sim_groups.values()), n_sim),
        '',
        '## 三、疑似错分（步骤 3）',
        '',
    ]
    if cross:
        md.append('**同一张图出现在多个类别**（分类明显错误，优先人工确认）：')
        md.append('')
        for k, v in cross.items():
            md.append('- `%s` 之间：%d 组' % (' vs '.join(k), len(v)))
            for g in v[:5]:
                md.append('  - ' + ' | '.join(os.path.basename(p) for p in g))
    else:
        md.append('未发现跨类别的完全重复图（此项需人工抽检确认）。')
    md += [
        '',
        '## 四、质量筛查',
        '',
        '- 过曝（亮度 > %.0f）：%d 张' % (args.bright_hi, sum(1 for r in ok if r['质量标记'].startswith('过曝'))),
        '- 过暗（亮度 < %.0f）：%d 张' % (args.bright_lo, sum(1 for r in ok if '过暗' in r['质量标记'])),
        '- 模糊（清晰度 < %.0f）：%d 张' % (args.blur, sum(1 for r in ok if '模糊' in r['质量标记'])),
        '',
        '> 委托方说明中提到最大困难为**曝光过度**、部分图像偏暗，该项建议重点人工复核。',
        '',
        '## 五、备注：清理了多少',
        '',
        '| 项目 | 数量 |',
        '|---|---|',
        '| 原始图片 | %d |' % len(ok),
        '| 删除/移走 - 精确重复 | %d |' % n_exact,
        '| 删除/移走 - 相似重复 | %d |' % n_sim,
        '| 删除/移走 - 质量异常 | %d |' % n_q,
        '| 疑似错分组数 | %d |' % n_cross,
        '| **清理后剩余** | **%d** |' % (len(ok) - n_exact - n_sim - n_q),
        '',
        '## 六、产物',
        '',
        '- `逐图指标.csv` —— 每张图的尺寸/亮度/清晰度/哈希',
        '- `待复核清单.csv` —— 所有被标记的问题图及原因',
        '- `重复分组.json` —— 每个重复组包含哪些文件',
    ]
    with open(os.path.join(args.out, '清洗报告.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    with open(os.path.join(args.out, '重复分组.json'), 'w', encoding='utf-8') as f:
        json.dump({
            '精确重复': [[ok[i]['路径'] for i in g] for g in exact_groups],
            '相似重复': {c: [[ok[i]['路径'] for i in g] for g in gs]
                      for c, gs in sim_groups.items()},
            '跨类重复': {(' vs '.join(k)): v for k, v in cross.items()},
        }, f, ensure_ascii=False, indent=2)

    # ---------- 可选：移走问题图 ----------
    moved = 0
    if args.move:
        base = os.path.join(args.out, '待复核')
        for i, reasons in todo.items():
            r = ok[i]
            sub = os.path.join(base, '+'.join(sorted(set(reasons))).replace('/', '_'), r['类别'])
            os.makedirs(sub, exist_ok=True)
            dst = os.path.join(sub, r['文件名'])
            n = 1
            while os.path.exists(dst):
                b, e = os.path.splitext(r['文件名'])
                dst = os.path.join(sub, '%s_%d%s' % (b, n, e))
                n += 1
            shutil.move(r['路径'], dst)
            moved += 1
        print('   已移动 %d 张到 %s' % (moved, base))

    print('=' * 62)
    print('原始 %d 张' % len(ok))
    print('  精确重复 %d | 相似重复 %d | 质量异常 %d | 疑似错分组 %d' % (n_exact, n_sim, n_q, n_cross))
    print('  预计保留 %d 张' % (len(ok) - n_exact - n_sim))
    if args.move:
        print('  实际移走 %d 张 -> %s' % (moved, os.path.join(args.out, '待复核')))
    else:
        print('  （本次只出报告，未动原图。确认后加 --move）')
    print('报告目录: %s' % args.out)
    print('=' * 62)


if __name__ == '__main__':
    main()
