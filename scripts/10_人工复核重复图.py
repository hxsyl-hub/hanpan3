# -*- coding: utf-8 -*-
"""
10_人工复核重复图.py —— 重复/相似图筛查与人工复核

设计思路（吸收了浏览器版审图工具的优点）
------------------------------------------
1. **按组，不按对**：相似的图用并查集聚成组，一组可能 3 张、5 张，
   一次看到全部，而不是两两对比（两两对比会出现"A 像 B、B 像 C、A 不像 C"的混乱）。
2. **分两类，区别对待**：
   - 完全重复：SHA256 相同（字节级一样）→ 不用人看，可自动清除，零风险
   - 相似候选：看起来像但文件不同 → 必须人判断（可能是同一焊点连拍，两张都要留）
3. **决定与执行分离**：先 `--export` 出清单（不动任何文件），人工填好"保留"列，
   再 `--apply` 执行。弹窗审核时每步都写盘，**关窗不丢进度**。
4. **只移不删**：所有"删除"都是移入暂存目录，随时可按原相对路径移回。

用法（推荐顺序）
----------------
    # ① 先摸底：看看有多少组，不动文件
    python 10_人工复核重复图.py --scan-only

    # ② 生成待审核清单（仍然不动文件）
    python 10_人工复核重复图.py --export

    # ③ 完全重复的不用看，自动清掉（每组保留 1 张，零风险）
    python 10_人工复核重复图.py --auto-exact           # 先看统计
    python 10_人工复核重复图.py --auto-exact --yes     # 确认后执行

    # ④ 剩下的"相似候选"人工看（弹窗）
    python 10_人工复核重复图.py --review

    # ⑤ 若第 ④ 步没看完就关了窗，用导出的清单接着审
    python 10_人工复核重复图.py --review --resume runs\\clean\\重复图复核\\待审核清单_xxx.csv

    # ⑥ 按清单执行（也可用 Excel 打开清单，手动填"保留"列后再执行）
    python 10_人工复核重复图.py --apply runs\\clean\\重复图复核\\待审核清单_xxx.csv

弹窗按键
--------
    1~6     移走对应编号的图片（可撤销）
    回车    全部保留，看下一组
    A       自动：保留第 1 张，移走本组其余（对完全重复组很好用）
    Z       撤销上一次移走
    Q       退出（已做的决定都已存盘）

产物
----
    runs\\clean\\重复图复核\\
    ├─ 待审核清单_<时间戳>.csv      审核清单（与浏览器版字段一致，可互相导入）
    ├─ 执行日志_<时间戳>.csv        实际移动的记录
    └─ duplicate_review_deleted\\   被移走的图，按原相对路径存放（可恢复）
"""

import argparse
import csv
import hashlib
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SRC = os.path.join(ROOT, 'data', 'raw', '右焊盘')
DEFAULT_OUT = os.path.join(ROOT, 'runs', 'clean', '重复图复核')

EXTENSIONS = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'}
MAX_DISPLAY = 6          # 弹窗一组最多同时显示几张

# 与浏览器审图工具保持一致的列（便于两边互相导入）
CSV_COLUMNS = ['重复组', '原分类文件夹', '图片相对路径', '重复关系',
               '保留', '审核状态', '文件SHA256', '备注', '精确小组']

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------- 基础工具

def read_image(path):
    """读取图片。用 imdecode + fromfile 以支持中文路径（cv2.imread 在 Windows 不行）。"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def dhash(image):
    """感知哈希：缩到 9x8 灰度后比较相邻像素，得到 64 位。对缩放/压缩不敏感。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).ravel()


class DSU:
    """并查集：把"A 像 B、B 像 C"合并成一组。"""

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


# ---------------------------------------------------------------- 扫描与分组

def scan_images(src):
    """扫描目录，返回 [(Path, sha256, dhash数组), ...]。读一次图，两个哈希都算出来。"""
    items = []
    n_fail = 0
    for path in sorted(Path(src).rglob('*')):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        try:
            image = read_image(path)
            if image is None:
                n_fail += 1
                continue
        except Exception:
            n_fail += 1
            continue
        items.append((path, file_sha256(path), dhash(image)))
    print('已读取 %d 张图片%s' % (len(items),
                                 ('（%d 张无法读取已跳过）' % n_fail) if n_fail else ''),
          flush=True)
    return items


def build_groups(items, max_distance):
    """把图片聚成组。返回 [{'id','kind','members':[(Path, sha)],'exact_id'}]

    kind:
      完全重复          —— 组内所有图 SHA256 相同（字节级一样，可自动清除）
      相似候选（需核对） —— 只是看着像，文件不同（必须人判断）
    """
    if not items:
        return []

    # 1) 按 SHA256 精确分组，得到"字节级完全相同"的集合
    by_sha = {}
    for idx, (path, sha, _) in enumerate(items):
        by_sha.setdefault(sha, []).append(idx)

    # 2) 每个 SHA 只留一张代表做 dHash 比对（完全相同的图不必互相比）
    reps = [g[0] for g in by_sha.values()]
    hashes = np.asarray([items[i][2] for i in reps], dtype=bool)

    print('正在比对 %d 个不同的图片内容……' % len(reps), flush=True)
    dsu = DSU(len(reps))
    for i in range(len(reps) - 1):
        distances = np.count_nonzero(hashes[i + 1:] != hashes[i], axis=1)
        for offset in np.flatnonzero(distances <= max_distance):
            dsu.union(i, i + 1 + int(offset))

    # 3) 展开：一个代表组 → 所有同 SHA 的图片
    clusters = {}
    for r in range(len(reps)):
        clusters.setdefault(dsu.find(r), []).append(r)
    cluster_list = [sorted(v, key=lambda x: str(items[reps[x]][0]))
                    for v in clusters.values() if len(v) > 1 or
                    len(by_sha[items[reps[v[0]]][1]]) > 1]
    cluster_list.sort(key=lambda c: str(items[reps[c[0]]][0]))

    groups, gi, ei = [], 0, 0
    for cluster in cluster_list:
        members = []
        shas = set()
        for r in cluster:
            sha = items[reps[r]][1]
            shas.add(sha)
            for idx in by_sha[sha]:
                members.append((items[idx][0], sha))
        members.sort(key=lambda m: str(m[0]))
        if len(members) < 2:
            continue
        gi += 1
        if len(shas) == 1:          # 全组字节级相同
            ei += 1
            groups.append({'id': 'group_%05d' % gi, 'kind': '完全重复',
                           'members': members, 'exact_id': 'exact_%05d' % ei})
        else:
            groups.append({'id': 'group_%05d' % gi, 'kind': '相似候选（需核对）',
                           'members': members, 'exact_id': ''})
    return groups


def stat_groups(groups, items=None):
    """打印分组统计。"""
    exact = [g for g in groups if g['kind'] == '完全重复']
    fuzzy = [g for g in groups if g['kind'] != '完全重复']
    n_exact_img = sum(len(g['members']) for g in exact)
    n_fuzzy_img = sum(len(g['members']) for g in fuzzy)
    print('=' * 70)
    print('分组统计')
    print('=' * 70)
    print('  完全重复组  %5d 组，涉及 %5d 张   → 整组字节级相同，可自动清除'
          % (len(exact), n_exact_img))
    print('  相似候选组  %5d 组，涉及 %5d 张   → 需人工判断（可能是连拍，都要留）'
          % (len(fuzzy), n_fuzzy_img))
    print()
    # 注意：有些组里"既有完全相同的、又有相似的"，会被整体归为相似候选组。
    # 因此自动清除要按 SHA256 全局统计，不能只看"完全重复组"，否则会漏清。
    if items is not None:
        n_dup = len(items) - len({sha for _, sha, _ in items})
        print('  字节级完全相同的图共 %d 张（含被并入相似候选组的），' % n_dup)
        print('  这些都能自动清除，无需人工判断 → --auto-exact')
    sizes = {}
    for g in groups:
        sizes[len(g['members'])] = sizes.get(len(g['members']), 0) + 1
    print('组大小分布：' + '  '.join('%d张×%d组' % (k, v)
                                     for k, v in sorted(sizes.items())))
    return exact, fuzzy


# ---------------------------------------------------------------- 清单读写

def export_csv(groups, dataset_path, out_dir):
    """导出待审核清单。只写文件，不动图片。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ('待审核清单_%s.csv'
                      % datetime.now().strftime('%Y%m%d_%H%M%S'))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for g in groups:
            for p, sha in g['members']:
                rel = p.relative_to(dataset_path)
                w.writerow([g['id'], rel.parts[0] if len(rel.parts) > 1 else '',
                            str(rel).replace('\\', '/'), g['kind'],
                            '', '待审核', sha, '', g['exact_id']])
    print('待审核清单：%s' % path)
    return path


def load_csv(path):
    """读回清单，返回 (rows, 字段列表)。"""
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    return rows


def save_csv(rows, path):
    """写回清单（审核过程中每步调用，保证关窗不丢进度）。"""
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def pending_groups(rows, only_fuzzy=True):
    """从清单里挑出还没审核的组。返回 [(组号, 关系, [(行下标, 相对路径, sha)])]"""
    by_group = {}
    for i, r in enumerate(rows):
        by_group.setdefault(r['重复组'], []).append(i)
    out = []
    for gid, idxs in by_group.items():
        if len(idxs) < 2:
            continue
        kind = rows[idxs[0]]['重复关系']
        if only_fuzzy and kind == '完全重复':
            continue
        if all(rows[i]['审核状态'] == '已审核' for i in idxs):
            continue
        out.append((gid, kind, [(i, rows[i]['图片相对路径'], rows[i]['文件SHA256'])
                                for i in idxs]))
    return out


# ---------------------------------------------------------------- 弹窗审核

def review_groups(rows, csv_path, dataset_path, recovery_dir):
    """弹窗逐组审核。每做一步就写盘，关窗不丢进度。"""
    groups = pending_groups(rows, only_fuzzy=True)
    print('需要人工判断的组：%d 组' % len(groups))
    if not groups:
        print('没有需要人工判断的组（完全重复组请用 --auto-exact 处理）。')
        return

    recovery_dir = Path(recovery_dir)
    state = {'i': -1, 'cur': None, 'closed': False, 'undo': []}

    fig = None
    axes = None

    def draw(fig, axes, members_img, gkind, gi, total):
        for ax, (idx, rel, sha), img in zip(axes, members_img[1], members_img[0]):
            ax.clear()
            if img is None:
                ax.text(0.5, 0.5, '（已移走）', ha='center', va='center')
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                          interpolation='nearest')
            cls = rel.split('/')[0] if '/' in rel else '-'
            name = rel.split('/')[-1]
            ax.set_title('[%d] %s\n%s' % (idx, cls, name), fontsize=10)
            ax.axis('off')
        fig.suptitle(
            '第 %d / %d 组 | %s | 共 %d 张\n'
            '数字键=移走该张    回车=都留    A=只留第1张    Z=撤销    Q=退出'
            % (gi, total, gkind, len(members_img[1])), fontsize=12)
        fig.canvas.draw_idle()

    def load_pair(pairs):
        """读取一组图片，返回 (images, meta)"""
        imgs, meta = [], []
        for idx, rel, sha in pairs:
            p = dataset_path / rel.replace('/', os.sep)
            meta.append((idx, rel, sha))
            if p.is_file():
                try:
                    imgs.append(read_image(str(p)))
                except Exception:
                    imgs.append(None)
            else:
                imgs.append(None)
        return imgs, meta

    def next_group():
        while True:
            state['i'] += 1
            if state['i'] >= len(groups):
                print('所有组已浏览完毕。')
                fig.suptitle('复核完成 | Q: 退出', fontsize=13)
                fig.canvas.draw_idle()
                return
            gid, kind, meta = groups[state['i']]
            alive = [(i, r, s) for i, r, s in meta
                     if (dataset_path / r.replace('/', os.sep)).is_file()]
            if len(alive) < 2:
                continue
            state['cur'] = (gid, kind, alive)
            imgs, metas = load_pair(alive[:MAX_DISPLAY])
            draw(fig, axes, (imgs, metas), kind, state['i'] + 1, len(groups))
            print('\n[%s] %s  %d 张' % (gid, kind, len(alive)))
            for n, (_, rel, _) in enumerate(metas, 1):
                print('   %d: %s' % (n, rel))
            return

    def mark(idx_list_keep, idx_list_move, action):
        """写入审核结果。keep=保留，move=移走。"""
        for i in idx_list_keep:
            rows[i]['保留'] = '1'
            rows[i]['审核状态'] = '已审核'
            rows[i]['备注'] = action
        for i in idx_list_move:
            rows[i]['保留'] = '0'
            rows[i]['审核状态'] = '已审核'
            rows[i]['备注'] = action
        save_csv(rows, csv_path)

    def do_move(idx, rel):
        p = dataset_path / rel.replace('/', os.sep)
        stored = recovery_dir / rel.replace('/', os.sep)
        try:
            stored.parent.mkdir(parents=True, exist_ok=True)
            if stored.exists():
                raise FileExistsError('暂存已存在：%s' % stored)
            shutil.move(str(p), str(stored))
        except Exception as e:
            print('移走失败：%s' % e)
            return None
        return stored

    def on_key(event):
        if state['closed'] or state['cur'] is None:
            return
        gid, kind, meta = state['cur']
        if event.key in ('q', 'escape'):
            plt.close(fig)
        elif event.key in ('z', 'Z'):
            if not state['undo']:
                print('没有可撤销的操作。')
                return
            stored, orig, idxs = state['undo'][-1]
            if not stored.is_file():
                print('撤销失败：暂存文件不存在')
                return
            shutil.move(str(stored), str(orig))
            state['undo'].pop()
            for i in idxs:
                rows[i]['保留'] = ''
                rows[i]['审核状态'] = '待审核'
                rows[i]['备注'] = ''
            save_csv(rows, csv_path)
            print('已恢复：%s' % orig.name)
            state['i'] -= 1
            next_group()
        elif event.key in ('enter', 'return'):
            mark([i for i, _, _ in meta], [], '人工判断：都保留')
            print('→ 都保留')
            next_group()
        elif event.key in ('a', 'A'):
            first = meta[0][0]
            rest = meta[1:]
            moved = []
            for i, rel, sha in rest:
                st = do_move(i, rel)
                if st:
                    moved.append((st, dataset_path / rel.replace('/', os.sep), [i]))
            mark([first], [i for i, _, _ in rest], '人工判断：只留第1张')
            state['undo'].extend(moved)
            print('→ 只留第 1 张，移走 %d 张' % len(moved))
            next_group()
        elif event.key.isdigit() and 1 <= int(event.key) <= len(meta):
            n = int(event.key) - 1
            i, rel, sha = meta[n]
            st = do_move(i, rel)
            if st:
                state['undo'].append((st, dataset_path / rel.replace('/', os.sep), [i]))
                mark([j for j, _, _ in meta if j != i], [i], '人工判断：移走第%d张' % (n + 1))
                print('→ 已移走：%s（按 Z 可撤销）' % rel.split('/')[-1])
            # 当前组保留在屏幕上，继续看
            imgs, metas = load_pair([m for m in meta][:MAX_DISPLAY])
            draw(fig, axes, (imgs, metas), kind, state['i'] + 1, len(groups))

    total = len(groups)
    fig, axes = plt.subplots(1, MAX_DISPLAY, figsize=(3.2 * MAX_DISPLAY, 5))
    fig.canvas.manager.set_window_title('重复图人工复核')
    manager = fig.canvas.manager
    hid = getattr(manager, 'key_press_handler_id', None)
    if hid is not None:
        fig.canvas.mpl_disconnect(hid)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('close_event', lambda e: state.__setitem__('closed', True))
    next_group()
    plt.tight_layout(rect=[0, 0, 1, 0.84])
    plt.show()

    print('清单已更新：%s' % csv_path)
    print('移走的图在：%s' % recovery_dir)
    print('还没看完？下次：--review --resume %s' % os.path.basename(csv_path))


# ---------------------------------------------------------------- 执行

def auto_exact(groups, dataset_path, out_dir, assume_yes=False):
    """自动移走字节级完全相同的多余图片（每个 SHA256 保留 1 张），零风险。

    注意口径：按 SHA256 **全局**统计，而不是只看"完全重复组"。
    因为有些组里既有完全相同的图、又有相似的图，整组会被归为"相似候选"——
    若只清理完全重复组，就会漏掉这些藏在相似组里的字节级重复。
    """
    by_sha = {}
    for g in groups:
        for p, sha in g['members']:
            by_sha.setdefault(sha, []).append(p)
    todo = []
    for sha, paths in by_sha.items():
        if len(paths) > 1:
            for p in sorted(paths)[1:]:      # 保留排序后的第一张
                todo.append((p, sha))
    n_move = len(todo)
    print('=' * 70)
    print('字节级完全相同（SHA256 相同）统计')
    print('=' * 70)
    print('涉及 %d 个不同的图片内容，每种保留 1 张' % len(
        [s for s, ps in by_sha.items() if len(ps) > 1]))
    print('可移走：%d 张' % n_move)
    if not n_move:
        print('没有字节级完全相同的图片，无需处理。')
        return
    print()
    print('示例（前 5 组）：')
    shown = 0
    for sha, paths in sorted(by_sha.items()):
        if len(paths) <= 1 or shown >= 5:
            continue
        ps = sorted(paths)
        print('  保留 %s，移走 %s' % (ps[0].name, '、'.join(p.name for p in ps[1:])))
        shown += 1
    print()
    if not assume_yes:
        ans = input('确认移走这 %d 张到暂存目录？输入 yes 执行：' % n_move).strip().lower()
        if ans != 'yes':
            print('已取消，未做任何改动。')
            return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = out_dir / ('执行日志_%s.csv' % stamp)
    rec_dir = out_dir / 'duplicate_review_deleted' / ('exact_' + stamp)
    moved = 0
    with log_path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['时间', '操作', 'SHA256', '原图路径', '暂存路径'])
        for p, sha in todo:
            rel = p.relative_to(dataset_path)
            stored = rec_dir / rel
            try:
                stored.parent.mkdir(parents=True, exist_ok=True)
                if stored.exists():
                    raise FileExistsError('暂存已存在：%s' % stored)
                shutil.move(str(p), str(stored))
            except Exception as e:
                w.writerow([datetime.now().isoformat(timespec='seconds'),
                            '失败:%s' % e, sha, str(p), ''])
                continue
            moved += 1
            w.writerow([datetime.now().isoformat(timespec='seconds'), '移入暂存',
                        sha, str(p), str(stored)])
            f.flush()
    print()
    print('已移走 %d 张（每种内容保留 1 张）。' % moved)
    print('执行日志：%s' % log_path)
    print('暂存目录：%s' % rec_dir)
    print('提示：移走后请重跑 scripts\\09_数据核对.py 同步文档里的张数。')


def apply_csv(csv_path, dataset_path, out_dir):
    """按审核后的清单执行：保留=0 的移入暂存，保留为空（未审核）的不动。"""
    rows = load_csv(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = out_dir / ('执行日志_%s.csv' % stamp)
    rec_dir = out_dir / 'duplicate_review_deleted' / ('apply_' + stamp)

    todo = [r for r in rows if r['保留'] == '0']
    skip = [r for r in rows if r['保留'] == '']
    print('清单共 %d 条：移走 %d 条，保留 %d 条，未审核 %d 条（不动）'
          % (len(rows), len(todo), sum(1 for r in rows if r['保留'] == '1'), len(skip)))
    if not todo:
        print('没有标记为移走的图片。')
        return

    moved = 0
    with log_path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['时间', '操作', 'SHA256', '原图路径', '暂存路径'])
        for r in todo:
            rel = r['图片相对路径']
            p = dataset_path / rel.replace('/', os.sep)
            if not p.is_file():
                w.writerow([datetime.now().isoformat(timespec='seconds'),
                            '跳过:文件不存在', r['文件SHA256'], str(p), ''])
                continue
            stored = rec_dir / rel.replace('/', os.sep)
            try:
                stored.parent.mkdir(parents=True, exist_ok=True)
                if stored.exists():
                    raise FileExistsError('暂存已存在：%s' % stored)
                shutil.move(str(p), str(stored))
            except Exception as e:
                w.writerow([datetime.now().isoformat(timespec='seconds'),
                            '失败:%s' % e, r['文件SHA256'], str(p), ''])
                continue
            moved += 1
            w.writerow([datetime.now().isoformat(timespec='seconds'), '移入暂存',
                        r['文件SHA256'], str(p), str(stored)])
            f.flush()
    print('已移走 %d 张。' % moved)
    print('执行日志：%s' % log_path)
    print('暂存目录：%s' % rec_dir)


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser(
        description='重复/相似图筛查与人工复核（按组、分两类、先出清单再执行）')
    ap.add_argument('--src', default=DEFAULT_SRC, help='要扫描的目录（默认 data\\raw\\右焊盘）')
    ap.add_argument('--out', default=DEFAULT_OUT, help='清单与暂存目录（默认 runs\\clean\\重复图复核）')
    ap.add_argument('--max-distance', type=int, default=2,
                    help='dHash 距离阈值，越小越严格（默认 2）')
    ap.add_argument('--scan-only', action='store_true', help='只统计，不生成清单、不动文件')
    ap.add_argument('--export', action='store_true', help='生成待审核清单 CSV（不动文件）')
    ap.add_argument('--auto-exact', '--auto-md5', dest='auto_exact', action='store_true',
                    help='自动移走完全重复组里多余的图（每组保留 1 张，零风险）')
    ap.add_argument('--yes', '-y', action='store_true', help='配合 --auto-exact，跳过确认')
    ap.add_argument('--review', action='store_true', help='弹窗人工审核相似候选组')
    ap.add_argument('--resume', help='指定要接着审的清单 CSV')
    ap.add_argument('--apply', help='按审核后的清单 CSV 执行移走')
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        print('目录不存在：%s' % args.src)
        sys.exit(1)

    # 审核/执行模式不需要重新扫描
    if args.review or args.apply:
        csv_path = args.resume or (args.apply)
        if not csv_path:
            latest = sorted(Path(args.out).glob('待审核清单_*.csv'))
            if not latest:
                print('未找到清单，请先运行 --export')
                sys.exit(1)
            csv_path = str(latest[-1])
            print('使用最新清单：%s' % csv_path)
        rows = load_csv(csv_path)
        recovery = Path(args.out) / 'duplicate_review_deleted'
        if args.apply:
            apply_csv(csv_path, Path(args.src).resolve(), args.out)
        else:
            review_groups(rows, csv_path, Path(args.src).resolve(), recovery)
        return

    # 统一用绝对路径，避免扫描与导出时路径基准不一致
    dataset = Path(args.src).resolve()
    print('扫描目录：%s' % dataset)
    print('距离阈值：%d' % args.max_distance)
    print()
    items = scan_images(str(dataset))
    if not items:
        print('没找到图片。')
        return
    groups = build_groups(items, args.max_distance)
    print()
    exact, fuzzy = stat_groups(groups, items)
    print()

    if args.scan_only:
        print('（--scan-only 只统计，未生成清单、未改动任何文件）')
        return

    csv_path = export_csv(groups, dataset, args.out)
    print()

    if args.auto_exact:
        auto_exact(groups, dataset, args.out, assume_yes=args.yes)
        return

    if not args.export and not args.review:
        print('下一步：')
        print('  1) 自动清掉完全重复：加 --auto-exact（先看统计，确认后加 --yes）')
        print('  2) 人工审核相似候选：--review')
        print('  3) 也可以先用 Excel 打开清单填"保留"列，再 --apply <清单>')


if __name__ == '__main__':
    main()
