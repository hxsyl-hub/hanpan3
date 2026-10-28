"""09_数据核对.py —— 一键核对文档中的关键数字，保证"有据可查"

设计目的
--------
写文档最容易犯的错误是"凭印象写数字"：早期跑出一版结果，后来脚本重跑、
数据微调，文档却没同步更新，于是出现数字与产物对不上的情况（本项目就
发生过：文档写裁减 2708 张，脚本统计实为 2706 张）。

本脚本直接从**产物文件**（数据集目录、均衡统计.csv、各实验的 metrics.csv）
重新计算所有关键数字，再检查这些数字是否出现在文档中。凡标 ✗ 的项，
说明文档里的数字与产物不一致，必须修正文档。

用法
----
    python scripts/09_数据核对.py                 # 核对全部
    python scripts/09_数据核对.py --doc-only      # 只核对文档数字，不打印明细

输出三张表
----------
1. 数据集核对：各类实际张数（来源是 data/dataset 目录本身）
2. 实验结果核对：每个实验的最佳轮与末轮指标（来源是 runs/train/*/metrics.csv）
3. 文档断言核对：上述实测值是否在文档中出现（来源是 docs/00_项目实施全流程.md）
"""
import argparse
import csv
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'dataset')
RAW = os.path.join(ROOT, 'data', 'raw', '右焊盘')       # 委托方原始图所在位置
RUNS = os.path.join(ROOT, 'runs', 'train')
# 需要核对的文档：同时搜索，避免"某份文档里的数字漏核"
DOCS = [os.path.join(ROOT, 'docs', n) for n in
        ('00_项目实施全流程.md', '01_数据集说明.md', '02_数据处理报告.md')]

IMG_EXT = {'.bmp', '.jpg', '.jpeg', '.png'}


def count_imgs(d):
    """统计一个目录下的图像文件数"""
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d)
                if os.path.splitext(f)[1].lower() in IMG_EXT])


def read_metrics(tag):
    """读取某实验的 metrics.csv，返回 (最佳轮 dict, 末轮 dict)

    实验目录已按阶段分组（runs/train/01_xxx/、02_xxx/），故递归查找。
    """
    hits = glob.glob(os.path.join(RUNS, '**', tag, 'metrics.csv'), recursive=True)
    if not hits:
        return None, None
    rows = list(csv.DictReader(open(hits[0], encoding='utf-8-sig')))
    if not rows:
        return None, None
    key = 'val_acc' if 'val_acc' in rows[0] else list(rows[0])[2]
    best = max(rows, key=lambda r: float(r[key]))
    return best, rows[-1]


def fnum(v, nd=4):
    """统一格式化，便于与文档中的字符串比对"""
    return ('%.' + str(nd) + 'f') % float(v)


def main():
    global DATA                     # --data 允许指向别的数据集目录（如 dataset_v2）
    ap = argparse.ArgumentParser(description='核对文档中的关键数字是否有据可查')
    ap.add_argument('--data', default=DATA,
                    help='要核对的数据集目录（默认 data\\dataset；'
                         '与 02 的 --out、04/06 的 --data 命名一致）')
    ap.add_argument('--doc-only', action='store_true', help='只做文档断言核对')
    args = ap.parse_args()

    DATA = os.path.abspath(args.data)
    if not os.path.isdir(DATA):
        print('数据集目录不存在：%s' % DATA)
        sys.exit(1)
    DS = os.path.relpath(DATA, ROOT)        # 显示用的相对路径

    doc = ''
    doc_names = []
    for p in DOCS:
        if os.path.isfile(p):
            doc += open(p, encoding='utf-8').read()
            doc_names.append(os.path.basename(p))
    src_txt = ' + '.join(doc_names)

    claims = []          # [(名称, 实测值字符串, 来源)]

    # ---------- 0. 原始数据核对 ----------
    # 这一节专门防"总数自洽但明细写错"：本项目曾把 XH3 写成 695（实为 694）、
    # XH2 写成 369（实为 370），两个错误相互抵消，总数仍是 7204，只看总数发现不了。
    if os.path.isdir(RAW):
        print('=' * 74)
        print('零、原始数据核对  来源：data\\raw\\右焊盘\\<类别>\\ 目录实际文件数')
        print('=' * 74)
        raw_tot = 0
        for c in sorted(os.listdir(RAW)):
            n = count_imgs(os.path.join(RAW, c))
            if not n:
                continue
            raw_tot += n
            print('  %-8s %6d' % (c, n))
            # 断言采用"类别 + 数量"的组合形式，避免单个数字在别处出现而误判为通过
            claims.append(('原始 %s 张数' % c, '%s %d' % (c, n), 'data\\raw\\右焊盘\\%s' % c))
        print('-' * 74)
        print('  %-8s %6d' % ('合计', raw_tot))
        claims.append(('原始总张数', str(raw_tot), 'data\\raw\\右焊盘'))
        print()

    # ---------- 1. 数据集核对 ----------
    if not args.doc_only:
        # 兼容两分与三分：有 test 目录就一起核对
        has_test = os.path.isdir(os.path.join(DATA, 'test'))
        splits = ['train', 'val'] + (['test'] if has_test else [])
        CN = {'train': '训练集', 'val': '验证集', 'test': '测试集'}
        print('=' * 74)
        print('一、数据集核对  来源：%s\\{%s}\\<类别>\\ 目录实际文件数'
              % (DS, ','.join(splits)))
        print('=' * 74)
        print('%-24s' % '类别' + ''.join('%10s' % CN[s] for s in splits))
        print('-' * 74)
        tot = dict((s, 0) for s in splits)
        for c in sorted(os.listdir(os.path.join(DATA, 'train'))):
            ns = dict((s, count_imgs(os.path.join(DATA, s, c))) for s in splits)
            for s in splits:
                tot[s] += ns[s]
            print('%-24s' % c + ''.join('%10d' % ns[s] for s in splits))
        print('-' * 74)
        print('%-24s' % '合计' + ''.join('%10d' % tot[s] for s in splits))
        grand = sum(tot.values()) or 1
        print('占比：' + '  '.join('%s %.1f%%' % (CN[s], tot[s] * 100.0 / grand)
                                 for s in splits))
        print()
        claims += [(CN[s] + '总张数', str(tot[s]), '%s\\%s' % (DS, s)) for s in splits]
        claims += [('OK 训练集张数', str(count_imgs(os.path.join(DATA, 'train', 'OK'))),
                    '%s\\train\\OK' % DS),
                   ('NG_Misalignment 验证集',
                    str(count_imgs(os.path.join(DATA, 'val', 'NG_Misalignment'))),
                    '%s\\val\\NG_Misalignment' % DS),
                   ('NG_PadDiscoloration 验证集',
                    str(count_imgs(os.path.join(DATA, 'val', 'NG_PadDiscoloration'))),
                    '%s\\val\\NG_PadDiscoloration' % DS)]
        if has_test:
            claims += [('NG_PadDiscoloration 测试集',
                        str(count_imgs(os.path.join(DATA, 'test', 'NG_PadDiscoloration'))),
                        '%s\\test\\NG_PadDiscoloration' % DS)]

        # 与脚本统计表对账
        bs = os.path.join(DATA, '均衡统计.csv')
        if os.path.isfile(bs):
            print('=' * 74)
            print('二、与脚本统计表对账  来源：%s\\均衡统计.csv' % DS)
            print('=' * 74)
            rows = list(csv.DictReader(open(bs, encoding='utf-8-sig')))
            print('%-12s %8s %8s %8s %8s  %s' %
                  ('类别', '处理前', '裁减', '新增', '处理后', '核对'))
            print('-' * 74)
            ok_all = True
            for r in rows:
                c = r['类别']
                real = count_imgs(os.path.join(DATA, 'train', c))
                same = (real == int(r['处理后']))
                ok_all &= same
                print('%-12s %8s %8s %8s %8s  %s' %
                      (c, r['处理前'], r['裁减'], r['新增'], r['处理后'],
                       '一致' if same else '不一致！实际 %d' % real))
            print('-' * 74)
            print('总判定：%s' % ('全部一致' if ok_all else '存在不一致，见上表'))
            cut = sum(int(r['裁减']) for r in rows)
            print('裁减合计：%d 张' % cut)
            claims.append(('OK 类裁减张数', str(cut), '%s\\均衡统计.csv' % DS))
            print()

    # ---------- 2. 实验结果核对 ----------
    print('=' * 74)
    print('三、实验结果核对  来源：runs\\train\\<分组>\\<实验>\\metrics.csv')
    print('=' * 74)
    print('%-52s %-18s %-18s' % ('实验（分组\\目录）', '最佳轮 val/macro', '末轮 val/macro'))
    print('-' * 74)
    found = []
    for root, dirs, files in os.walk(RUNS):
        if 'metrics.csv' in files:
            found.append(os.path.relpath(root, RUNS))
    for rel in sorted(found):
        tag = os.path.basename(rel)
        b, l = read_metrics(tag)
        if not b:
            continue
        print('%-52s %-18s %-18s' % (
            rel[:52],
            '%s / %s' % (fnum(b['val_acc']), fnum(b.get('macro_acc', 0))),
            '%s / %s' % (fnum(l['val_acc']), fnum(l.get('macro_acc', 0)))))
    print()

    # 关键实验的断言
    KEY = [
        ('densenet121_e50_p1_f0_ce_online', '最优配置 G（末轮）', 'val_acc'),
        ('densenet121_e50_p1_f0_ce_none', 'F：densenet121+预训练（末轮）', 'val_acc'),
        ('googlenet_e50_p1_f0_ce_online', 'E：googlenet+预训练+增强（末轮）', 'val_acc'),
        ('googlenet_e50_p1_f0_ce_none', 'googlenet+预训练（末轮）', 'val_acc'),
        ('googlenet_e50_p0_f0_ce_none', 'googlenet 基线（末轮）', 'val_acc'),
        ('googlenet_e50_p1_f1_ce_none', 'C：冻结骨干（末轮）', 'val_acc'),
    ]
    print('=' * 74)
    print('四、关键实验断言（同时记录最佳轮与末轮，避免口径混淆）')
    print('=' * 74)
    for tag, name, _ in KEY:
        b, l = read_metrics(tag)
        if not b:
            print('%-34s 缺失 metrics.csv' % name)
            continue
        print('%-34s 末轮 %s（宏平均 %s） | 最佳轮 %s 在第 %s 轮' %
              (name, fnum(l['val_acc']), fnum(l.get('macro_acc', 0)),
               fnum(b['val_acc']), b['epoch']))
        claims.append((name + ' 末轮值', fnum(l['val_acc']), 'metrics.csv 最后一行'))
        claims.append((name + ' 最佳轮值', fnum(b['val_acc']), 'metrics.csv 最大值行'))
    print()

    # ---------- 3. 文档断言核对 ----------
    print('=' * 74)
    print('五、文档断言核对：上述实测值是否能在文档中找到')
    print('=' * 74)
    print('来源文档：%s' % src_txt)
    print('-' * 74)
    if not doc:
        print('未找到文档，跳过')
        return
    bad = 0
    for name, val, src in claims:
        hit = val in doc
        bad += 0 if hit else 1
        print('%-28s %-10s %s   来源：%s' %
              (name, val, '✓ 文档中已出现' if hit else '✗ 文档中未出现', src))
    print('-' * 74)
    print('结论：%d 项有据可查，%d 项在文档中查不到（需补写或修正）' %
          (len(claims) - bad, bad))
    if bad:
        print()
        print('处理方式：以产物文件的实测值为准修改文档，不要修改产物。')
        print('（本项目已发生过一次：文档写 2708，实测 2706，最终以 均衡统计.csv 为准。）')
    print('=' * 74)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
    main()
