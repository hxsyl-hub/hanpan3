# -*- coding: utf-8 -*-
"""焊点分类训练脚本：基线对比 + 改进实验（按《重要说明》要求）

项目要求与对应实现：
  - 相同 batch / data / lr / epoch=50，对比 vgg16 / googlenet / resnet34 / resnet50 / mobilenet
  - 先跑【不带预训练权重】的基线 → 对比【带预训练】→ 再对比【冻结 vs 全量】
  - 评价指标：总体 ACC + **每个类别的 ACC**（甲方明确要求）
  - 在线增强：直方图 / 翻转 / 旋转 / 模糊，每张随机选其一
  - 损失函数：普通 CE / class_weights / focal loss
  - 注意力：SE / CA / CBAM（可插到深层）
  - 训练技巧：--init-weights 用上一轮 best.pt 继续训

典型用法：
  # ① 冒烟测试（本机 CPU，20 张 1 epoch，确认脚本能跑）
  python 04_训练.py --model resnet34 --epochs 1 --subset 20 --batch 8

  # ② 基线对比（算力云 GPU，不带预训练权重）
  python 04_训练.py --model resnet34 --epochs 50 --batch 32 --lr 1e-3 --device cuda

  # ③ 带预训练权重对比
  python 04_训练.py --model resnet34 --epochs 50 --pretrained --device cuda

  # ④ 冻结 backbone 只训分类头
  python 04_训练.py --model resnet34 --epochs 50 --pretrained --freeze --device cuda

  # ⑤ 改进：加权 CE + 注意力 + 在线增强
  python 04_训练.py --model resnet34 --epochs 50 --pretrained --loss weighted --attn cbam --aug online --device cuda
"""
import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'dataset')
RUNS = os.path.join(ROOT, 'runs', 'train')

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ================= 注意力模块 =================
class SE(nn.Module):
    """Squeeze-and-Excitation：通道注意力"""
    def __init__(self, c, r=16):
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Linear(c, max(c // r, 4)), nn.ReLU(True),
                                 nn.Linear(max(c // r, 4), c), nn.Sigmoid())

    def forward(self, x):
        return x * self.net(x).unsqueeze(-1).unsqueeze(-1)


class CA(nn.Module):
    """Coordinate Attention：坐标注意力（带位置信息）"""
    def __init__(self, c, r=32):
        super().__init__()
        m = max(c // r, 4)
        self.conv = nn.Conv2d(c, m, 1)
        self.h = nn.Conv2d(m, c, 1)
        self.w = nn.Conv2d(m, c, 1)

    def forward(self, x):
        n, c, hh, ww = x.shape
        xh = F.adaptive_avg_pool2d(x, (hh, 1))
        xw = F.adaptive_avg_pool2d(x, (1, ww))
        y = self.conv(torch.cat([xh, xw.transpose(2, 3)], dim=2)).transpose(2, 3)
        # 转置后拼接维度在 dim=3 上，此处按 dim=3 切分（原先写成 dim=2 会在任何模型上报错）
        yh, yw = torch.split(y, [hh, ww], dim=3)
        return x * self.h(yh.transpose(2, 3)).sigmoid() * self.w(yw).sigmoid()


class CBAM(nn.Module):
    """CBAM：通道注意力 + 空间注意力串联"""
    def __init__(self, c, r=16, k=7):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c, max(c // r, 4)), nn.ReLU(True),
                                 nn.Linear(max(c // r, 4), c))
        self.spa = nn.Conv2d(2, 1, k, padding=k // 2)

    def forward(self, x):
        # 平均池化与最大池化各自过共享 MLP 后相加（不可在通道维拼接，否则维度翻倍）
        a = self.mlp(x.mean((2, 3)))
        m = self.mlp(x.amax((2, 3)))
        x = x * (a + m).sigmoid().unsqueeze(-1).unsqueeze(-1)
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * self.spa(s).sigmoid()


ATTN = {'se': SE, 'ca': CA, 'cbam': CBAM}


# ================= 模型 =================
# 支持的模型；mobilenet 是 mobilenet_v3_small 的简写
MODEL_LIST = [
    'vgg11', 'vgg13', 'vgg16', 'vgg19',
    'googlenet',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnext50_32x4d',
    'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
    'efficientnet_b0', 'efficientnet_b1',
    'densenet121', 'shufflenet_v2_x1_0', 'squeezenet1_0',
    'mnasnet0_5', 'convnext_tiny', 'regnet_x_400mf',
]
MODEL_ALIAS = {'mobilenet': 'mobilenet_v3_small'}


def feat_channels(feat, size=96):
    """用一次空跑探出骨干的真实输出通道数

    mobilenet_v3 等结构的 classifier.in_features 与 features 实际输出通道并不相同，
    直接拿 in_features 建注意力模块会在前向时维度不匹配。
    """
    if feat is None:
        return None
    with torch.no_grad():
        try:
            y = feat(torch.zeros(1, 3, size, size))
        except Exception:
            return None
    return y.shape[1] if y.dim() == 4 else None


def build_model(arch, num_classes, pretrained=False, freeze=False, attn=None):
    arch = MODEL_ALIAS.get(arch, arch)
    if arch not in MODEL_LIST:
        raise ValueError('未知模型: %s（可用: %s）' % (arch, ', '.join(MODEL_LIST)))
    w = 'DEFAULT' if pretrained else None
    m = getattr(models, arch)(weights=w)
    feat, ch, head = None, 0, None

    if arch.startswith('vgg'):
        head = nn.Linear(m.classifier[6].in_features, num_classes)
        m.classifier[6] = head
        feat, ch = m.features, 512
    elif arch == 'googlenet':
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        head = m.fc
        if m.aux1 is not None:
            m.aux1.fc2 = nn.Linear(m.aux1.fc2.in_features, num_classes)
            m.aux2.fc2 = nn.Linear(m.aux2.fc2.in_features, num_classes)
        ch = m.fc.in_features
    elif arch.startswith('resnet') or arch.startswith('resnext'):
        head = nn.Linear(m.fc.in_features, num_classes)
        m.fc = head
        feat, ch = m.layer4, m.fc.in_features
    elif arch.startswith('mobilenet_v3'):
        head = nn.Linear(m.classifier[3].in_features, num_classes)
        m.classifier[3] = head
        feat, ch = m.features, m.classifier[3].in_features
    elif arch in ('mobilenet_v2', 'mnasnet0_5') or arch.startswith('efficientnet'):
        head = nn.Linear(m.classifier[1].in_features, num_classes)
        m.classifier[1] = head
        # mnasnet 的骨干叫 layers，其余叫 features
        feat, ch = getattr(m, 'features', None), m.classifier[1].in_features
    elif arch.startswith('densenet'):
        head = nn.Linear(m.classifier.in_features, num_classes)
        m.classifier = head
        feat, ch = m.features, m.classifier.in_features
    elif arch.startswith('shufflenet'):
        head = nn.Linear(m.fc.in_features, num_classes)
        m.fc = head
        ch = m.fc.in_features
    elif arch.startswith('squeezenet'):
        head = nn.Conv2d(m.classifier[1].in_channels, num_classes, kernel_size=1)
        m.classifier[1] = head
        m.num_classes = num_classes
        feat, ch = m.features, 512
    elif arch.startswith('convnext'):
        head = nn.Linear(m.classifier[2].in_features, num_classes)
        m.classifier[2] = head
        ch = m.classifier[2].in_features
    elif arch.startswith('regnet'):
        head = nn.Linear(m.fc.in_features, num_classes)
        m.fc = head
        ch = m.fc.in_features

    # 注意力：插到深层（resnet 的 layer4 各 block 之后；其余用 Sequential 追加）
    if attn:
        A = ATTN[attn]
        if arch == 'googlenet':
            # googlenet 没有统一的 features 容器，改为包住最后一个 Inception 模块
            blk = m.inception5b
            att = A(ch)

            def make_g(orig, att_mod):
                def f(x):
                    return att_mod(orig(x))
                return f

            blk.forward = make_g(blk.forward, att)
            print('  注意力已插入 googlenet 的 inception5b 之后（通道数 %d）' % ch)
        elif feat is not None and all(hasattr(b, 'bn3') for b in feat):
            for b in feat:
                b.__dict__.setdefault('_attn', A(b.bn3.num_features))
                orig = b.forward

                def make(orig, blk):
                    def f(x):
                        return blk._attn(orig(x))
                    return f
                b.forward = make(orig, b)
        elif feat is not None:
            c_true = feat_channels(feat) or ch
            if c_true != ch:
                print('  注意力通道数按骨干实际输出修正为 %d（分类头为 %d）' % (c_true, ch))
            feat.add_module('attn', A(c_true))
        else:
            print('  [提示] %s 暂不支持自动插注意力，已跳过' % arch)

    if freeze:
        for p in m.parameters():
            p.requires_grad = False
        if head is None:
            raise ValueError('%s 无法确定分类头，无法冻结' % arch)
        for p in head.parameters():
            p.requires_grad = True
        n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print('  冻结 backbone，只训分类头（可训练参数 %s）' % format(n_tr, ','))
    return m


# ================= 损失 =================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.g, self.w, self.ls = gamma, weight, label_smoothing

    def forward(self, logit, y):
        ce = F.cross_entropy(logit, y, weight=self.w, label_smoothing=self.ls,
                             reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.g * ce).mean()


# ================= 混合精度（兼容新旧 torch 写法） =================
def amp_ctx(device):
    try:
        return torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda'))
    except TypeError:
        return torch.cuda.amp.autocast(enabled=(device.type == 'cuda'))


def make_scaler(device):
    try:
        return torch.amp.GradScaler(device_type=device.type, enabled=(device.type == 'cuda'))
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))


# ================= 数据 =================
class ResizePad:
    """等比缩放 + 居中补边，保持原始长宽比（避免强行拉伸导致形变）

    补边颜色取图像自身的平均色，比补黑边更接近真实背景。
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        s = self.size / max(w, h)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        img = img.resize((nw, nh), Image.BICUBIC)
        if nw == self.size and nh == self.size:
            return img
        try:
            fill = tuple(int(v) for v in np.asarray(img).reshape(-1, 3).mean(0))
        except Exception:
            fill = (0, 0, 0)
        canvas = Image.new('RGB', (self.size, self.size), fill)
        canvas.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas


class RandomOneAug:
    """在线增强：每张图随机选取一种算子

    算子共 7 种：直方图均衡 / 翻转 / 旋转 / 模糊 / 亮度 / 对比度 / 平移。
    亮度与对比度抖动对应产线光照波动，平移对应焊点在视野中的位移。
    """

    OPS = ['hist', 'flip', 'rotate', 'blur', 'bright', 'contrast', 'shift']

    def __call__(self, img):
        op = random.choice(self.OPS)
        if op == 'hist':
            return ImageOps.equalize(img)
        if op == 'flip':
            return img.transpose(random.choice(
                [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]))
        if op == 'rotate':
            return img.rotate(random.uniform(-15, 15), resample=Image.BICUBIC)
        if op == 'blur':
            return img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.5)))
        if op == 'bright':
            return ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
        if op == 'contrast':
            return ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
        dx = random.uniform(-0.08, 0.08) * img.size[0]
        dy = random.uniform(-0.08, 0.08) * img.size[1]
        return img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy),
                             resample=Image.BICUBIC)


def make_loaders(args):
    size = args.img_size
    # --keep-ratio：等比缩放保留长宽比（原图 140×173，强行拉成正方形会形变约 24%）
    first = ResizePad(size) if args.keep_ratio else transforms.Resize((size, size))
    base = [first, transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)]
    aug = [RandomOneAug()] + base if args.aug == 'online' else base
    t_train = transforms.Compose(aug)
    t_val = transforms.Compose(base)

    tr = datasets.ImageFolder(os.path.join(DATA, 'train'), transform=t_train)
    va = datasets.ImageFolder(os.path.join(DATA, 'val'), transform=t_val)
    classes = tr.classes

    if args.subset:
        rng = random.Random(0)
        def pick(ds, n):
            idx = list(range(len(ds)))
            rng.shuffle(idx)
            return idx[:n]
        tr = Subset(tr, pick(tr, args.subset))
        va = Subset(va, pick(va, max(1, args.subset // 4)))

    nw = args.workers
    tr_dl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=nw,
                       pin_memory=args.device == 'cuda', persistent_workers=nw > 0)
    va_dl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=nw,
                       pin_memory=args.device == 'cuda', persistent_workers=nw > 0)
    return tr_dl, va_dl, classes, tr


def class_weights(ds, num_classes, mode='inverse'):
    """按训练集中各类样本数算权重（样本越少权重越大）"""
    if isinstance(ds, Subset):
        labels = [ds.dataset.targets[i] for i in ds.indices]
    else:
        labels = ds.targets
    cnt = np.bincount(labels, minlength=num_classes).astype(float)
    cnt = np.maximum(cnt, 1)
    if mode == 'inverse':
        w = cnt.sum() / cnt
    else:                                  # sqrt 反向，更温和
        w = np.sqrt(cnt.sum() / cnt)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


# ================= 训练 / 验证 =================
@torch.no_grad()
def evaluate(model, dl, num_classes, device):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    tot_loss, n = 0.0, 0
    crit = nn.CrossEntropyLoss()
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        out = model(x)
        logit = out.logits if hasattr(out, 'logits') else out
        tot_loss += crit(logit, y).item() * y.size(0)
        pred = logit.argmax(1)
        n += y.size(0)
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            cm[t, p] += 1
    acc_all = cm.trace() / max(cm.sum(), 1)
    per = np.array([cm[i, i] / max(cm[i].sum(), 1) for i in range(num_classes)])
    return tot_loss / max(n, 1), acc_all, per, cm


def main():
    global DATA          # --data 允许指向别的数据集目录
    ap = argparse.ArgumentParser(description='焊点分类训练')
    ap.add_argument('--model', default='resnet34', choices=MODEL_LIST + ['mobilenet'],
                    help='mobilenet 为 mobilenet_v3_small 的简写')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--pretrained', action='store_true', help='加载 ImageNet 预训练权重')
    ap.add_argument('--freeze', action='store_true', help='冻结 backbone，只训分类头')
    ap.add_argument('--loss', default='ce', choices=['ce', 'weighted', 'focal'],
                    help='ce=普通交叉熵 / weighted=class_weights / focal=focal loss')
    ap.add_argument('--attn', default=None, choices=['se', 'ca', 'cbam'], help='注意力模块')
    ap.add_argument('--aug', default='none', choices=['none', 'online'], help='在线增强')
    ap.add_argument('--init-weights', default=None, help='用某个 best.pt 作为初始权重继续训')
    ap.add_argument('--subset', type=int, default=0, help='只取 N 张做冒烟测试')
    ap.add_argument('--name', default=None, help='实验名（默认按参数自动生成）')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data', default=DATA)
    # ---- 以下为改进实验新增开关，默认关闭，不影响原有命令 ----
    ap.add_argument('--patience', type=int, default=0,
                    help='早停：连续 N 轮验证 ACC 不涨即停止，0 表示不启用')
    ap.add_argument('--label-smoothing', type=float, default=0.0,
                    help='标签平滑，常用 0.1，抑制过度自信')
    ap.add_argument('--weight-decay', type=float, default=0.0001, help='AdamW 权重衰减')
    ap.add_argument('--grad-clip', type=float, default=0.0, help='梯度裁剪阈值，0 表示不裁剪')
    ap.add_argument('--warmup', type=int, default=0, help='学习率预热轮数，微调预训练建议 3~5')
    ap.add_argument('--resume', action='store_true', help='从本实验目录的 last.pt 断点续训')
    ap.add_argument('--loss-mode', default='inverse', choices=['inverse', 'sqrt'],
                    help='类别权重计算方式：inverse 激进 / sqrt 温和')
    ap.add_argument('--focal-gamma', type=float, default=2.0, help='Focal Loss 的 gamma')
    ap.add_argument('--keep-ratio', action='store_true',
                    help='等比缩放 + 补边，保留原始长宽比（默认强行拉伸为正方形）')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    DATA = args.data
    if not os.path.isdir(os.path.join(DATA, 'train')):
        sys.exit('找不到 %s（先跑 02_划分数据集.py / 03_离线数据增强.py）' % os.path.join(DATA, 'train'))

    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    tr_dl, va_dl, classes, tr_ds = make_loaders(args)
    nc = len(classes)

    model = build_model(args.model, nc, args.pretrained, args.freeze, args.attn)
    if args.init_weights:
        # torch >= 2.6 的 torch.load 默认 weights_only=True，这里显式关闭并兼容旧版本
        try:
            sd = torch.load(args.init_weights, map_location='cpu', weights_only=False)
        except TypeError:
            sd = torch.load(args.init_weights, map_location='cpu')
        sd = sd.get('model', sd)
        miss = model.load_state_dict(sd, strict=False)
        print('  从 %s 载入初始权重（不匹配 %d 项）' % (args.init_weights, len(miss.missing_keys)))
    model = model.to(device)

    cw = None
    if args.loss in ('weighted', 'focal'):
        cw = class_weights(tr_ds, nc, args.loss_mode).to(device)
        print('  class_weights(%s):' % args.loss_mode,
              [round(v, 2) for v in cw.cpu().tolist()])

    ls = args.label_smoothing
    if args.loss == 'weighted':
        crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=ls)
    elif args.loss == 'focal':
        crit = FocalLoss(gamma=args.focal_gamma, weight=cw, label_smoothing=ls)
    else:
        crit = nn.CrossEntropyLoss(label_smoothing=ls)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度：可选 warmup 线性升温 + 余弦退火
    if args.warmup > 0:
        w, E = args.warmup, args.epochs

        def lr_fn(e):
            if e < w:
                return (e + 1) / float(w)
            prog = (e - w) / max(1, E - w)
            return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    else:
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = make_scaler(device)

    tag = args.name or '%s_%s_p%d_f%d_%s_%s%s' % (
        args.model, 'e%d' % args.epochs, int(args.pretrained), int(args.freeze),
        args.loss, args.aug, '_' + args.attn if args.attn else '')
    out = os.path.join(RUNS, tag)
    os.makedirs(out, exist_ok=True)

    hist, best = [], 0.0

    # ---- 断点续训 ----
    start_ep = 1
    if args.resume:
        ck = os.path.join(out, 'last.pt')
        if os.path.isfile(ck):
            try:
                sd = torch.load(ck, map_location='cpu', weights_only=False)
            except TypeError:
                sd = torch.load(ck, map_location='cpu')
            model.load_state_dict(sd['model'], strict=False)
            if 'opt' in sd:
                try:
                    opt.load_state_dict(sd['opt'])
                except Exception:
                    pass
            start_ep = int(sd.get('epoch', 0)) + 1
            best = float(sd.get('val_acc', 0.0))
            print('  断点续训：从 last.pt 的第 %d 轮继续（已记录 best=%.4f）' % (start_ep - 1, best))
        else:
            print('  [提示] 未找到 last.pt，按从头训练处理')

    print('=' * 72)
    print('实验: %s' % tag)
    print('模型 %s | 预训练 %s | 冻结 %s | 损失 %s | 增强 %s | 注意力 %s' % (
        args.model, args.pretrained, args.freeze, args.loss, args.aug, args.attn))
    print('batch %d | lr %g | epochs %d | img %d | device %s' % (
        args.batch, args.lr, args.epochs, args.img_size, device.type))
    print('类别 %d 个: %s' % (nc, ', '.join(classes)))
    print('输出: %s' % out)
    print('=' * 72)

    metrics_csv = os.path.join(out, 'metrics.csv')
    t0 = time.time()
    wait = 0                      # 早停计数：连续多少轮没涨
    best_ep = 0                   # 最佳结果出现在第几轮
    for ep in range(start_ep, args.epochs + 1):
        model.train()
        run_loss, run_c, run_n = 0.0, 0, 0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with amp_ctx(device):
                out_ = model(x)
                logit = out_.logits if hasattr(out_, 'logits') else out_
                loss = crit(logit, y)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.grad_clip)
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * y.size(0)
            run_c += (logit.argmax(1) == y).sum().item()
            run_n += y.size(0)
        sch.step()

        vl, va_acc, per, cm = evaluate(model, va_dl, nc, device)
        hist.append({'epoch': ep, 'train_loss': run_loss / max(run_n, 1),
                     'train_acc': run_c / max(run_n, 1), 'val_loss': vl, 'val_acc': va_acc,
                     'macro_acc': per.mean(), **{'acc_' + c: per[i] for i, c in enumerate(classes)}})

        with open(metrics_csv, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
            w.writeheader()
            w.writerows(hist)

        # last.pt 附带优化器与调度器状态，供 --resume 断点续训
        torch.save({'model': model.state_dict(), 'classes': classes, 'epoch': ep,
                    'val_acc': va_acc, 'best': best, 'opt': opt.state_dict(),
                    'sch': sch.state_dict()}, os.path.join(out, 'last.pt'))
        if va_acc > best:
            best = va_acc
            best_ep = ep
            torch.save({'model': model.state_dict(), 'classes': classes, 'epoch': ep,
                        'val_acc': va_acc, 'macro': float(per.mean())},
                       os.path.join(out, 'best.pt'))
            wait = 0
        else:
            wait += 1

        print('ep %3d/%d  loss %.4f  train %.4f | val %.4f  macro %.4f  best %.4f  (%.0fs)' % (
            ep, args.epochs, run_loss / max(run_n, 1), run_c / max(run_n, 1),
            va_acc, per.mean(), best, time.time() - t0))
        if ep % 10 == 0 or ep == args.epochs:
            print('    每类 ACC: ' + ' '.join('%s=%.3f' % (c, per[i]) for i, c in enumerate(classes)))

        if args.patience > 0 and wait >= args.patience:
            print('  早停触发：连续 %d 轮验证 ACC 未提升，在第 %d 轮停止（最佳 %.4f 在第 %d 轮）'
                  % (wait, ep, best, best_ep))
            break

    # 最终报告：载入 best.pt（最佳轮权重）再评估，保证报告数值与 best 一致
    ck_best = os.path.join(out, 'best.pt')
    if os.path.isfile(ck_best):
        try:
            sdb = torch.load(ck_best, map_location='cpu', weights_only=False)
        except TypeError:
            sdb = torch.load(ck_best, map_location='cpu')
        model.load_state_dict(sdb['model'])
        model = model.to(device)
        print('  最终报告基于 best.pt（第 %s 轮，验证 ACC %.4f）'
              % (sdb.get('epoch', '-'), sdb.get('val_acc', best)))
    _, _, per, cm = evaluate(model, va_dl, nc, device)
    with open(os.path.join(out, '混淆矩阵.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['真实\\预测'] + classes)
        for i, c in enumerate(classes):
            w.writerow([c] + cm[i].tolist())
    with open(os.path.join(out, '最终报告.md'), 'w', encoding='utf-8') as f:
        f.write('# %s 训练报告\n\n- 时间：%s\n- 用时：%.0f 秒\n- 总体 ACC：**%.4f**  '
                '宏平均 ACC：%.4f\n- 目标 95%%：%s\n\n## 每类 ACC\n\n| 类别 | ACC | 支持数 |\n|---|---|---|\n'
                % (tag, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), time.time() - t0,
                   best, per.mean(), '达成 ✅' if best >= 0.95 else '未达成 ❌'))
        for i, c in enumerate(classes):
            f.write('| %s | %.4f | %d |\n' % (c, per[i], cm[i].sum()))
        f.write('\n配置：%s\n' % vars(args))

    print('=' * 72)
    print('完成 | 最佳总体 ACC %.4f | 宏平均 %.4f | 用时 %.0f 秒' % (best, per.mean(), time.time() - t0))
    print('每类 ACC: ' + ' '.join('%s=%.3f' % (c, per[i]) for i, c in enumerate(classes)))
    print('产物: %s' % out)
    print('=' * 72)


if __name__ == '__main__':
    main()
