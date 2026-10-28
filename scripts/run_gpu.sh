#!/usr/bin/env bash
# ============================================================
# 焊点检测 - 算力云 GPU 一键运行脚本（Linux）
#
# 本脚本内容与《文档 00》附录 A「下次上云的完整操作清单」完全一致，
# 只是把分步命令串起来执行。想逐步排错时照着附录 A 手动敲即可。
#
# 用法：
#   bash scripts/run_gpu.sh            # 默认：V2 综合改进 + 原配置对照
#   bash scripts/run_gpu.sh check      # 只做环境自检
#   bash scripts/run_gpu.sh v2         # 只跑 V2 综合改进
#
# 覆盖变量示例：
#   EPOCHS=120 BATCH=32 bash scripts/run_gpu.sh
#
# ⚠️ 上传前必须先把本机改好的 04_训练.py / 05_批量训练.py 覆盖到Scripts目录，
#    否则跑的仍是旧代码（早停、标签平滑、等比缩放等功能不会生效）。
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-python}"
MODE="${1:-auto}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-64}"

echo "========== 0. 环境自检 =========="
$PY -c "
import sys, torch, torchvision
print('python      ', sys.version.split()[0])
print('torch       ', torch.__version__)
print('torchvision ', torchvision.__version__)
print('CUDA 可用   ', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print('显卡        ', p.name)
    print('显存        ', round(p.total_memory / 1024**3, 1), 'GB')
"

echo "========== 1. 数据检查 =========="
for d in data/dataset/train data/dataset/val; do
    if [ ! -d "$d" ]; then
        echo "缺少 $d，请先上传 data/dataset"
        exit 1
    fi
    echo "  $d : $(find "$d" -type f | wc -l) 个文件"
done

echo "========== 2. 确认脚本已是新版 =========="
if grep -q "keep_ratio" scripts/04_训练.py 2>/dev/null; then
    echo "  04_训练.py 已含新参数（--keep-ratio 等），OK"
else
    echo "  ⚠️ 04_训练.py 看起来还是旧版，请先覆盖上传本机最新版"
    exit 1
fi

if [ "$MODE" = "check" ]; then
    echo "自检完成，退出。"
    exit 0
fi

if [ "$MODE" = "auto" ] || [ "$MODE" = "v2" ]; then
    echo "========== 3. V2 综合改进（延长轮数 + 早停 + 等比缩放 + 标签平滑）=========="
    $PY scripts/05_批量训练.py --models densenet121 --epochs "$EPOCHS" \
        --batch "$BATCH" --img-size 128 --pretrained --aug online --device cuda \
        --extra "--keep-ratio --patience 15 --label-smoothing 0.1" \
        --out-name V2_综合改进
fi

if [ "$MODE" = "auto" ]; then
    echo "========== 4. 对照组：上一轮原配置复现 =========="
    $PY scripts/05_批量训练.py --models densenet121 --epochs 50 --batch "$BATCH" \
        --img-size 128 --pretrained --aug online --device cuda \
        --out-name V2_原配置复现
fi

echo "========== 完成 =========="
echo "结果见 runs/train/V2_综合改进.md"
echo ""
echo "打包下载： zip -r 云端结果.zip runs/train -x '*.pt'"
