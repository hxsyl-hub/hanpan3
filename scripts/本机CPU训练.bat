@echo off
chcp 65001 >nul
title 焊点分类 - 本机 CPU 训练

REM ============================================================
REM  本机 CPU 训练启动脚本
REM  双击即可运行，日志实时写入 runs\train\训练日志.txt
REM  如需调整模型/轮数，修改下面 --models 和 --epochs 两处即可
REM ============================================================

set PY=D:\miniconda_envs\hanpan\python.exe
set ROOT=D:\python\hanpan
set LOG=%ROOT%\runs\train\训练日志.txt

if not exist "%ROOT%\runs\train" mkdir "%ROOT%\runs\train"

echo 开始时间: %date% %time%
echo 日志文件: %LOG%
echo.

cd /d %ROOT%\scripts

REM ---- 配置区（按需修改）----
REM  --models    模型列表，可选 vgg16 googlenet resnet34 resnet50 mobilenet
REM  --epochs    训练轮数
REM  --batch     批量大小（CPU 内存小就调小）
REM  --img-size  输入尺寸（128 比 224 快约 3 倍，原图仅 140x173）
REM  --device    cpu 或 cuda

"%PY%" -u 05_批量训练.py ^
  --models mobilenet resnet34 ^
  --epochs 50 ^
  --batch 16 ^
  --img-size 128 ^
  --device cpu ^
  --workers 2 > "%LOG%" 2>&1

echo.
echo 结束时间: %date% %time%
echo 结果见: %ROOT%\runs\train\对比结果.md
pause
