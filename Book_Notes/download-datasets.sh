#!/bin/bash

# 创建 Datasets 文件夹（如果不存在）
mkdir -p Datasets
cd Datasets

# ✅ 下载 MNIST 数据集 (使用 pytorch 自动下载)
echo "🔄 正在下载 MNIST 数据集..."
if [ ! -d "MNIST" ]; then
    mkdir MNIST
    cd MNIST
    echo "📦 使用 Python 自动下载 MNIST 数据集..."
    /usr/bin/python3 -c "
import torchvision.datasets as datasets
datasets.MNIST(root='.', download=True)
"
    echo "✅ MNIST 数据集下载完成！"
    cd ..
else
    echo "✅ MNIST 数据集已存在，跳过下载。"
fi