#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查FCGR数据尺寸
"""

import numpy as np
import os

def check_fcgr_size(fcgr_path, expected_size=(64, 64)):
    """
    检查FCGR数据的尺寸
    
    参数:
        fcgr_path: fcgr.npz文件路径
        expected_size: 期望的矩阵尺寸，默认(64, 64)
    """
    if not os.path.exists(fcgr_path):
        print(f"错误: 文件不存在 {fcgr_path}")
        return
    
    # 读取npz文件
    data = np.load(fcgr_path, allow_pickle=True)
    fcgr_list = data['fcgr']
    
    print(f"FCGR文件路径: {fcgr_path}")
    print(f"总样本数: {len(fcgr_list)}")
    print(f"期望尺寸: {expected_size}")
    print("-" * 50)
    
    # 统计不同尺寸
    size_count = {}
    for idx, fcgr in enumerate(fcgr_list):
        shape = fcgr.shape
        if shape not in size_count:
            size_count[shape] = 0
        size_count[shape] += 1
        
        # 每1000个样本打印一次
        if idx % 1000 == 0:
            print(f"样本 {idx}: 尺寸 {shape}")
    
    print("-" * 50)
    print("尺寸统计:")
    for shape, count in size_count.items():
        if shape == expected_size:
            print(f"✓ {shape}: {count} 个样本 (符合预期)")
        else:
            print(f"✗ {shape}: {count} 个样本 (不符合预期)")

if __name__ == "__main__":
    # 修改为你的fcgr.npz路径
    fcgr_file = "fcgr_output/train/fcgr.npz"
    check_fcgr_size(fcgr_file)