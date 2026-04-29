#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNA序列标签生成器

功能：为DNA序列数据集生成标签文件

标签映射规则：
- arch、ba → prokaryote（原核生物）
- e、fungi → eukaryote（真核生物）
- ev → eukaryotic_virus（真核病毒）
- pv → prokaryotic_virus（原核病毒）
- pla → plasmid（质粒）

使用方法：
python generate_dna_labels.py --train_dir TRAIN_DIR --val_dir VAL_DIR --test_dir TEST_DIR

参数说明：
--train_dir: 训练集文件夹路径（包含arch_train.fasta等文件）
--val_dir: 验证集文件夹路径（包含arch_val.fasta等文件）
--test_dir: 测试集文件夹路径（包含arch_test.fasta等文件）

示例：
python generate_dna_labels.py --train_dir "d:/5-class/train" --val_dir "d:/5-class/val" --test_dir "d:/5-class/test"

输出：
- 在每个数据集文件夹中生成labels.txt文件
- 标签顺序与文件夹中所有fasta文件的序列顺序一致
"""

import os
import argparse

# 标签映射字典
LABEL_MAPPING = {
    'arch': 'prokaryote',    # 原核生物
    'ba': 'prokaryote',      # 原核生物
    'e': 'eukaryote',        # 真核生物
    'fungi': 'eukaryote',    # 真核生物
    'ev': 'eukaryotic_virus',  # 真核病毒
    'pv': 'prokaryotic_virus',  # 原核病毒
    'pla': 'plasmid'         # 质粒
}

def extract_sequence_id(header):
    """
    从FASTA头中提取序列标识符
    
    参数：
    header: FASTA序列头（以'>'开头的行）
    
    返回：
    提取的序列标识符（格式：标识符_数值），如 "NC_006396.1_1000_781"
    """
    # 移除开头的'>'并分割行
    parts = header[1:].split()
    
    if not parts:
        return ""
    
    # 提取第一个部分作为序列标识符
    sequence_id = parts[0]
    
    # 查找最后两个数值
    numerical_values = []
    for part in reversed(parts):
        if part.isdigit():
            numerical_values.insert(0, part)
            if len(numerical_values) == 2:
                break
    
    # 组合序列标识符和数值
    if numerical_values:
        return f"{sequence_id}_{'_'.join(numerical_values)}"
    else:
        return sequence_id

def generate_labels_for_directory(directory_path):
    """
    为指定目录中的所有fasta文件生成标签
    
    参数：
    directory_path: 包含fasta文件的目录路径
    
    返回：
    生成的标签列表，每个元素是(sequence_id, label)元组
    """
    if not os.path.isdir(directory_path):
        print(f"错误：目录不存在 - {directory_path}")
        return []
    
    sequence_label_pairs = []
    
    # 获取目录中的所有fasta文件
    files = os.listdir(directory_path)
    fasta_files = [f for f in files if f.endswith('.fasta')]
    
    if not fasta_files:
        print(f"警告：目录中没有fasta文件 - {directory_path}")
        return []
    
    # 按文件名排序，确保标签顺序一致
    fasta_files.sort()
    
    for fasta_file in fasta_files:
        # 提取文件前缀（类别标识）
        file_prefix = fasta_file.split('_')[0]
        
        # 获取对应的标签
        label = LABEL_MAPPING.get(file_prefix, 'unknown')
        if label == 'unknown':
            print(f"警告：未知的文件前缀 '{file_prefix}' 在文件 {fasta_file}")
        
        # 处理每个序列
        fasta_path = os.path.join(directory_path, fasta_file)
        sequence_count = 0
        
        try:
            with open(fasta_path, 'r') as f:
                for line in f:
                    if line.startswith('>'):
                        # 提取序列标识符
                        sequence_id = extract_sequence_id(line)
                        sequence_label_pairs.append((sequence_id, label))
                        sequence_count += 1
            
            print(f"处理完成：{fasta_file} - {sequence_count} 个序列，标签类型：{label}")
            
        except Exception as e:
            print(f"错误：处理文件 {fasta_file} 时出错 - {e}")
    
    return sequence_label_pairs

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='DNA序列标签生成器')
    parser.add_argument('--train_dir', type=str, default='d:/5-class/train', 
                        help='训练集文件夹路径')
    parser.add_argument('--val_dir', type=str, default='d:/5-class/val', 
                        help='验证集文件夹路径')
    parser.add_argument('--test_dir', type=str, default='d:/5-class/test', 
                        help='测试集文件夹路径')
    
    args = parser.parse_args()
    
    # 处理每个数据集
    datasets = [
        ('训练集', args.train_dir),
        ('验证集', args.val_dir),
        ('测试集', args.test_dir)
    ]
    
    for dataset_name, dataset_dir in datasets:
        print(f"\n开始处理{dataset_name}...")
        
        # 生成标签
        labels = generate_labels_for_directory(dataset_dir)
        
        if labels:
            # 生成标签文件路径
            label_file_path = os.path.join(dataset_dir, 'labels.txt')
            
            # 写入标签文件
            try:
                with open(label_file_path, 'w') as f:
                    for sequence_id, label in labels:
                        f.write(f"{sequence_id} {label}\n")
                
                print(f"✓ 标签文件已生成：{label_file_path}")
                print(f"  标签总数：{len(labels)}")
                
            except Exception as e:
                print(f"✗ 生成标签文件时出错：{e}")
        else:
            print(f"✗ 未生成标签文件")
    
    print("\n所有处理完成！")

if __name__ == "__main__":
    main()