#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNA序列FCGR生成器

功能：为DNA序列数据集生成FCGR矩阵（包含原始序列和反向互补链）

使用方法：
python generate_fcgr.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR --k K --normalize NORMALIZE

参数说明：
--input_dir: 数据集文件夹路径（包含fasta文件和labels.txt）
--output_dir: FCGR数据输出目录
--k: k-mer大小，默认6
--normalize: 归一化方式，可选值："cgrclust"（使用CGRclust的归一化方法）或"frequency"（仅频率），默认"cgrclust"

示例：
python generate_fcgr.py --input_dir "d:/5-class/train" --output_dir "d:/5-class/fcgr_output/train" --k 6 --normalize "cgrclust"

输出：
- 在输出目录中生成fcgr.npz文件，包含双通道FCGR矩阵和对应的标签
- 每个FCGR矩阵的形状为(2, 64, 64)，其中：
  - 第一通道：原始序列的FCGR
  - 第二通道：反向互补链的FCGR
"""

import os
import argparse
import numpy as np
from Bio import SeqIO
import sys

# 尝试导入tqdm用于进度显示
try:
    from tqdm import tqdm
except ImportError:
    print("警告：未安装tqdm库，将不会显示进度条。请使用 'pip install tqdm' 安装以启用进度显示功能。")
    tqdm = lambda x, desc="": x

# 添加CGRclust项目的路径到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'CGRclust-main', 'src', 'utils'))

from CGR_utils import FCGR

# 直接实现序列预处理功能
def preprocess_seq(seq: str) -> str:
    """预处理DNA序列，将非规范核苷酸替换为'N'"""
    processed_seq = []
    for nuc in seq:
        if nuc in ['A', 'C', 'G', 'T']:
            processed_seq.append(nuc)
        else:
            processed_seq.append('N')
    return ''.join(processed_seq)

def reverse_complement(seq: str) -> str:
    """
    生成DNA序列的反向互补链
    
    参数：
    seq: DNA序列字符串
    
    返回：
    反向互补链字符串
    """
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join([complement[base] for base in reversed(seq)])

def read_label_file(label_path):
    """
    读取标签文件并返回序列ID到标签的映射字典
    
    参数：
    label_path: 标签文件路径
    
    返回：
    label_dict: 序列ID到标签的映射字典
    """
    label_dict = {}
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    sequence_id, label = parts
                    label_dict[sequence_id] = label
    except Exception as e:
        print(f"错误：读取标签文件 {label_path} 时出错 - {e}")
    return label_dict

def extract_sequence_id(header):
    """
    从FASTA序列头部提取序列ID
    
    参数：
    header: FASTA序列头部字符串（不含'>'）
    
    返回：
    sequence_id: 提取的序列ID（格式：accession_numerical1_numerical2）
    """
    parts = header.split()
    if not parts:
        return ""
    
    sequence_id = parts[0]
    numerical_values = []
    
    # 从后向前查找两个数值
    for part in reversed(parts):
        if part.isdigit():
            numerical_values.insert(0, part)
            if len(numerical_values) == 2:
                break
    
    # 如果找到两个数值，添加到序列ID后面
    if numerical_values:
        sequence_id = f"{sequence_id}_{'_'.join(numerical_values)}"
    
    return sequence_id

def read_fasta_files(directory_path):
    """
    读取指定目录下的所有fasta文件
    
    参数：
    directory_path: 包含fasta文件的目录路径
    
    返回：
    sequences: 序列列表
    labels: 标签列表
    sequence_ids: 序列ID列表
    """
    sequences = []
    labels = []
    sequence_ids = []
    
    # 读取标签文件
    label_path = os.path.join(directory_path, 'labels.txt')
    print(f"  标签文件路径：{label_path}")
    label_dict = read_label_file(label_path)
    
    if not label_dict:
        print(f"警告：未找到有效标签或标签文件 {label_path} 为空")
        # 检查标签文件是否存在
        if not os.path.exists(label_path):
            print(f"  标签文件不存在：{label_path}")
        else:
            print(f"  标签文件存在但内容为空或格式不正确")
    
    # 获取目录中的所有fasta文件
    files = os.listdir(directory_path)
    fasta_files = [f for f in files if f.endswith('.fasta')]
    
    if not fasta_files:
        print(f"警告：目录中没有fasta文件 - {directory_path}")
        return sequences, labels, sequence_ids
    
    # 按文件名排序，确保处理顺序一致
    fasta_files.sort()
    
    for fasta_file in fasta_files:
        # 读取fasta文件
        fasta_path = os.path.join(directory_path, fasta_file)
        
        try:
            with open(fasta_path, 'r') as handle:
                # 获取记录总数以显示进度
                records = list(SeqIO.parse(handle, 'fasta'))
                
                # 使用tqdm显示进度
                for record in tqdm(records, desc=f"读取 {fasta_file}", unit="序列"):
                    # 预处理序列
                    sequence = preprocess_seq(str(record.seq).replace('U', 'T'))
                    
                    # 提取序列ID
                    seq_id = extract_sequence_id(record.description)
                    
                    # 查找标签
                    label = label_dict.get(seq_id, 'unknown')
                    if label == 'unknown':
                        print(f"警告：未找到序列ID {seq_id} 的标签")
                    
                    sequences.append(sequence)
                    labels.append(label)
                    sequence_ids.append(seq_id)
            
            print(f"处理完成：{fasta_file} - 读取了 {sum(1 for seq in sequences)} 个序列")
            
        except Exception as e:
            print(f"错误：处理文件 {fasta_file} 时出错 - {e}")
    
    return sequences, labels, sequence_ids

def generate_fcgr_for_sequences(sequences, k, normalize_method):
    """
    为序列列表生成FCGR矩阵（包含原始序列和反向互补链）
    
    参数：
    sequences: 序列列表
    k: k-mer大小
    normalize_method: 归一化方式，可选值："cgrclust"或"frequency"
    
    返回：
    fcgr_list: FCGR矩阵列表，每个元素是形状为(2, 64, 64)的数组
              第一通道是原始序列的FCGR，第二通道是反向互补链的FCGR
    """
    fcgr_generator = FCGR(k=k)
    fcgr_list = []
    
    # 使用tqdm显示进度
    for sequence in tqdm(sequences, desc="生成FCGR矩阵", unit="序列"):
        # 生成原始序列的FCGR
        fcgr_original = fcgr_generator(sequence)
        
        # 生成反向互补链的FCGR
        rc_sequence = reverse_complement(sequence)
        fcgr_reverse = fcgr_generator(rc_sequence)
        
        # 应用归一化
        if normalize_method == 'cgrclust':
            # 使用CGRclust的归一化方法
            m1, M1 = fcgr_original.min(), fcgr_original.max()
            if M1 - m1 > 0:
                fcgr_original = (fcgr_original - m1) / (M1 - m1)
            else:
                fcgr_original = np.zeros_like(fcgr_original)
            
            m2, M2 = fcgr_reverse.min(), fcgr_reverse.max()
            if M2 - m2 > 0:
                fcgr_reverse = (fcgr_reverse - m2) / (M2 - m2)
            else:
                fcgr_reverse = np.zeros_like(fcgr_reverse)
        elif normalize_method == 'frequency':
            # 不进行归一化，只保留频率
            pass
        
        # 将两个FCGR合并为双通道数组 (2, 64, 64)
        fcgr_dual = np.stack([fcgr_original, fcgr_reverse], axis=0)
        fcgr_list.append(fcgr_dual)
    
    return fcgr_list

def save_fcgr(fcgr_list, labels, sequence_ids, output_path):
    """
    保存FCGR矩阵到npz文件
    
    参数：
    fcgr_list: FCGR矩阵列表
    labels: 标签列表
    sequence_ids: 序列ID列表
    output_path: 输出文件路径
    """
    try:
        np.savez(output_path,
                 fcgr=np.array(fcgr_list),
                 labels=np.array(labels),
                 sequence_ids=np.array(sequence_ids))
        print(f"✓ FCGR文件已保存：{output_path}")
    except Exception as e:
        print(f"✗ 保存FCGR文件时出错：{e}")

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='DNA序列FCGR生成器')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='数据集文件夹路径（包含fasta文件和labels.txt）')
    parser.add_argument('--k', type=int, default=6,
                        help='k-mer大小')
    parser.add_argument('--normalize', type=str, default='cgrclust',
                        choices=['cgrclust', 'frequency'],
                        help='归一化方式')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='FCGR数据输出目录')
    
    args = parser.parse_args()
    
    # 验证input_dir是否存在
    if not os.path.exists(args.input_dir):
        print(f"错误：输入目录 {args.input_dir} 不存在")
        sys.exit(1)
    
    print(f"\n开始处理数据集...")
    print(f"  输入目录：{args.input_dir}")
    
    # 读取fasta文件
    sequences, labels, sequence_ids = read_fasta_files(args.input_dir)
    
    if not sequences:
        print(f"✗ 未找到序列，程序退出")
        sys.exit(1)
    
    # 生成FCGR
    print(f"生成FCGR矩阵...")
    fcgr_list = generate_fcgr_for_sequences(sequences, args.k, args.normalize)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"  输出目录：{args.output_dir}")
    
    # 保存FCGR
    output_path = os.path.join(args.output_dir, 'fcgr.npz')
    save_fcgr(fcgr_list, labels, sequence_ids, output_path)
    
    print("\n所有处理完成！")

if __name__ == "__main__":
    main()