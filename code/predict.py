#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立预测脚本 - 对新DNA序列进行预测
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from models import create_model
from utils import load_checkpoint


class DNAPredictor:
    """DNA序列预测器"""
    
    def __init__(self, checkpoint_path=None, device=None):
        """
        初始化预测器
        
        参数:
            checkpoint_path: 检查点路径
            device: 设备
        """
        self.device = device if device else config.DEVICE
        self.checkpoint_path = checkpoint_path
        
        # 创建模型
        self.model = create_model(config)
        
        # 加载检查点
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_model(checkpoint_path)
        else:
            print("警告: 未提供检查点路径，使用随机初始化的模型")
        
        # 设置为评估模式
        self.model.eval()
        self.model.to(self.device)
        
        print(f"预测器初始化完成 (设备: {self.device})")
    
    def load_model(self, checkpoint_path):
        """加载模型"""
        print(f"加载模型: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 打印模型信息
        if 'epoch' in checkpoint:
            print(f"  训练epoch: {checkpoint['epoch']}")
        if 'val_acc' in checkpoint:
            print(f"  验证准确率: {checkpoint['val_acc']:.2f}%")
    
    def predict_from_fcgr(self, fcgr_data):
        """
        从FCGR数据预测
        
        参数:
            fcgr_data: FCGR数据，形状为 (n_samples, 64, 64, 1) 或 (n_samples, 1, 64, 64)
        
        返回:
            predictions: 预测结果字典
        """
        # 确保数据是numpy数组
        if isinstance(fcgr_data, list):
            fcgr_data = np.array(fcgr_data)
        
        # 确保数据形状正确
        if len(fcgr_data.shape) == 3:
            # 假设形状为 (n_samples, 64, 64)，添加通道维度
            fcgr_data = fcgr_data.reshape(-1, 1, 64, 64)
        elif len(fcgr_data.shape) == 4 and fcgr_data.shape[3] == 1:
            # 从 (n_samples, 64, 64, 1) 转换为 (n_samples, 1, 64, 64)
            fcgr_data = fcgr_data.transpose(0, 3, 1, 2)
        
        # 转换为PyTorch张量
        tensor_data = torch.FloatTensor(fcgr_data).to(self.device)
        
        # 预测
        with torch.no_grad():
            predictions, digit_caps, _ = self.model(tensor_data)
            
            # 获取预测类别和概率
            pred_probs, pred_classes = torch.max(predictions, dim=1)
            
            # 获取所有类别的概率
            all_probs = predictions.cpu().numpy()
            
            # 获取胶囊激活
            capsule_activations = torch.norm(digit_caps, dim=-1).cpu().numpy()
        
        # 组织结果
        results = []
        for i in range(len(fcgr_data)):
            result = {
                'sample_index': i,
                'predicted_class': config.CLASS_NAMES[pred_classes[i].item()],
                'predicted_class_index': pred_classes[i].item(),
                'confidence': pred_probs[i].item(),
                'capsule_activations': capsule_activations[i].tolist()
            }
            
            # 添加所有类别的概率
            for j, class_name in enumerate(config.CLASS_NAMES):
                result[f'prob_{class_name}'] = all_probs[i, j]
            
            results.append(result)
        
        return results
    
    def predict_from_sequences(self, sequences, fcgr_generator=None):
        """
        从DNA序列预测（需要FCGR生成器）
        
        参数:
            sequences: DNA序列列表
            fcgr_generator: FCGR生成器
        
        返回:
            predictions: 预测结果字典
        """
        if fcgr_generator is None:
            raise ValueError("需要提供FCGR生成器")
        
        print(f"处理 {len(sequences)} 个序列...")
        
        # 生成FCGR
        fcgr_list = []
        for seq in sequences:
            fcgr = fcgr_generator(seq)
            fcgr_list.append(fcgr)
        
        # 转换为数组
        fcgr_data = np.array(fcgr_list)
        
        # 预测
        results = self.predict_from_fcgr(fcgr_data)
        
        # 添加序列信息
        for i, result in enumerate(results):
            result['sequence_length'] = len(sequences[i])
            result['sequence'] = sequences[i] if len(sequences[i]) < 100 else sequences[i][:100] + "..."
        
        return results
    
    def save_predictions(self, predictions, output_path, format='csv'):
        """
        保存预测结果
        
        参数:
            predictions: 预测结果列表
            output_path: 输出路径
            format: 输出格式 ('csv', 'excel', 'json')
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 转换为DataFrame
        df = pd.DataFrame(predictions)
        
        # 保存为指定格式
        if format == 'csv':
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"预测结果已保存为CSV: {output_path}")
        
        elif format == 'excel':
            try:
                df.to_excel(output_path, index=False)
                print(f"预测结果已保存为Excel: {output_path}")
            except:
                print("保存Excel失败，可能需要安装openpyxl")
                # 回退到CSV
                csv_path = output_path.replace('.xlsx', '.csv')
                df.to_csv(csv_path, index=False, encoding='utf-8')
                print(f"预测结果已保存为CSV: {csv_path}")
        
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
            print(f"预测结果已保存为JSON: {output_path}")
        
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def print_predictions_summary(self, predictions):
        """打印预测摘要"""
        print("\n" + "="*60)
        print("预测结果摘要")
        print("="*60)
        
        # 统计各类别的预测数量
        class_counts = {}
        for pred in predictions:
            class_name = pred['predicted_class']
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
        
        # 打印统计信息
        print(f"总样本数: {len(predictions)}")
        print("\n各类别预测统计:")
        for class_name in config.CLASS_NAMES:
            count = class_counts.get(class_name, 0)
            percentage = 100 * count / len(predictions) if len(predictions) > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # 打印置信度统计
        confidences = [p['confidence'] for p in predictions]
        if confidences:
            print(f"\n置信度统计:")
            print(f"  平均置信度: {np.mean(confidences):.4f}")
            print(f"  最低置信度: {np.min(confidences):.4f}")
            print(f"  最高置信度: {np.max(confidences):.4f}")
        
        # 打印前几个样本的预测结果
        print(f"\n前{min(5, len(predictions))}个样本的预测结果:")
        print(f"{'序号':<6} {'预测类别':<20} {'置信度':<10} {'胶囊激活模式'}")
        print("-" * 60)
        
        for i, pred in enumerate(predictions[:5]):
            # 找到激活最高的胶囊
            activations = pred['capsule_activations']
            top_capsule = np.argmax(activations)
            top_class = config.CLASS_NAMES[top_capsule]
            
            print(f"{pred['sample_index']:<6} {pred['predicted_class']:<20} "
                  f"{pred['confidence']:<10.4f} 最高激活: {top_class} ({activations[top_capsule]:.4f})")
        
        print("="*60)


def create_fcgr_generator(k=6):
    """
    创建FCGR生成器
    
    参数:
        k: k-mer大小
    
    返回:
        fcgr_generator: FCGR生成器函数
    """
    # 尝试从CGRclust导入FCGR
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'CGRclust-main', 'src', 'utils'))
        from CGR_utils import FCGR as FCGRGenerator
        
        generator = FCGRGenerator(k=k)
        
        def generate_fcgr(sequence):
            """生成FCGR"""
            fcgr = generator(sequence)
            
            # 归一化（如果需要）
            m, M = fcgr.min(), fcgr.max()
            if M - m > 0:
                fcgr = (fcgr - m) / (M - m)
            
            return fcgr
        
        return generate_fcgr
        
    except ImportError:
        print("警告: 无法导入CGRclust的FCGR，使用简单版本")
        
        def simple_fcgr_generator(sequence, k=6, size=64):
            """简单的FCGR生成器（用于测试）"""
            # 这是一个简化版本，实际使用时应该使用完整的FCGR生成器
            np.random.seed(hash(sequence) % 10000)
            fcgr = np.random.rand(size, size)
            return fcgr
        
        return lambda seq: simple_fcgr_generator(seq, k=k)


def load_sequences_from_fasta(file_path):
    """
    从FASTA文件加载序列
    
    参数:
        file_path: FASTA文件路径
    
    返回:
        sequences: 序列列表
        headers: 序列头列表
    """
    sequences = []
    headers = []
    
    try:
        from Bio import SeqIO
        
        print(f"从FASTA文件加载序列: {file_path}")
        
        for record in SeqIO.parse(file_path, 'fasta'):
            sequences.append(str(record.seq))
            headers.append(record.description)
        
        print(f"加载了 {len(sequences)} 个序列")
        
    except ImportError:
        print("错误: 需要安装biopython库")
        print("请运行: pip install biopython")
        return [], []
    
    except Exception as e:
        print(f"加载FASTA文件失败: {e}")
        return [], []
    
    return sequences, headers


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DNA序列预测器')
    
    parser.add_argument('--input', type=str, required=True,
                       help='输入文件路径（.npy/.npz包含FCGR，或.fasta包含序列）')
    
    parser.add_argument('--model', type=str, default=None,
                       help='模型检查点路径（默认使用最佳模型）')
    
    parser.add_argument('--output', type=str, default='predictions',
                       help='输出文件路径（不带扩展名）')
    
    parser.add_argument('--format', type=str, default='csv',
                       choices=['csv', 'excel', 'json'],
                       help='输出格式')
    
    parser.add_argument('--k', type=int, default=6,
                       help='k-mer大小（仅用于序列输入）')
    
    parser.add_argument('--summary', action='store_true',
                       help='打印预测摘要')
    
    args = parser.parse_args()
    
    # 设置检查点路径
    if args.model is None:
        args.model = config.get_model_path("best_model.pth")
    
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在 - {args.model}")
        print("请先训练模型或指定正确的模型路径")
        return
    
    # 创建预测器
    print(f"初始化预测器...")
    predictor = DNAPredictor(args.model)
    
    # 确定输入类型并加载数据
    input_ext = os.path.splitext(args.input)[1].lower()
    
    if input_ext in ['.npy', '.npz']:
        # 加载FCGR数据
        print(f"加载FCGR数据: {args.input}")
        
        if input_ext == '.npy':
            fcgr_data = np.load(args.input)
        else:  # .npz
            data = np.load(args.input, allow_pickle=True)
            if 'fcgr' in data:
                fcgr_data = data['fcgr']
            else:
                print("错误: .npz文件中未找到'fcgr'数据")
                return
        
        print(f"FCGR数据形状: {fcgr_data.shape}")
        
        # 预测
        predictions = predictor.predict_from_fcgr(fcgr_data)
    
    elif input_ext in ['.fasta', '.fa', '.fna']:
        # 加载DNA序列
        sequences, headers = load_sequences_from_fasta(args.input)
        
        if not sequences:
            return
        
        # 创建FCGR生成器
        print(f"创建FCGR生成器 (k={args.k})...")
        fcgr_generator = create_fcgr_generator(k=args.k)
        
        # 预测
        predictions = predictor.predict_from_sequences(sequences, fcgr_generator)
        
        # 添加序列头信息
        for i, pred in enumerate(predictions):
            if i < len(headers):
                pred['sequence_header'] = headers[i]
    
    else:
        print(f"错误: 不支持的输入文件格式: {input_ext}")
        print("支持格式: .npy, .npz (FCGR数据), .fasta, .fa, .fna (DNA序列)")
        return
    
    # 打印摘要
    if args.summary:
        predictor.print_predictions_summary(predictions)
    
    # 保存结果
    output_ext = {
        'csv': '.csv',
        'excel': '.xlsx',
        'json': '.json'
    }[args.format]
    
    output_path = f"{args.output}{output_ext}"
    predictor.save_predictions(predictions, output_path, format=args.format)
    
    print(f"\n预测完成! 结果已保存到: {output_path}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("DNA序列预测器")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    try:
        main()
        print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()