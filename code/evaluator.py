#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估器模块 - 负责模型评估和性能分析
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score, average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from config import config
from utils import plot_confusion_matrix, plot_roc_curve, analyze_capsule_activations


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, device=None):
        """
        初始化评估器
        
        参数:
            model: 训练好的模型
            device: 设备
        """
        self.model = model
        self.device = device if device else config.DEVICE
        
        # 将模型移动到设备并设置为评估模式
        self.model.to(self.device)
        self.model.eval()
        
        # 存储评估结果
        self.results = {}
        self.predictions = None
        self.targets = None
        self.probabilities = None
        self.sequence_ids = None
    
    def evaluate(self, data_loader):
        """
        评估模型
        
        参数:
            data_loader: 数据加载器
        
        返回:
            results: 评估结果字典
        """
        print("开始评估模型...")
        
        # 重置结果
        self.results = {}
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_features = []  # 胶囊特征
        all_sequence_ids = []  # 序列ID
        
        total_loss = 0.0
        total_class_loss = 0.0
        total_recon_loss = 0.0
        
        # 导入损失函数（避免循环导入）
        from losses import CapsuleTotalLoss
        loss_fn = CapsuleTotalLoss(reconstruction_weight=config.RECONSTRUCTION_WEIGHT)
        
        # 获取数据集的sequence_ids
        dataset = data_loader.dataset
        if hasattr(dataset, 'dataset'):  # 如果是Subset
            dataset = dataset.dataset
        
        if hasattr(dataset, 'sequence_ids'):
            self.sequence_ids = dataset.sequence_ids
        else:
            self.sequence_ids = None
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                # 移动数据到设备
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                predictions, digit_caps, reconstruction = self.model(data, target)
                
                # 计算损失
                total_loss_batch, class_loss_batch, recon_loss_batch = loss_fn(
                    target, predictions, data, reconstruction
                )
                
                # 累计损失
                total_loss += total_loss_batch.item()
                total_class_loss += class_loss_batch.item()
                total_recon_loss += recon_loss_batch.item()
                
                # 获取预测结果
                _, predicted = predictions.max(1)
                _, target_labels = target.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target_labels.cpu().numpy())
                all_probabilities.append(predictions.cpu().numpy())
                
                # 提取胶囊特征
                capsule_features = torch.norm(digit_caps, dim=-1).cpu().numpy()
                all_features.append(capsule_features)
                
                # 打印进度
                if batch_idx % 10 == 0:
                    print(f"处理批次: {batch_idx}/{len(data_loader)}")
        
        # 转换为numpy数组
        self.predictions = np.array(all_predictions)
        self.targets = np.array(all_targets)
        self.probabilities = np.vstack(all_probabilities)
        self.features = np.vstack(all_features)
        
        # 计算评估指标
        self._compute_metrics()
        
        # 计算损失
        self.results['test_loss'] = total_loss / len(data_loader)
        self.results['test_class_loss'] = total_class_loss / len(data_loader)
        self.results['test_recon_loss'] = total_recon_loss / len(data_loader)
        
        print("评估完成!")
        
        return self.results
    
    def _compute_metrics(self):
        """计算各种评估指标"""
        # 基础指标
        self.results['accuracy'] = accuracy_score(self.targets, self.predictions)
        
        # 多类别指标（宏平均和微平均）
        self.results['precision_macro'] = precision_score(self.targets, self.predictions, 
                                                         average='macro', zero_division=0)
        self.results['recall_macro'] = recall_score(self.targets, self.predictions, 
                                                   average='macro', zero_division=0)
        self.results['f1_macro'] = f1_score(self.targets, self.predictions, 
                                           average='macro', zero_division=0)
        
        self.results['precision_micro'] = precision_score(self.targets, self.predictions, 
                                                         average='micro', zero_division=0)
        self.results['recall_micro'] = recall_score(self.targets, self.predictions, 
                                                   average='micro', zero_division=0)
        self.results['f1_micro'] = f1_score(self.targets, self.predictions, 
                                           average='micro', zero_division=0)
        
        # 加权指标
        self.results['precision_weighted'] = precision_score(self.targets, self.predictions, 
                                                           average='weighted', zero_division=0)
        self.results['recall_weighted'] = recall_score(self.targets, self.predictions, 
                                                     average='weighted', zero_division=0)
        self.results['f1_weighted'] = f1_score(self.targets, self.predictions, 
                                             average='weighted', zero_division=0)
        
        # AUC-ROC（多类别）
        try:
            # 对概率进行softmax归一化
            from scipy.special import softmax
            probabilities_normalized = softmax(self.probabilities, axis=1)
            self.results['roc_auc_macro'] = roc_auc_score(self.targets, probabilities_normalized, 
                                                         multi_class='ovo', average='macro')
            self.results['roc_auc_weighted'] = roc_auc_score(self.targets, probabilities_normalized, 
                                                           multi_class='ovo', average='weighted')
        except Exception as e:
            print(f"ROC-AUC计算失败: {e}")
            self.results['roc_auc_macro'] = 0.0
            self.results['roc_auc_weighted'] = 0.0
        
        # 平均精确率
        try:
            from scipy.special import softmax
            probabilities_normalized = softmax(self.probabilities, axis=1)
            self.results['average_precision_macro'] = average_precision_score(
                self.targets, probabilities_normalized, average='macro'
            )
            self.results['average_precision_weighted'] = average_precision_score(
                self.targets, probabilities_normalized, average='weighted'
            )
        except Exception as e:
            print(f"平均精确率计算失败: {e}")
            self.results['average_precision_macro'] = 0.0
            self.results['average_precision_weighted'] = 0.0
        
        # 每个类别的指标
        self.results['per_class_metrics'] = {}
        for i, class_name in enumerate(config.CLASS_NAMES):
            class_mask = self.targets == i
            if np.sum(class_mask) > 0:
                class_precision = precision_score(self.targets, self.predictions, 
                                                 labels=[i], average=None, zero_division=0)[0]
                class_recall = recall_score(self.targets, self.predictions, 
                                           labels=[i], average=None, zero_division=0)[0]
                class_f1 = f1_score(self.targets, self.predictions, 
                                   labels=[i], average=None, zero_division=0)[0]
                
                # 计算每个类别的准确率
                class_accuracy = 0.0
                try:
                    # 该类别的准确率 = 正确预测为该类别的样本数 / 该类别的真实样本数
                    true_positives = np.sum((self.targets == i) & (self.predictions == i))
                    class_accuracy = true_positives / np.sum(self.targets == i)
                except Exception as e:
                    print(f"计算类别 {class_name} 的准确率失败: {e}")
                    class_accuracy = 0.0
                
                # 计算每个类别的ROC-AUC（one-vs-rest）
                class_roc_auc = 0.0
                try:
                    from scipy.special import softmax
                    probabilities_normalized = softmax(self.probabilities, axis=1)
                    
                    # 创建二进制标签（one-vs-rest）
                    binary_targets = (self.targets == i).astype(int)
                    class_roc_auc = roc_auc_score(binary_targets, probabilities_normalized[:, i])
                except Exception as e:
                    print(f"计算类别 {class_name} 的ROC-AUC失败: {e}")
                    class_roc_auc = 0.0
                
                # 计算每个类别的平均精确率（Average Precision）
                class_avg_precision = 0.0
                try:
                    from scipy.special import softmax
                    probabilities_normalized = softmax(self.probabilities, axis=1)
                    
                    # 创建二进制标签（one-vs-rest）
                    binary_targets = (self.targets == i).astype(int)
                    class_avg_precision = average_precision_score(binary_targets, probabilities_normalized[:, i])
                except Exception as e:
                    print(f"计算类别 {class_name} 的平均精确率失败: {e}")
                    class_avg_precision = 0.0
                
                self.results['per_class_metrics'][class_name] = {
                    'accuracy': class_accuracy,
                    'precision': class_precision,
                    'recall': class_recall,
                    'f1': class_f1,
                    'roc_auc': class_roc_auc,
                    'average_precision': class_avg_precision,
                    'support': int(np.sum(class_mask))
                }
        
        # 计算按长度分组的指标
        if self.sequence_ids is not None:
            self._compute_metrics_by_length()
        
        # 计算三分类指标 (已禁用)
        # self._compute_3class_metrics()
    
    def _compute_metrics_by_length(self):
        """按序列长度计算评估指标"""
        # 从sequence_ids中提取长度信息
        # 格式假设为: accession_numerical1_numerical2，其中numerical2表示长度
        lengths = []
        for seq_id in self.sequence_ids:
            if isinstance(seq_id, str):
                parts = seq_id.split('_')
                if len(parts) >= 3 and parts[-1].isdigit():
                    lengths.append(int(parts[-1]))
                else:
                    lengths.append(0)  # 未知长度
            else:
                lengths.append(0)  # 无效的sequence_id
        
        lengths = np.array(lengths)
        
        # 获取唯一的长度值（假设有4种不同长度）
        unique_lengths = np.unique(lengths[lengths > 0])
        
        if len(unique_lengths) == 0:
            print("警告: 无法从sequence_ids中提取长度信息")
            return
        
        # 为每个长度组计算指标
        self.results['metrics_by_length'] = {}
        
        for length in unique_lengths:
            # 获取该长度的样本索引
            length_mask = lengths == length
            if np.sum(length_mask) < 10:  # 样本太少，跳过
                continue
            
            # 获取该长度组的预测和真实标签
            length_predictions = self.predictions[length_mask]
            length_targets = self.targets[length_mask]
            
            # 计算指标
            try:
                accuracy = accuracy_score(length_targets, length_predictions)
                precision = precision_score(length_targets, length_predictions, 
                                           average='macro', zero_division=0)
                recall = recall_score(length_targets, length_predictions, 
                                     average='macro', zero_division=0)
                f1 = f1_score(length_targets, length_predictions, 
                             average='macro', zero_division=0)
                
                self.results['metrics_by_length'][length] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'num_samples': int(np.sum(length_mask))
                }
            except Exception as e:
                print(f"计算长度 {length} 的指标时出错: {e}")
    
    def _convert_to_3class(self, labels_5class):
        """
        将五分类标签转换为三分类标签
        
        五分类: 0-prokaryotic_virus, 1-eukaryotic_virus, 2-prokaryote, 3-eukaryote, 4-plasmid
        三分类: 0-virus, 1-organism, 2-plasmid
        """
        labels_3class = np.zeros_like(labels_5class)
        for i, label in enumerate(labels_5class):
            if label in [0, 1]:  # prokaryotic_virus, eukaryotic_virus -> virus
                labels_3class[i] = 0
            elif label in [2, 3]:  # prokaryote, eukaryote -> organism
                labels_3class[i] = 1
            elif label == 4:  # plasmid -> plasmid
                labels_3class[i] = 2
        return labels_3class
    
    def _compute_3class_metrics(self):
        """计算三分类的评估指标"""
        # 转换标签
        targets_3class = self._convert_to_3class(self.targets)
        predictions_3class = self._convert_to_3class(self.predictions)
        
        # 合并概率分数
        # virus: prokaryotic_virus + eukaryotic_virus
        # organism: prokaryote + eukaryote
        # plasmid: plasmid
        from scipy.special import softmax
        probabilities_normalized = softmax(self.probabilities, axis=1)
        
        probabilities_3class = np.zeros((len(probabilities_normalized), 3))
        probabilities_3class[:, 0] = probabilities_normalized[:, 0] + probabilities_normalized[:, 1]  # virus
        probabilities_3class[:, 1] = probabilities_normalized[:, 2] + probabilities_normalized[:, 3]  # organism
        probabilities_3class[:, 2] = probabilities_normalized[:, 4]  # plasmid
        
        # 三分类类别名称
        class_names_3class = ['virus', 'organism', 'plasmid']
        
        # 计算基础指标
        accuracy_3class = accuracy_score(targets_3class, predictions_3class)
        
        # 宏平均指标
        precision_macro_3class = precision_score(targets_3class, predictions_3class, 
                                                 average='macro', zero_division=0)
        recall_macro_3class = recall_score(targets_3class, predictions_3class, 
                                           average='macro', zero_division=0)
        f1_macro_3class = f1_score(targets_3class, predictions_3class, 
                                   average='macro', zero_division=0)
        
        # 微平均指标
        precision_micro_3class = precision_score(targets_3class, predictions_3class, 
                                                 average='micro', zero_division=0)
        recall_micro_3class = recall_score(targets_3class, predictions_3class, 
                                           average='micro', zero_division=0)
        f1_micro_3class = f1_score(targets_3class, predictions_3class, 
                                   average='micro', zero_division=0)
        
        # 加权指标
        precision_weighted_3class = precision_score(targets_3class, predictions_3class, 
                                                   average='weighted', zero_division=0)
        recall_weighted_3class = recall_score(targets_3class, predictions_3class, 
                                             average='weighted', zero_division=0)
        f1_weighted_3class = f1_score(targets_3class, predictions_3class, 
                                     average='weighted', zero_division=0)
        
        # ROC-AUC
        try:
            roc_auc_macro_3class = roc_auc_score(targets_3class, probabilities_3class, 
                                                 multi_class='ovr', average='macro')
            roc_auc_weighted_3class = roc_auc_score(targets_3class, probabilities_3class, 
                                                   multi_class='ovr', average='weighted')
        except Exception as e:
            print(f"三分类ROC-AUC计算失败: {e}")
            roc_auc_macro_3class = 0.0
            roc_auc_weighted_3class = 0.0
        
        # 平均精确率
        try:
            avg_precision_macro_3class = average_precision_score(
                targets_3class, probabilities_3class, average='macro'
            )
            avg_precision_weighted_3class = average_precision_score(
                targets_3class, probabilities_3class, average='weighted'
            )
        except Exception as e:
            print(f"三分类平均精确率计算失败: {e}")
            avg_precision_macro_3class = 0.0
            avg_precision_weighted_3class = 0.0
        
        # 每个类别的指标
        per_class_metrics_3class = {}
        for i, class_name in enumerate(class_names_3class):
            class_mask = targets_3class == i
            if np.sum(class_mask) > 0:
                class_precision = precision_score(targets_3class, predictions_3class, 
                                                 labels=[i], average=None, zero_division=0)[0]
                class_recall = recall_score(targets_3class, predictions_3class, 
                                           labels=[i], average=None, zero_division=0)[0]
                class_f1 = f1_score(targets_3class, predictions_3class, 
                                   labels=[i], average=None, zero_division=0)[0]
                
                # 计算每个类别的准确率
                true_positives = np.sum((targets_3class == i) & (predictions_3class == i))
                class_accuracy = true_positives / np.sum(targets_3class == i)
                
                # 计算每个类别的ROC-AUC（one-vs-rest）
                class_roc_auc = 0.0
                try:
                    binary_targets = (targets_3class == i).astype(int)
                    class_roc_auc = roc_auc_score(binary_targets, probabilities_3class[:, i])
                except Exception as e:
                    print(f"计算三分类类别 {class_name} 的ROC-AUC失败: {e}")
                
                # 计算每个类别的平均精确率
                class_avg_precision = 0.0
                try:
                    binary_targets = (targets_3class == i).astype(int)
                    class_avg_precision = average_precision_score(binary_targets, probabilities_3class[:, i])
                except Exception as e:
                    print(f"计算三分类类别 {class_name} 的平均精确率失败: {e}")
                
                per_class_metrics_3class[class_name] = {
                    'accuracy': class_accuracy,
                    'precision': class_precision,
                    'recall': class_recall,
                    'f1': class_f1,
                    'roc_auc': class_roc_auc,
                    'average_precision': class_avg_precision,
                    'support': int(np.sum(class_mask))
                }
        
        # 计算混淆矩阵
        cm_3class = confusion_matrix(targets_3class, predictions_3class)
        
        # 存储结果
        self.results['3class_metrics'] = {
            'accuracy': accuracy_3class,
            'precision_macro': precision_macro_3class,
            'recall_macro': recall_macro_3class,
            'f1_macro': f1_macro_3class,
            'precision_micro': precision_micro_3class,
            'recall_micro': recall_micro_3class,
            'f1_micro': f1_micro_3class,
            'precision_weighted': precision_weighted_3class,
            'recall_weighted': recall_weighted_3class,
            'f1_weighted': f1_weighted_3class,
            'roc_auc_macro': roc_auc_macro_3class,
            'roc_auc_weighted': roc_auc_weighted_3class,
            'avg_precision_macro': avg_precision_macro_3class,
            'avg_precision_weighted': avg_precision_weighted_3class,
            'per_class_metrics': per_class_metrics_3class,
            'confusion_matrix': cm_3class,
            'class_names': class_names_3class
        }
    
    def print_detailed_report(self):
        """打印详细评估报告"""
        print("\n" + "="*80)
        print("详细评估报告")
        print("="*80)
        
        # 打印基础指标
        print(f"\n基础指标:")
        print(f"  准确率: {self.results['accuracy']:.4f}")
        print(f"  测试损失: {self.results.get('test_loss', 'N/A'):.4f}")
        
        # 打印宏平均指标
        print(f"\n宏平均指标:")
        print(f"  精确率: {self.results['precision_macro']:.4f}")
        print(f"  召回率: {self.results['recall_macro']:.4f}")
        print(f"  F1分数: {self.results['f1_macro']:.4f}")
        print(f"  ROC-AUC: {self.results.get('roc_auc_macro', 'N/A'):.4f}")
        
        # 打印加权指标
        print(f"\n加权平均指标:")
        print(f"  精确率: {self.results['precision_weighted']:.4f}")
        print(f"  召回率: {self.results['recall_weighted']:.4f}")
        print(f"  F1分数: {self.results['f1_weighted']:.4f}")
        print(f"  ROC-AUC: {self.results.get('roc_auc_weighted', 'N/A'):.4f}")
        
        # 打印每个类别的指标
        print(f"\n每个类别的指标:")
        print(f"{'类别':<20} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'ROC-AUC':<10} {'平均精确率':<12} {'样本数':<10}")
        print("-" * 90)
        
        for class_name, metrics in self.results['per_class_metrics'].items():
            print(f"{class_name:<20} {metrics['accuracy']:<10.4f} "
                  f"{metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f} "
                  f"{metrics['roc_auc']:<10.4f} {metrics['average_precision']:<12.4f} "
                  f"{metrics['support']:<10}")
        
        # 打印按长度分组的指标
        if 'metrics_by_length' in self.results and self.results['metrics_by_length']:
            print(f"\n按序列长度分组的指标:")
            print(f"{'长度':<10} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'样本数':<10}")
            print("-" * 60)
            
            # 按长度排序
            sorted_lengths = sorted(self.results['metrics_by_length'].keys())
            for length in sorted_lengths:
                metrics = self.results['metrics_by_length'][length]
                print(f"{length:<10} {metrics['accuracy']:<10.4f} "
                      f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                      f"{metrics['f1']:<10.4f} {metrics['num_samples']:<10}")
        
        # 打印混淆矩阵
        print(f"\n混淆矩阵:")
        cm = confusion_matrix(self.targets, self.predictions)
        print(pd.DataFrame(cm, 
                          index=[f"真实_{name}" for name in config.CLASS_NAMES],
                          columns=[f"预测_{name}" for name in config.CLASS_NAMES]))
        
        print("="*80)
    
    def visualize_results(self, save_dir=None):
        """
        可视化评估结果
        
        参数:
            save_dir: 保存目录（可选）
        """
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
        
        # 1. 混淆矩阵
        print("\n绘制混淆矩阵...")
        cm_path = os.path.join(save_dir, 'confusion_matrix.png') if save_dir else None
        plot_confusion_matrix(self.targets, self.predictions, save_path=cm_path)
        
        # 2. ROC曲线
        print("\n绘制ROC曲线...")
        roc_path = os.path.join(save_dir, 'roc_curve.png') if save_dir else None
        plot_roc_curve(self.targets, self.probabilities, save_path=roc_path)
        
        # 3. 胶囊激活分析
        print("\n分析胶囊激活模式...")
        capsule_path = os.path.join(save_dir, 'capsule_analysis') if save_dir else None
        # 注意：这里需要数据加载器，我们稍后在主程序中调用
        
        # 4. 各类别性能条形图
        print("\n绘制各类别性能对比图...")
        self._plot_per_class_performance(save_dir)
        
        # 5. 预测概率分布
        print("\n绘制预测概率分布...")
        self._plot_probability_distribution(save_dir)
    
    def _plot_per_class_performance(self, save_dir=None):
        """绘制各类别性能对比图"""
        # 准备数据
        class_names = list(self.results['per_class_metrics'].keys())
        precision_values = [metrics['precision'] for metrics in self.results['per_class_metrics'].values()]
        recall_values = [metrics['recall'] for metrics in self.results['per_class_metrics'].values()]
        f1_values = [metrics['f1'] for metrics in self.results['per_class_metrics'].values()]
        
        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 精确率条形图
        bars1 = axes[0].bar(range(len(class_names)), precision_values, color='skyblue')
        axes[0].set_xlabel('类别')
        axes[0].set_ylabel('精确率')
        axes[0].set_title('各类别精确率')
        axes[0].set_xticks(range(len(class_names)))
        axes[0].set_xticklabels(class_names, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3)
        
        # 在条形图上添加数值
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # 召回率条形图
        bars2 = axes[1].bar(range(len(class_names)), recall_values, color='lightgreen')
        axes[1].set_xlabel('类别')
        axes[1].set_ylabel('召回率')
        axes[1].set_title('各类别召回率')
        axes[1].set_xticks(range(len(class_names)))
        axes[1].set_xticklabels(class_names, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # F1分数条形图
        bars3 = axes[2].bar(range(len(class_names)), f1_values, color='salmon')
        axes[2].set_xlabel('类别')
        axes[2].set_ylabel('F1分数')
        axes[2].set_title('各类别F1分数')
        axes[2].set_xticks(range(len(class_names)))
        axes[2].set_xticklabels(class_names, rotation=45, ha='right')
        axes[2].grid(True, alpha=0.3)
        
        for bar in bars3:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        plt.suptitle('各类别性能对比', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, 'per_class_performance.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"各类别性能对比图已保存: {save_path}")
        
        plt.show()
    
    def _plot_probability_distribution(self, save_dir=None):
        """绘制预测概率分布"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, class_name in enumerate(config.CLASS_NAMES):
            if i >= len(axes):
                break
            
            # 获取该类别的预测概率
            class_probs = self.probabilities[:, i]
            
            # 真实标签为该类别的样本
            true_class_mask = self.targets == i
            
            # 绘制概率分布直方图
            ax = axes[i]
            ax.hist(class_probs[true_class_mask], bins=20, alpha=0.7, 
                   label='正确分类', color='green', density=True)
            ax.hist(class_probs[~true_class_mask], bins=20, alpha=0.7,
                   label='错误分类', color='red', density=True)
            
            ax.set_xlabel('预测概率')
            ax.set_ylabel('密度')
            ax.set_title(f'{class_name} 的概率分布')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(config.CLASS_NAMES), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('各类别预测概率分布', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, 'probability_distribution.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"预测概率分布图已保存: {save_path}")
        
        plt.show()
    
    def analyze_misclassifications(self, data_loader, top_k=10):
        """
        分析错误分类的样本
        
        参数:
            data_loader: 数据加载器
            top_k: 显示前k个最不确定的样本
        """
        self.model.eval()
        
        # 收集错误分类的样本
        misclassified_samples = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # 预测
                predictions, digit_caps, _ = self.model(data)
                
                # 获取预测结果
                _, predicted = predictions.max(1)
                _, target_labels = target.max(1)
                
                # 找出错误分类的样本
                incorrect_mask = predicted != target_labels
                
                for i in range(len(data)):
                    if incorrect_mask[i]:
                        sample = {
                            'index': batch_idx * data_loader.batch_size + i,
                            'image': data[i].cpu().numpy(),
                            'true_label': target_labels[i].item(),
                            'pred_label': predicted[i].item(),
                            'probabilities': predictions[i].cpu().numpy(),
                            'confidence': predictions[i].max().item(),
                            'entropy': self._calculate_entropy(predictions[i])
                        }
                        misclassified_samples.append(sample)
        
        print(f"\n错误分类分析:")
        print(f"  总样本数: {len(data_loader.dataset)}")
        print(f"  错误分类数: {len(misclassified_samples)}")
        print(f"  错误率: {100 * len(misclassified_samples) / len(data_loader.dataset):.2f}%")
        
        if misclassified_samples:
            # 按不确定性排序
            misclassified_samples.sort(key=lambda x: x['entropy'], reverse=True)
            
            # 显示前k个最不确定的错误分类
            print(f"\n前{top_k}个最不确定的错误分类:")
            print(f"{'序号':<10} {'真实类别':<20} {'预测类别':<20} {'置信度':<10} {'熵':<10}")
            print("-" * 70)
            
            for i, sample in enumerate(misclassified_samples[:top_k]):
                true_class = config.CLASS_NAMES[sample['true_label']]
                pred_class = config.CLASS_NAMES[sample['pred_label']]
                print(f"{sample['index']:<10} {true_class:<20} {pred_class:<20} "
                      f"{sample['confidence']:<10.4f} {sample['entropy']:<10.4f}")
            
            # 分析错误模式
            self._analyze_error_patterns(misclassified_samples)
    
    def _calculate_entropy(self, probabilities):
        """计算预测概率的熵"""
        # 添加小值避免log(0)
        probs = probabilities.cpu().numpy() + 1e-10
        entropy = -np.sum(probs * np.log(probs))
        return entropy
    
    def _analyze_error_patterns(self, misclassified_samples):
        """分析错误分类模式"""
        # 统计真实类别和预测类别的混淆
        confusion_counts = {}
        
        for sample in misclassified_samples:
            true_class = config.CLASS_NAMES[sample['true_label']]
            pred_class = config.CLASS_NAMES[sample['pred_label']]
            key = (true_class, pred_class)
            
            if key not in confusion_counts:
                confusion_counts[key] = 0
            confusion_counts[key] += 1
        
        # 打印最常见的错误模式
        print(f"\n最常见的错误分类模式:")
        print(f"{'真实类别':<20} {'预测类别':<20} {'次数':<10} {'比例':<10}")
        print("-" * 60)
        
        sorted_patterns = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)
        total_errors = len(misclassified_samples)
        
        for (true_class, pred_class), count in sorted_patterns[:10]:
            percentage = 100 * count / total_errors
            print(f"{true_class:<20} {pred_class:<20} {count:<10} {percentage:<10.2f}%")
    
    def save_results(self, save_path):
        """
        保存评估结果
        
        参数:
            save_path: 保存路径
        """
        import json
        import os
        
        def convert_to_serializable(obj):
            """递归转换对象为可序列化的格式"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        # 创建可序列化的结果字典
        results_serializable = convert_to_serializable(self.results)
        
        # 添加预测结果
        results_serializable['predictions'] = self.predictions.tolist()
        results_serializable['targets'] = self.targets.tolist()
        results_serializable['probabilities'] = self.probabilities.tolist()
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存到文件
        with open(save_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"评估结果已保存: {save_path}")
    
    def generate_report(self, save_dir=None):
        """
        生成完整的评估报告
        
        参数:
            save_dir: 保存目录
        
        返回:
            report: 报告字典
        """
        report = {
            'summary': {
                'accuracy': self.results['accuracy'],
                'precision_macro': self.results['precision_macro'],
                'recall_macro': self.results['recall_macro'],
                'f1_macro': self.results['f1_macro'],
                'test_loss': self.results.get('test_loss', 0.0)
            },
            'per_class_metrics': self.results['per_class_metrics'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存报告
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            report_path = os.path.join(save_dir, 'evaluation_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"评估报告已保存: {report_path}")
        
        return report


def create_evaluator(model, checkpoint_path=None):
    """
    创建评估器
    
    参数:
        model: 模型
        checkpoint_path: 检查点路径（可选）
    
    返回:
        evaluator: 评估器
    """
    # 如果提供了检查点路径，加载模型权重
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"从检查点加载模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("使用当前模型权重进行评估")
    
    # 创建评估器
    evaluator = ModelEvaluator(model)
    
    return evaluator


if __name__ == "__main__":
    # 测试评估器
    print("测试评估器...")
    
    # 创建虚拟模型
    from models import create_model
    model = create_model(config)
    
    # 创建评估器
    evaluator = create_evaluator(model)
    
    # 创建虚拟数据
    np.random.seed(42)
    n_samples = 100
    n_classes = config.NUM_CLASSES
    
    # 虚拟预测结果
    predictions = np.random.randint(0, n_classes, n_samples)
    targets = np.random.randint(0, n_classes, n_samples)
    probabilities = np.random.rand(n_samples, n_classes)
    
    # 归一化概率
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    
    # 手动设置评估器结果
    evaluator.predictions = predictions
    evaluator.targets = targets
    evaluator.probabilities = probabilities
    evaluator._compute_metrics()
    
    # 打印详细报告
    evaluator.print_detailed_report()
    
    # 可视化结果（需要图形界面）
    # evaluator.visualize_results()
    
    print("\n评估器测试完成!")