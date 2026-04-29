#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具函数模块 - 包含日志记录、检查点保存、可视化等功能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import config


class Logger:
    """日志记录器"""
    
    def __init__(self, log_file=None, console=True):
        """
        初始化日志记录器
        
        参数:
            log_file: 日志文件路径
            console: 是否在控制台打印
        """
        self.log_file = log_file
        self.console = console
        
        # 创建日志文件目录
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # 清空或创建日志文件
            with open(log_file, 'w') as f:
                f.write(f"{config.PROJECT_NAME} 训练日志\n")
                f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n")
    
    def log(self, message, level="INFO"):
        """
        记录日志
        
        参数:
            message: 日志消息
            level: 日志级别
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        
        # 输出到控制台
        if self.console:
            print(log_message)
        
        # 写入日志文件
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_message + "\n")
    
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, 
                 class_loss=None, recon_loss=None):
        """记录epoch信息"""
        message = (f"Epoch {epoch+1:03d} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if class_loss is not None:
            message += f", Class Loss: {class_loss:.4f}"
        if recon_loss is not None:
            message += f", Recon Loss: {recon_loss:.4f}"
        
        self.log(message)
    
    def log_batch(self, epoch, batch, total_batches, loss, class_loss=None, 
                 recon_loss=None, acc=None):
        """记录batch信息"""
        if batch % config.LOG_INTERVAL == 0:
            message = (f"Epoch {epoch+1:03d} - Batch {batch}/{total_batches} - "
                      f"Loss: {loss:.4f}")
            
            if class_loss is not None:
                message += f", Class Loss: {class_loss:.4f}"
            if recon_loss is not None:
                message += f", Recon Loss: {recon_loss:.4f}"
            if acc is not None:
                message += f", Acc: {acc:.2f}%"
            
            self.log(message, level="DEBUG")


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=10, min_delta=0, verbose=True):
        """
        初始化早停
        
        参数:
            patience: 容忍的epoch数
            min_delta: 最小改善值
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss, model=None):
        """
        检查是否需要早停
        
        参数:
            val_loss: 验证指标（可以是损失或准确率）
            model: 模型（用于保存最佳状态）
        
        返回:
            是否需要早停
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            if model is not None:
                self.best_model_state = model.state_dict().copy()
        elif val_loss < self.best_loss + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'早停计数器: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('触发早停！')
        else:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None:
                self.best_model_state = model.state_dict().copy()
        
        return self.early_stop


def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, path):
    """
    保存检查点
    
    参数:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: epoch编号
        val_acc: 验证准确率
        path: 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_acc': val_acc,
        'config': {
            'num_classes': config.NUM_CLASSES,
            'primary_capsules': config.PRIMARY_CAPSULES,
            'primary_dim': config.PRIMARY_DIM,
            'digit_dim': config.DIGIT_DIM,
            'num_routing': config.NUM_ROUTING
        }
    }
    
    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 保存检查点
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device=None):
    """
    加载检查点
    
    参数:
        path: 检查点路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 设备
    
    返回:
        epoch: 加载的epoch编号
        val_acc: 验证准确率
    """
    if device is None:
        device = config.DEVICE
    
    checkpoint = torch.load(path, map_location=device)
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    val_acc = checkpoint.get('val_acc', 0.0)
    
    print(f"加载检查点: epoch={epoch}, val_acc={val_acc:.2f}%")
    
    return epoch, val_acc


def plot_training_history(history, save_path=None):
    """
    绘制训练历史
    
    参数:
        history: 训练历史字典
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 损失曲线
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='训练损失', color='blue', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='验证损失', color='orange', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].set_title('训练和验证损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 准确率曲线
    if 'train_acc' in history and 'val_acc' in history:
        axes[0, 1].plot(history['train_acc'], label='训练准确率', color='blue', linewidth=2)
        axes[0, 1].plot(history['val_acc'], label='验证准确率', color='orange', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('准确率 (%)')
        axes[0, 1].set_title('训练和验证准确率')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 学习率曲线
    if 'learning_rate' in history:
        axes[0, 2].plot(history['learning_rate'], color='green', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('学习率')
        axes[0, 2].set_title('学习率变化')
        axes[0, 2].grid(True, alpha=0.3)
    
    # 分类损失
    if 'class_loss' in history:
        axes[1, 0].plot(history['class_loss'], color='red', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('损失')
        axes[1, 0].set_title('分类损失')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 重构损失
    if 'recon_loss' in history:
        axes[1, 1].plot(history['recon_loss'], color='purple', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('损失')
        axes[1, 1].set_title('重构损失')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 准确率差值
    if 'train_acc' in history and 'val_acc' in history:
        gap = np.array(history['train_acc']) - np.array(history['val_acc'])
        axes[1, 2].plot(gap, color='brown', linewidth=2)
        axes[1, 2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('准确率差值 (%)')
        axes[1, 2].set_title('训练-验证准确率差值')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存: {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    绘制混淆矩阵
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        save_path: 保存路径（可选）
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 绘制热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存: {save_path}")
    
    plt.show()
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def plot_roc_curve(y_true, y_scores, class_names=None, save_path=None):
    """
    绘制ROC曲线
    
    参数:
        y_true: 真实标签（整数）
        y_scores: 预测分数（概率）
        class_names: 类别名称列表
        save_path: 保存路径（可选）
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    
    # 将标签二值化
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 计算微平均ROC曲线
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    
    # 绘制每个类别的ROC曲线
    colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))
    for i, color in enumerate(colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    # 绘制微平均ROC曲线
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'微平均 (AUC = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='随机分类器')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率')
    plt.ylabel('真正率')
    plt.title('多类别ROC曲线')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线已保存: {save_path}")
    
    plt.show()


def visualize_predictions(model, data_loader, num_samples=5, save_dir=None):
    """
    可视化预测结果
    
    参数:
        model: 训练好的模型
        data_loader: 数据加载器
        num_samples: 可视化的样本数
        save_dir: 保存目录（可选）
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 获取一个批次的数据
    images, labels = next(iter(data_loader))
    images, labels = images.to(device), labels.to(device)
    
    # 限制样本数
    num_samples = min(num_samples, len(images))
    
    # 预测
    with torch.no_grad():
        predictions, digit_caps, reconstructions = model(images[:num_samples], 
                                                        labels[:num_samples])
    
    # 获取预测类别
    pred_classes = torch.argmax(predictions, dim=1)
    true_classes = torch.argmax(labels[:num_samples], dim=1)
    
    # 创建可视化
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples*4))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # 原始图像
        img = images[i].cpu().squeeze().numpy()
        axes[i, 0].imshow(img, cmap='viridis')
        axes[i, 0].set_title(f"原始图像\n真实: {config.CLASS_NAMES[true_classes[i]]}")
        axes[i, 0].axis('off')
        
        # 重构图像
        if reconstructions is not None:
            recon_img = reconstructions[i].cpu().squeeze().numpy()
            axes[i, 1].imshow(recon_img, cmap='viridis')
            axes[i, 1].set_title("重构图像")
            axes[i, 1].axis('off')
            
            # 重构误差
            error_img = np.abs(img - recon_img)
            im = axes[i, 2].imshow(error_img, cmap='hot')
            axes[i, 2].set_title("重构误差")
            axes[i, 2].axis('off')
            plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        # 胶囊激活条形图
        capsule_activations = torch.norm(digit_caps[i], dim=-1).cpu().numpy()
        axes[i, 3].bar(range(len(config.CLASS_NAMES)), capsule_activations)
        axes[i, 3].set_xlabel('胶囊')
        axes[i, 3].set_ylabel('激活值')
        axes[i, 3].set_title(f"胶囊激活\n预测: {config.CLASS_NAMES[pred_classes[i]]}")
        axes[i, 3].set_xticks(range(len(config.CLASS_NAMES)))
        axes[i, 3].set_xticklabels(config.CLASS_NAMES, rotation=45, ha='right')
        axes[i, 3].axvline(x=pred_classes[i], color='red', linestyle='--', alpha=0.7)
        axes[i, 3].axvline(x=true_classes[i], color='green', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'predictions_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测可视化已保存: {save_path}")
    
    plt.show()


def analyze_capsule_activations(model, data_loader, save_dir=None):
    """
    分析胶囊激活模式
    
    参数:
        model: 训练好的模型
        data_loader: 数据加载器
        save_dir: 保存目录（可选）
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 收集所有胶囊激活
    all_activations = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            predictions, digit_caps, _ = model(images)
            
            # 获取胶囊激活和标签
            activations = torch.norm(digit_caps, dim=-1).cpu().numpy()
            labels_indices = torch.argmax(labels, dim=1).cpu().numpy()
            
            all_activations.append(activations)
            all_labels.append(labels_indices)
    
    # 合并数据
    all_activations = np.vstack(all_activations)
    all_labels = np.concatenate(all_labels)
    
    # 分析每个类别的平均激活
    print("\n胶囊激活分析:")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        # 获取该类别的样本
        class_mask = all_labels == class_idx
        if np.sum(class_mask) == 0:
            continue
        
        class_activations = all_activations[class_mask]
        
        # 计算平均激活
        mean_activations = class_activations.mean(axis=0)
        
        # 绘制该类别的激活条形图
        ax = axes[class_idx]
        bars = ax.bar(range(len(config.CLASS_NAMES)), mean_activations)
        
        # 高亮最高激活的胶囊
        top_capsule = np.argmax(mean_activations)
        bars[top_capsule].set_color('red')
        
        ax.set_title(f'{class_name}\n最高激活: {config.CLASS_NAMES[top_capsule]}')
        ax.set_xlabel('胶囊')
        ax.set_ylabel('平均激活值')
        ax.set_xticks(range(len(config.CLASS_NAMES)))
        ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # 打印统计信息
        print(f"\n{class_name}:")
        print(f"  样本数: {np.sum(class_mask)}")
        print(f"  最高激活胶囊: {config.CLASS_NAMES[top_capsule]} (索引: {top_capsule})")
        print(f"  最高激活值: {mean_activations[top_capsule]:.4f}")
    
    # 隐藏多余的子图
    for i in range(len(config.CLASS_NAMES), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('各类别胶囊激活模式', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'capsule_activations.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"胶囊激活分析图已保存: {save_path}")
    
    plt.show()
    
    # 计算胶囊激活的相关性矩阵
    print("\n胶囊激活相关性矩阵:")
    activation_corr = np.corrcoef(all_activations.T)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(activation_corr, annot=True, fmt='.2f', cmap='coolwarm',
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    plt.title('胶囊激活相关性矩阵')
    
    if save_dir:
        save_path = os.path.join(save_dir, 'capsule_correlation.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"胶囊相关性矩阵已保存: {save_path}")
    
    plt.show()


def save_results(results, save_path):
    """
    保存评估结果
    
    参数:
        results: 结果字典
        save_path: 保存路径
    """
    # 转换为可序列化的格式
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results_serializable[key] = value.tolist()
        elif torch.is_tensor(value):
            results_serializable[key] = value.cpu().numpy().tolist()
        else:
            results_serializable[key] = value
    
    # 保存到文件
    with open(save_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"评估结果已保存: {save_path}")


def print_results_summary(results):
    """
    打印结果摘要
    
    参数:
        results: 结果字典
    """
    print("\n" + "="*60)
    print("评估结果摘要")
    print("="*60)
    
    # 基本指标
    if 'test_acc' in results:
        print(f"测试准确率: {results['test_acc']:.2f}%")
    
    if 'test_loss' in results:
        print(f"测试损失: {results['test_loss']:.4f}")
    
    if 'test_class_loss' in results:
        print(f"测试分类损失: {results['test_class_loss']:.4f}")
    
    if 'test_recon_loss' in results:
        print(f"测试重构损失: {results['test_recon_loss']:.4f}")
    
    # 分类报告
    if 'predictions' in results and 'targets' in results:
        from sklearn.metrics import classification_report
        print("\n详细分类报告:")
        print(classification_report(results['targets'], results['predictions'], 
                                   target_names=config.CLASS_NAMES))
    
    print("="*60)


if __name__ == "__main__":
    # 测试工具函数
    print("测试工具函数...")
    
    # 测试Logger
    logger = Logger(console=True)
    logger.log("测试日志消息")
    logger.log_epoch(0, 0.5, 80.0, 0.6, 75.0, 0.4, 0.2)
    
    # 测试EarlyStopping
    early_stopping = EarlyStopping(patience=3, verbose=True)
    losses = [0.5, 0.4, 0.45, 0.43, 0.42, 0.41]
    for i, loss in enumerate(losses):
        if early_stopping(loss):
            print(f"在第 {i+1} 次迭代触发早停")
            break
    
    # 创建虚拟数据测试可视化函数
    np.random.seed(42)
    
    # 虚拟混淆矩阵数据
    y_true = np.random.randint(0, 5, 100)
    y_pred = np.random.randint(0, 5, 100)
    y_scores = np.random.rand(100, 5)
    
    # 测试绘图函数
    print("\n测试绘图函数...")
    
    # 创建临时目录
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # 测试混淆矩阵
    plot_confusion_matrix(y_true, y_pred, save_path=os.path.join(temp_dir, 'cm.png'))
    
    # 测试ROC曲线
    plot_roc_curve(y_true, y_scores, save_path=os.path.join(temp_dir, 'roc.png'))
    
    # 测试训练历史绘图
    history = {
        'train_loss': [0.8, 0.6, 0.5, 0.4, 0.3],
        'val_loss': [0.7, 0.55, 0.45, 0.35, 0.3],
        'train_acc': [70, 75, 80, 85, 90],
        'val_acc': [65, 70, 75, 80, 85],
        'learning_rate': [0.001, 0.001, 0.001, 0.0005, 0.0005]
    }
    plot_training_history(history, save_path=os.path.join(temp_dir, 'history.png'))
    
    print(f"\n测试完成！临时文件保存在: {temp_dir}")