#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据加载模块 - 修复版本，确保训练、验证、测试数据完全独立
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from config import config
import os
import sys

class FCGRDataset(Dataset):
    """FCGR数据集类"""
    
    def __init__(self, fcgr_path, transform=None, augment=False, class_names=None, 
                 mode='train'):
        """
        初始化FCGR数据集
        
        参数:
            fcgr_path: FCGR .npz文件路径
            transform: 数据转换函数
            augment: 是否使用数据增强
            class_names: 类别名称列表
            mode: 数据集模式（train, val, test）
        """
        self.fcgr_path = fcgr_path
        self.transform = transform
        self.augment = augment
        self.mode = mode
        self.class_names = class_names if class_names else config.CLASS_NAMES
        
        # 验证文件是否存在
        if not os.path.exists(self.fcgr_path):
            print(f"警告: FCGR文件不存在 - {self.fcgr_path}")
            self.fcgr_data = np.array([])
            self.labels = np.array([])
            self.sequence_ids = np.array([])
            return
        
        # 加载数据
        self.load_data()
        
        # 打印数据集统计信息
        if len(self.fcgr_data) > 0:
            self.print_stats()
    
    def load_data(self):
        """加载FCGR数据"""
        try:
            # 加载npz文件
            data = np.load(self.fcgr_path, allow_pickle=True)
            
            # 获取FCGR矩阵
            self.fcgr_data = data['fcgr']
            
            # 获取标签
            labels = data['labels']
            
            # 转换标签为整数索引
            self.labels = self.convert_labels_to_indices(labels)
            
            # 如果有序列ID，也加载
            if 'sequence_ids' in data:
                self.sequence_ids = data['sequence_ids']
            else:
                self.sequence_ids = np.arange(len(self.labels))
            
            print(f"✓ 成功加载 {self.mode} 数据: {self.fcgr_path}")
            print(f"  数据形状: {self.fcgr_data.shape}")
            print(f"  标签形状: {self.labels.shape}")
            
        except Exception as e:
            print(f"✗ 加载数据失败: {e}")
            self.fcgr_data = np.array([])
            self.labels = np.array([])
            self.sequence_ids = np.array([])
    
    def convert_labels_to_indices(self, labels):
        """
        将标签转换为整数索引
        
        参数:
            labels: 原始标签列表
        
        返回:
            indices: 整数标签索引
        """
        # 创建标签到索引的映射
        label_to_index = {name: idx for idx, name in enumerate(self.class_names)}
        
        # 转换标签
        indices = []
        unknown_labels = []
        
        for label in labels:
            if label in label_to_index:
                indices.append(label_to_index[label])
            else:
                # 记录未知标签
                if label not in unknown_labels:
                    unknown_labels.append(label)
                indices.append(-1)  # 标记为未知
        
        # 如果有未知标签，打印警告
        if unknown_labels:
            print(f"警告: 发现 {len(unknown_labels)} 个未知标签: {unknown_labels}")
        
        return np.array(indices)
    
    def print_stats(self):
        """打印数据集统计信息"""
        print(f"\n{self.mode.capitalize()}数据集统计信息:")
        print(f"  样本数量: {len(self)}")
        print(f"  FCGR形状: {self.fcgr_data.shape[1:]}")
        
        # 计算类别分布
        valid_indices = self.labels >= 0
        valid_labels = self.labels[valid_indices]
        
        if len(valid_labels) > 0:
            unique, counts = np.unique(valid_labels, return_counts=True)
            print(f"  类别分布:")
            for idx, count in zip(unique, counts):
                class_name = self.class_names[idx] if idx < len(self.class_names) else f"未知{idx}"
                print(f"    {class_name}: {count} ({count/len(self)*100:.2f}%)")
        
        # 统计未知标签
        unknown_count = np.sum(self.labels < 0)
        if unknown_count > 0:
            print(f"  未知标签: {unknown_count} ({unknown_count/len(self)*100:.2f}%)")
    
    def __len__(self):
        return len(self.fcgr_data)
    
    def __getitem__(self, idx):
        # 获取FCGR图像
        image = self.fcgr_data[idx]
        
        # 处理不同的FCGR形状
        if len(image.shape) == 3:
            # 双通道FCGR (2, 64, 64) - 原始序列和反向互补链
            # 已经是正确的形状，直接使用
            pass
        elif len(image.shape) == 2:
            # 单通道FCGR (64, 64) - 添加通道维度
            image = image[np.newaxis, ...]  # 变成 (1, H, W)
        
        # 转换为PyTorch张量
        image = torch.FloatTensor(image)
        
        # 获取标签
        label = self.labels[idx]
        
        # 如果标签无效，使用第一个类别（但这种情况应该很少）
        if label < 0:
            label = 0
        
        label_onehot = F.one_hot(torch.tensor(label), num_classes=config.NUM_CLASSES).float()
        
        # 只在训练模式下应用数据增强
        if self.mode == 'train' and self.augment and config.USE_AUGMENTATION:
            image = self.augment_image(image)
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
        
        return image, label_onehot
    
    def augment_image(self, image):
        """
        对FCGR图像应用数据增强
        
        参数:
            image: 输入图像张量
        
        返回:
            增强后的图像
        """
        import random
        
        # 只有一定概率应用增强
        if random.random() > config.AUGMENTATION_PROB:
            return image
        
        # 复制图像以免修改原始数据
        augmented = image.clone()
        
        # 应用不同的增强方法
        # 1. 随机水平翻转
        if random.random() > 0.5:
            augmented = torch.flip(augmented, dims=[2])
        
        # 2. 随机垂直翻转
        if random.random() > 0.5:
            augmented = torch.flip(augmented, dims=[1])
        
        # 3. 随机旋转90度
        if random.random() > 0.5:
            k = random.randint(0, 3)
            augmented = torch.rot90(augmented, k, dims=[1, 2])
        
        # 4. 添加高斯噪声
        if random.random() > 0.7:
            noise = torch.randn_like(augmented) * 0.01
            augmented = augmented + noise
            augmented = torch.clamp(augmented, 0, 1)
        
        # 5. 亮度/对比度调整
        if random.random() > 0.7:
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            augmented = augmented * brightness
            augmented = (augmented - augmented.mean()) * contrast + augmented.mean()
            augmented = torch.clamp(augmented, 0, 1)
        
        return augmented


class FCRGDataModule:
    """FCGR数据模块 - 修复版本，独立加载训练、验证、测试数据"""
    
    def __init__(self, config):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # 验证数据路径
        self.validate_data_paths()
        
        # 加载数据
        self.load_datasets()
        self.create_dataloaders()
    
    def validate_data_paths(self):
        """验证数据路径"""
        print("验证数据路径...")
        
        data_paths = {
            'train': self.config.TRAIN_FCGR_PATH,
            'val': self.config.VAL_FCGR_PATH,
            'test': self.config.TEST_FCGR_PATH
        }
        
        for mode, path in data_paths.items():
            if os.path.exists(path):
                print(f"  ✓ {mode}数据路径: {path}")
            else:
                print(f"  ✗ {mode}数据路径不存在: {path}")
    
    def load_datasets(self):
        """加载所有数据集"""
        print("\n加载数据集...")
        
        # 训练集（使用数据增强）
        self.train_dataset = FCGRDataset(
            fcgr_path=self.config.TRAIN_FCGR_PATH,
            augment=True,
            class_names=self.config.CLASS_NAMES,
            mode='train'
        )
        
        # 验证集
        self.val_dataset = FCGRDataset(
            fcgr_path=self.config.VAL_FCGR_PATH,
            augment=False,
            class_names=self.config.CLASS_NAMES,
            mode='val'
        )
        
        # 测试集
        self.test_dataset = FCGRDataset(
            fcgr_path=self.config.TEST_FCGR_PATH,
            augment=False,
            class_names=self.config.CLASS_NAMES,
            mode='test'
        )
        
        # 检查数据集是否为空
        datasets = {
            '训练集': self.train_dataset,
            '验证集': self.val_dataset,
            '测试集': self.test_dataset
        }
        
        empty_datasets = []
        for name, dataset in datasets.items():
            if len(dataset) == 0:
                empty_datasets.append(name)
        
        if empty_datasets:
            print(f"\n警告: 以下数据集为空: {', '.join(empty_datasets)}")
            
            # 如果没有验证集或测试集，尝试从训练集分割
            if len(self.val_dataset) == 0:
                print("验证集为空，将从训练集分割验证集...")
                self.split_val_from_train()
        
        print("\n数据集加载完成!")
    
    def split_val_from_train(self):
        """从训练集分割验证集"""
        if len(self.train_dataset) == 0:
            print("错误: 训练集也为空，无法分割验证集")
            return
        
        print("从训练集分割验证集...")
        
        # 获取训练集大小
        train_size = len(self.train_dataset)
        
        # 计算分割大小（20%用于验证）
        val_size = int(train_size * 0.2)
        
        if val_size < 10:
            val_size = min(10, train_size)  # 至少10个样本
        
        # 随机选择验证集索引
        all_indices = np.arange(train_size)
        val_indices = np.random.choice(all_indices, size=val_size, replace=False)
        train_indices = np.setdiff1d(all_indices, val_indices)
        
        # 创建子数据集
        from torch.utils.data import Subset
        
        # 备份原始训练集
        original_train_dataset = self.train_dataset
        
        # 创建新的训练集和验证集
        self.train_dataset = Subset(original_train_dataset, train_indices)
        self.val_dataset = Subset(original_train_dataset, val_indices)
        
        print(f"  原始训练集大小: {train_size}")
        print(f"  新训练集大小: {len(self.train_dataset)}")
        print(f"  验证集大小: {len(self.val_dataset)}")
    
    def create_dataloaders(self):
        """创建数据加载器"""
        num_workers = min(4, os.cpu_count() or 1)
        
        # 训练数据加载器
        if len(self.train_dataset) > 0:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            self.train_loader = None
        
        # 验证数据加载器
        if len(self.val_dataset) > 0:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            self.val_loader = None
        
        # 测试数据加载器
        if len(self.test_dataset) > 0:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            self.test_loader = None
        
        # 打印数据加载器信息
        self.print_dataloader_info()
    
    def print_dataloader_info(self):
        """打印数据加载器信息"""
        print(f"\n数据加载器信息:")
        
        loaders = {
            '训练集': self.train_loader,
            '验证集': self.val_loader,
            '测试集': self.test_loader
        }
        
        for name, loader in loaders.items():
            if loader is not None:
                dataset = loader.dataset
                if hasattr(dataset, 'dataset'):  # 如果是Subset
                    dataset = dataset.dataset
                
                print(f"  {name}: {len(loader)} batches, {len(dataset)} samples")
            else:
                print(f"  {name}: 无数据")
    
    def get_class_weights(self):
        """计算类别权重（用于处理不平衡数据）"""
        if self.train_loader is None or len(self.train_dataset) == 0:
            print("警告: 训练集为空，无法计算类别权重")
            return None
        
        # 获取所有标签
        all_labels = []
        
        if hasattr(self.train_dataset, 'dataset'):  # 如果是Subset
            # 获取完整的标签
            if hasattr(self.train_dataset.dataset, 'labels'):
                all_labels = self.train_dataset.dataset.labels
                # 只取训练集的标签
                all_labels = all_labels[self.train_dataset.indices]
        elif hasattr(self.train_dataset, 'labels'):
            all_labels = self.train_dataset.labels
        
        # 过滤无效标签
        all_labels = [label for label in all_labels if label >= 0]
        
        if not all_labels:
            print("警告: 没有有效的标签数据")
            return None
        
        # 计算类别权重
        from sklearn.utils import class_weight
        
        try:
            class_weights = class_weight.compute_class_weight(
                'balanced',
                classes=np.arange(config.NUM_CLASSES),
                y=all_labels
            )
            
            # 转换为字典
            class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
            
            print(f"类别权重: {class_weights_dict}")
            return class_weights_dict
            
        except Exception as e:
            print(f"计算类别权重失败: {e}")
            return None
    
    def visualize_samples(self, num_samples=5):
        """可视化一些样本"""
        import matplotlib.pyplot as plt
        
        # 尝试从训练集获取样本
        if self.train_loader is not None:
            try:
                # 获取一个批次的数据
                images, labels = next(iter(self.train_loader))
                
                # 选择要显示的样本
                num_samples = min(num_samples, len(images))
                
                fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
                if num_samples == 1:
                    axes = [axes]
                
                for i in range(num_samples):
                    # 获取图像和标签
                    img = images[i].squeeze().numpy()
                    label_idx = torch.argmax(labels[i]).item()
                    label_name = self.config.CLASS_NAMES[label_idx]
                    
                    # 显示图像
                    axes[i].imshow(img, cmap='viridis')
                    axes[i].set_title(f"类别: {label_name}")
                    axes[i].axis('off')
                
                plt.suptitle('训练集样本示例', fontsize=16)
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"可视化样本失败: {e}")
        else:
            print("训练集为空，无法可视化样本")


def create_dataloaders():
    """创建数据加载器（便捷函数）"""
    data_module = FCRGDataModule(config)
    
    # 检查数据加载器是否创建成功
    if data_module.train_loader is None:
        print("错误: 训练数据加载器创建失败")
        sys.exit(1)
    
    if data_module.val_loader is None:
        print("警告: 验证数据加载器创建失败，将使用部分训练数据作为验证集")
    
    return data_module.train_loader, data_module.val_loader, data_module.test_loader


# 测试函数
def test_data_loading():
    """测试数据加载"""
    print("\n" + "="*60)
    print("测试数据加载")
    print("="*60)
    
    # 创建数据模块
    data_module = FCRGDataModule(config)
    
    # 检查数据加载器
    if data_module.train_loader is not None:
        print("\n测试训练数据加载器...")
        images, labels = next(iter(data_module.train_loader))
        print(f"  批次图像形状: {images.shape}")
        print(f"  批次标签形状: {labels.shape}")
        print(f"  训练集总批次数: {len(data_module.train_loader)}")
    
    if data_module.val_loader is not None:
        print("\n测试验证数据加载器...")
        images, labels = next(iter(data_module.val_loader))
        print(f"  批次图像形状: {images.shape}")
        print(f"  批次标签形状: {labels.shape}")
        print(f"  验证集总批次数: {len(data_module.val_loader)}")
    
    if data_module.test_loader is not None:
        print("\n测试测试数据加载器...")
        images, labels = next(iter(data_module.test_loader))
        print(f"  批次图像形状: {images.shape}")
        print(f"  批次标签形状: {labels.shape}")
        print(f"  测试集总批次数: {len(data_module.test_loader)}")
    
    # 计算类别权重
    class_weights = data_module.get_class_weights()
    
    # 可视化样本（可选）
    if config.VISUALIZE_TRAINING:
        data_module.visualize_samples(3)
    
    print("\n数据加载测试完成!")
    return data_module


if __name__ == "__main__":
    # 测试数据加载
    test_data_loading()