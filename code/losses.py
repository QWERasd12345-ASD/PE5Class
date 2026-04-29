#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
损失函数定义
包含：胶囊网络的Margin Loss、重构损失和总损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import config


class CapsuleMarginLoss(nn.Module):
    """胶囊网络Margin Loss - 分类损失"""
    
    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_val=0.5, class_weights=None):
        """
        初始化Margin Loss
        
        参数:
            m_plus: 正确类别的上界
            m_minus: 错误类别的下界
            lambda_val: 错误类别损失的权重
            class_weights: 类别权重字典
        """
        super(CapsuleMarginLoss, self).__init__()
        
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_val = lambda_val
        self.class_weights = class_weights
    
    def forward(self, y_true, y_pred):
        """
        计算Margin Loss
        
        参数:
            y_true: 真实标签 [batch_size, num_classes] (one-hot)
            y_pred: 预测值（胶囊模长） [batch_size, num_classes]
        
        返回:
            loss: 损失值
        """
        # 正确类别的损失: max(0, m_plus - ||v_k||)^2
        loss_pos = y_true * torch.clamp(self.m_plus - y_pred, min=0) ** 2
        
        # 错误类别的损失: max(0, ||v_k|| - m_minus)^2
        loss_neg = self.lambda_val * (1 - y_true) * torch.clamp(y_pred - self.m_minus, min=0) ** 2
        
        # 总损失
        loss = loss_pos + loss_neg
        
        # 应用类别权重（如果提供）
        if self.class_weights is not None:
            # 转换为张量
            weights = torch.tensor(list(self.class_weights.values()), 
                                  device=y_true.device, dtype=torch.float32)
            weights = weights.unsqueeze(0)  # [1, num_classes]
            loss = loss * weights
        
        # 求和并取平均
        loss = loss.sum(dim=1).mean()
        
        return loss


class FCGRReconstructionLoss(nn.Module):
    """FCGR重构损失 - 专门为FCGR数据设计"""
    
    def __init__(self, alpha_mse=0.6, alpha_ssim=0.3, alpha_freq=0.1):
        """
        初始化FCGR重构损失
        
        参数:
            alpha_mse: MSE损失的权重
            alpha_ssim: SSIM损失的权重
            alpha_freq: 频率分布损失的权重
        """
        super(FCGRReconstructionLoss, self).__init__()
        
        self.alpha_mse = alpha_mse
        self.alpha_ssim = alpha_ssim
        self.alpha_freq = alpha_freq
    
    def forward(self, original, reconstructed):
        """
        计算重构损失
        
        参数:
            original: 原始FCGR图像 [batch_size, 1, height, width]
            reconstructed: 重构的FCGR图像 [batch_size, 1, height, width]
        
        返回:
            loss: 重构损失
        """
        batch_size = original.size(0)
        
        # 1. 均方误差损失 (MSE)
        mse_loss = F.mse_loss(reconstructed, original)
        
        # 2. 结构相似性损失 (SSIM)
        ssim_loss = self._ssim_loss(original, reconstructed)
        
        # 3. 频率分布损失 (针对FCGR特性)
        freq_loss = self._frequency_loss(original, reconstructed)
        
        # 组合损失
        total_loss = (self.alpha_mse * mse_loss + 
                     self.alpha_ssim * ssim_loss + 
                     self.alpha_freq * freq_loss)
        
        return total_loss
    
    def _ssim_loss(self, img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
        """
        计算简化版的SSIM损失
        
        参数:
            img1, img2: 输入图像
            window_size: 窗口大小
            C1, C2: 稳定性常数
        
        返回:
            ssim_loss: 1 - SSIM
        """
        # 将图像归一化到[0, 1]
        img1_norm = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-7)
        img2_norm = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-7)
        
        # 计算均值
        mu1 = F.avg_pool2d(img1_norm, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(img2_norm, window_size, stride=1, padding=window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # 计算方差
        sigma1_sq = F.avg_pool2d(img1_norm * img1_norm, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2_norm * img2_norm, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(img1_norm * img2_norm, window_size, stride=1, padding=window_size//2) - mu1_mu2
        
        # 计算SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # 返回1 - SSIM作为损失
        return 1 - ssim_map.mean()
    
    def _frequency_loss(self, original, reconstructed):
        """
        计算频率分布损失
        
        参数:
            original: 原始FCGR图像
            reconstructed: 重构的FCGR图像
        
        返回:
            freq_loss: 频率分布损失
        """
        batch_size = original.size(0)
        
        # 展平图像
        original_flat = original.view(batch_size, -1)
        reconstructed_flat = reconstructed.view(batch_size, -1)
        
        # 计算频率分布（softmax归一化）
        original_probs = F.softmax(original_flat, dim=1)
        reconstructed_probs = F.softmax(reconstructed_flat, dim=1)
        
        # 计算KL散度
        kl_div = F.kl_div(reconstructed_probs.log(), original_probs, reduction='batchmean')
        
        return kl_div


class CapsuleTotalLoss(nn.Module):
    """胶囊网络总损失 - 组合分类损失和重构损失"""
    
    def __init__(self, margin_loss_params=None, recon_loss_params=None, 
                 reconstruction_weight=0.0005):
        """
        初始化总损失
        
        参数:
            margin_loss_params: Margin Loss的参数
            recon_loss_params: 重构损失的参数
            reconstruction_weight: 重构损失的权重
        """
        super(CapsuleTotalLoss, self).__init__()
        
        # 默认参数
        if margin_loss_params is None:
            margin_loss_params = {
                'm_plus': config.M_PLUS,
                'm_minus': config.M_MINUS,
                'lambda_val': config.LAMBDA_VAL
            }
        
        if recon_loss_params is None:
            recon_loss_params = {
                'alpha_mse': 0.6,
                'alpha_ssim': 0.3,
                'alpha_freq': 0.1
            }
        
        # 创建损失函数
        self.margin_loss = CapsuleMarginLoss(**margin_loss_params)
        self.recon_loss = FCGRReconstructionLoss(**recon_loss_params)
        self.reconstruction_weight = reconstruction_weight
    
    def forward(self, y_true, y_pred, original_images, reconstructed_images):
        """
        计算总损失
        
        参数:
            y_true: 真实标签
            y_pred: 预测值
            original_images: 原始图像
            reconstructed_images: 重构图像
        
        返回:
            total_loss: 总损失
            class_loss: 分类损失
            recon_loss: 重构损失
        """
        # 计算分类损失
        class_loss = self.margin_loss(y_true, y_pred)
        
        # 计算重构损失
        recon_loss = self.recon_loss(original_images, reconstructed_images)
        
        # 计算总损失
        total_loss = class_loss + self.reconstruction_weight * recon_loss
        
        return total_loss, class_loss, recon_loss


class MultiTaskLoss(nn.Module):
    """多任务损失 - 可扩展支持其他辅助任务"""
    
    def __init__(self, task_weights=None):
        """
        初始化多任务损失
        
        参数:
            task_weights: 各任务损失的权重字典
        """
        super(MultiTaskLoss, self).__init__()
        
        # 默认权重
        if task_weights is None:
            task_weights = {
                'classification': 1.0,
                'reconstruction': 0.0005
            }
        
        self.task_weights = task_weights
    
    def forward(self, losses_dict):
        """
        计算多任务损失
        
        参数:
            losses_dict: 损失字典，键为任务名，值为损失值
        
        返回:
            total_loss: 总损失
            losses_dict: 更新后的损失字典（包含加权损失）
        """
        total_loss = 0
        weighted_losses = {}
        
        for task_name, loss_value in losses_dict.items():
            if task_name in self.task_weights:
                weight = self.task_weights[task_name]
                weighted_loss = weight * loss_value
                weighted_losses[f'{task_name}_weighted'] = weighted_loss
                total_loss += weighted_loss
            else:
                weighted_losses[task_name] = loss_value
        
        weighted_losses['total'] = total_loss
        
        return total_loss, weighted_losses


def create_loss_function(config, class_weights=None):
    """创建损失函数"""
    
    # 更新Margin Loss的参数
    margin_loss_params = {
        'm_plus': config.M_PLUS,
        'm_minus': config.M_MINUS,
        'lambda_val': config.LAMBDA_VAL,
        'class_weights': class_weights
    }
    
    # 创建总损失函数
    loss_fn = CapsuleTotalLoss(
        margin_loss_params=margin_loss_params,
        reconstruction_weight=config.RECONSTRUCTION_WEIGHT
    )
    
    print("\n损失函数信息:")
    print(f"  Margin Loss: m_plus={config.M_PLUS}, m_minus={config.M_MINUS}, lambda={config.LAMBDA_VAL}")
    print(f"  重构损失权重: {config.RECONSTRUCTION_WEIGHT}")
    
    if class_weights:
        print(f"  类别权重: {class_weights}")
    
    return loss_fn


if __name__ == "__main__":
    # 测试损失函数
    print("测试损失函数...")
    
    # 创建虚拟数据
    batch_size = 4
    num_classes = 5
    
    # 创建虚拟标签和预测
    y_true = torch.tensor([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0]
    ], dtype=torch.float32)
    
    y_pred = torch.randn(batch_size, num_classes)
    y_pred = torch.abs(y_pred)  # 确保为正数（模拟胶囊模长）
    
    # 创建虚拟图像
    original_images = torch.randn(batch_size, 1, 64, 64)
    reconstructed_images = torch.randn(batch_size, 1, 64, 64)
    
    # 测试Margin Loss
    margin_loss = CapsuleMarginLoss()
    margin_loss_value = margin_loss(y_true, y_pred)
    print(f"\nMargin Loss: {margin_loss_value.item():.4f}")
    
    # 测试重构损失
    recon_loss = FCGRReconstructionLoss()
    recon_loss_value = recon_loss(original_images, reconstructed_images)
    print(f"重构损失: {recon_loss_value.item():.4f}")
    
    # 测试总损失
    total_loss_fn = CapsuleTotalLoss(reconstruction_weight=0.0005)
    total_loss, class_loss, recon_loss = total_loss_fn(
        y_true, y_pred, original_images, reconstructed_images
    )
    print(f"总损失: {total_loss.item():.4f}")
    print(f"分类损失: {class_loss.item():.4f}")
    print(f"重构损失: {recon_loss.item():.4f}")
    
    # 测试多任务损失
    multi_task_loss = MultiTaskLoss()
    losses_dict = {
        'classification': class_loss,
        'reconstruction': recon_loss
    }
    total_multi_loss, weighted_losses = multi_task_loss(losses_dict)
    print(f"\n多任务总损失: {total_multi_loss.item():.4f}")
    for name, loss in weighted_losses.items():
        print(f"  {name}: {loss.item():.4f}")