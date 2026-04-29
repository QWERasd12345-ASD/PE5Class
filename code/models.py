#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
胶囊网络模型定义
包含：胶囊层、主胶囊层、解码器和完整胶囊网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

class Squash(nn.Module):
    """Squash激活函数 - 将向量长度压缩到[0,1)之间"""
    def __init__(self, epsilon=1e-7):
        super(Squash, self).__init__()
        self.epsilon = epsilon

    def forward(self, vectors):
        norm = torch.norm(vectors, dim=-1, keepdim=True)
        scale = norm**2 / (1 + norm**2) / (norm + self.epsilon)
        squashed = scale * vectors
        return squashed


class PrimaryCapsuleLayer(nn.Module):
    """主胶囊层 - 将卷积特征转换为初始胶囊向量"""
    def __init__(self, in_channels, out_capsules, dim_capsules, kernel_size=9, stride=2):
        super(PrimaryCapsuleLayer, self).__init__()
        self.out_capsules = out_capsules
        self.dim_capsules = dim_capsules

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_capsules * dim_capsules,
            kernel_size=kernel_size,
            stride=stride,
            padding=0
        )
        self.relu = nn.ReLU()
        self.squash = Squash()
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x)
        x = self.relu(x)
        height = x.size(2)
        width = x.size(3)
        x = x.view(batch_size, self.out_capsules, self.dim_capsules, height, width)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(batch_size, -1, self.dim_capsules)
        x = self.squash(x)
        return x


class CapsuleLayer(nn.Module):
    """胶囊层 - 包含动态路由算法"""
    def __init__(self, num_input_capsules, dim_input_capsules,
                 num_output_capsules, dim_output_capsules, num_routing=3):
        super(CapsuleLayer, self).__init__()
        self.num_input_capsules = num_input_capsules
        self.dim_input_capsules = dim_input_capsules
        self.num_output_capsules = num_output_capsules
        self.dim_output_capsules = dim_output_capsules
        self.num_routing = num_routing

        self.W = nn.Parameter(
            torch.randn(1, num_input_capsules, num_output_capsules,
                        dim_output_capsules, dim_input_capsules)
        )
        self._init_weights()
        self.squash = Squash()

    def _init_weights(self):
        nn.init.xavier_normal_(self.W)

    def forward(self, x):
        batch_size = x.size(0)
        num_input_capsules = x.size(1)
        dim_input_capsules = x.size(2)

        # 扩展 W 到 batch_size
        W = self.W.expand(batch_size, num_input_capsules, self.num_output_capsules,
                          self.dim_output_capsules, dim_input_capsules)

        # 输入形状: [batch, in_caps, in_dim]
        x = x.unsqueeze(2).unsqueeze(4)  # [batch, in_caps, 1, in_dim, 1]

        # 计算 u_hat: [batch, in_caps, out_caps, out_dim]
        u_hat = torch.matmul(W, x).squeeze(4)  # [batch, in_caps, out_caps, out_dim]

        b_ij = torch.zeros(batch_size, num_input_capsules,
                          self.num_output_capsules, 1, device=x.device)

        for r in range(self.num_routing):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1)  # [batch, out_caps, out_dim]
            v_j = self.squash(s_j)

            if r < self.num_routing - 1:
                agreement = (u_hat * v_j.unsqueeze(1)).sum(dim=-1, keepdim=True)
                b_ij = b_ij + agreement

        return v_j


class FCGRDecoder(nn.Module):
    """FCGR解码器 - 从胶囊向量重构FCGR图像"""
    def __init__(self, input_dim, output_channels=1, output_size=(64, 64), dropout_rate=0.2):
        super(FCGRDecoder, self).__init__()
        self.output_size = output_size
        self.dense_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 8*8*32),
            nn.ReLU()
        )
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.final_conv = nn.Conv2d(8, output_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for layer in self.dense_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        for layer in self.deconv_layers:
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.final_conv.weight)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)

    def forward(self, x):
        x = self.dense_layers(x)
        x = x.view(-1, 32, 8, 8)
        x = self.deconv_layers(x)
        x = self.sigmoid(self.final_conv(x))
        return x


class DNACapsuleNetwork(nn.Module):
    """DNA胶囊网络 - 完整的FCGR分类模型（支持双通道输入）"""
    def __init__(self, config):
        super(DNACapsuleNetwork, self).__init__()
        self.config = config
        
        # 两个独立的特征提取器，分别处理原始序列和反向互补链
        self.feature_extractor_original = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=9, padding=4),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.Conv2d(256, 256, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.feature_extractor_reverse = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=9, padding=4),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.Conv2d(256, 256, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.primary_caps = PrimaryCapsuleLayer(
            in_channels=512,  # 两个256通道拼接
            out_capsules=config.PRIMARY_CAPSULES,
            dim_capsules=config.PRIMARY_DIM,
            kernel_size=9,
            stride=2
        )

        # 用一个固定的假输入计算主胶囊输出的尺寸
        with torch.no_grad():
            dummy_x = torch.randn(1, 2, 64, 64)
            dummy_features = self._extract_features(dummy_x)
            dummy_primary = self.primary_caps(dummy_features)
            num_input_capsules = dummy_primary.size(1)
            dim_input_capsules = dummy_primary.size(2)

        self.digit_caps = CapsuleLayer(
            num_input_capsules=num_input_capsules,
            dim_input_capsules=dim_input_capsules,
            num_output_capsules=config.NUM_CLASSES,
            dim_output_capsules=config.DIGIT_DIM,
            num_routing=config.NUM_ROUTING
        )
        self.decoder = FCGRDecoder(
            input_dim=config.NUM_CLASSES * config.DIGIT_DIM,
            output_channels=2,  # 输出双通道
            output_size=(64, 64),
            dropout_rate=0.2
        )
    
    def _extract_features(self, x):
        """
        从双通道输入中提取特征并拼接
        
        参数:
            x: 输入张量，形状为 (batch_size, 2, 64, 64)
               第一通道是原始序列的FCGR，第二通道是反向互补链的FCGR
        
        返回:
            拼接后的特征，形状为 (batch_size, 512, H, W)
        """
        # 分离两个通道
        x_original = x[:, 0:1, :, :]  # (batch_size, 1, 64, 64)
        x_reverse = x[:, 1:2, :, :]    # (batch_size, 1, 64, 64)
        
        # 分别提取特征
        features_original = self.feature_extractor_original(x_original)
        features_reverse = self.feature_extractor_reverse(x_reverse)
        
        # 拼接特征
        features = torch.cat([features_original, features_reverse], dim=1)  # (batch_size, 512, H, W)
        
        return features

    def forward(self, x, labels=None):
        # 提取特征
        features = self._extract_features(x)
        
        # 胶囊网络处理
        primary_caps = self.primary_caps(features)
        digit_caps = self.digit_caps(primary_caps)
        class_probs = torch.norm(digit_caps, dim=-1)

        reconstruction = None
        if labels is not None or self.training:
            reconstruction = self.reconstruct(digit_caps, labels)

        return class_probs, digit_caps, reconstruction

    def reconstruct(self, digit_caps, labels=None):
        batch_size = digit_caps.size(0)
        if labels is not None:
            labels_expanded = labels.unsqueeze(-1).expand(-1, -1, self.config.DIGIT_DIM)
            masked_caps = digit_caps * labels_expanded
        else:
            digit_caps_norm = torch.norm(digit_caps, dim=-1)
            pred_labels = torch.argmax(digit_caps_norm, dim=1)
            pred_labels_onehot = F.one_hot(pred_labels, self.config.NUM_CLASSES).float()
            pred_labels_expanded = pred_labels_onehot.unsqueeze(-1).expand(-1, -1, self.config.DIGIT_DIM)
            masked_caps = digit_caps * pred_labels_expanded
        flattened_caps = masked_caps.view(batch_size, -1)
        reconstruction = self.decoder(flattened_caps)
        return reconstruction

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            class_probs, digit_caps, _ = self.forward(x)
            predictions = torch.argmax(class_probs, dim=1)
            return predictions, class_probs, digit_caps

    def get_capsule_activations(self, x):
        self.eval()
        with torch.no_grad():
            class_probs, digit_caps, _ = self.forward(x)
            capsule_activations = torch.norm(digit_caps, dim=-1)
            return capsule_activations.cpu().numpy(), digit_caps.cpu().numpy()


def create_model(config):
    model = DNACapsuleNetwork(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n" + "="*50)
    print("胶囊网络模型信息")
    print("="*50)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print("\n模型结构:")
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {module.__class__.__name__} ({num_params:,} 参数)")
    print("="*50)
    return model


if __name__ == "__main__":
    print("测试胶囊网络模型（双通道输入）...")
    model = create_model(config)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 2, 64, 64).to(config.DEVICE)
    dummy_labels = torch.tensor([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0]
    ], dtype=torch.float32).to(config.DEVICE)
    class_probs, digit_caps, reconstruction = model(dummy_input, dummy_labels)
    print(f"类别概率形状: {class_probs.shape}")
    print(f"数字胶囊形状: {digit_caps.shape}")
    print(f"重构图像形状: {reconstruction.shape}")