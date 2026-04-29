#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置文件 - 定义所有超参数和路径配置
"""

import os
import torch
from datetime import datetime

class Config:
    """配置文件类"""
    
    # 基本设置
    PROJECT_NAME = "DNA_Capsule_Network"
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据路径
    DATA_DIR = "."  # 修改为你的数据路径
    TRAIN_FCGR_PATH = os.path.join(DATA_DIR, "fcgr_output", "train", "fcgr.npz")
    VAL_FCGR_PATH = os.path.join(DATA_DIR, "fcgr_output", "val", "fcgr.npz")
    TEST_FCGR_PATH = os.path.join(DATA_DIR, "fcgr_output", "test", "fcgr.npz")
    
    # 输出目录
    OUTPUT_DIR = "outputs"
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    
    # 类别映射（根据你的5个类别）
    CLASS_NAMES = [
        'prokaryotic_virus',  # 原核病毒
        'eukaryotic_virus',   # 真核病毒
        'prokaryote',         # 原核生物
        'eukaryote',          # 真核生物
        'plasmid'            # 质粒
    ]
    
    # 模型参数
    NUM_CLASSES = 5
    INPUT_SHAPE = (1, 64, 64)  # FCGR图像形状
    
    # 胶囊网络参数
    PRIMARY_CAPSULES = 32
    PRIMARY_DIM = 8
    DIGIT_DIM = 16
    NUM_ROUTING = 3
    RECONSTRUCTION_WEIGHT = 0.0005
    
    # 训练参数
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    
    # 优化器参数
    OPTIMIZER = 'adam'  # 'adam', 'sgd', 'adamw'
    MOMENTUM = 0.9  # SGD使用
    BETAS = (0.9, 0.999)  # Adam使用
    
    # 学习率调度
    SCHEDULER = 'plateau'  # 'plateau', 'cosine', 'step'
    LR_PATIENCE = 5
    LR_FACTOR = 0.5
    LR_MIN = 1e-6
    
    # 早停
    EARLY_STOPPING = True
    EARLY_STOP_PATIENCE = 30
    EARLY_STOP_MIN_DELTA = 0.001
    
    # 数据增强
    USE_AUGMENTATION = False
    AUGMENTATION_PROB = 0
    
    # 损失函数参数
    M_PLUS = 0.9
    M_MINUS = 0.1
    LAMBDA_VAL = 0.5
    
    # 评估参数
    EVAL_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix']
    
    # 日志记录
    LOG_INTERVAL = 10  # 每多少个batch记录一次
    SAVE_INTERVAL = 5  # 每多少个epoch保存一次检查点
    
    # 可视化
    VISUALIZE_TRAINING = True
    VISUALIZE_PREDICTIONS = True
    NUM_VISUALIZE_SAMPLES = 5
    
    def __init__(self):
        """初始化配置"""
        # 设置随机种子
        self.set_seed()
        
        # 创建输出目录
        self.create_directories()
        
        # 生成时间戳用于版本管理
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.LOG_DIR, f"run_{self.timestamp}")
        
        # 打印配置信息
        self.print_config()
    
    def set_seed(self):
        """设置随机种子"""
        import random
        import numpy as np
        
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.SEED)
            torch.cuda.manual_seed_all(self.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def create_directories(self):
        """创建必要的目录"""
        dirs = [self.OUTPUT_DIR, self.MODEL_DIR, self.LOG_DIR, self.CHECKPOINT_DIR]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"创建目录: {dir_path}")
    
    def print_config(self):
        """打印配置信息"""
        print("\n" + "="*50)
        print(f"{self.PROJECT_NAME} - 配置信息")
        print("="*50)
        
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_') and not callable(v)}
        
        for key, value in config_dict.items():
            if key.upper() == key:  # 只打印全大写的配置项
                print(f"{key}: {value}")
        
        print("="*50)
    
    def get_model_path(self, model_name="best_model.pth"):
        """获取模型保存路径"""
        return os.path.join(self.MODEL_DIR, model_name)
    
    def get_checkpoint_path(self, epoch):
        """获取检查点保存路径"""
        return os.path.join(self.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
    
    def get_log_file(self):
        """获取日志文件路径"""
        return os.path.join(self.run_dir, "training.log")
    
    def get_result_file(self):
        """获取结果文件路径"""
        return os.path.join(self.run_dir, "results.json")
    
    def get_visualization_dir(self):
        """获取可视化保存目录"""
        vis_dir = os.path.join(self.run_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        return vis_dir


# 创建全局配置实例
config = Config()