#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主训练程序 - 修复版本，确保正确使用独立验证数据
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from data_loader import FCRGDataModule
from models import create_model
from losses import create_loss_function
from trainer import create_trainer
from evaluator import create_evaluator
from utils import (plot_training_history, visualize_predictions,
                  analyze_capsule_activations, save_results)


# 自动设置设备为 GPU（如果可用）
config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练DNA胶囊网络')

    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test', 'predict', 'analyze', 'data_check'],
                       help='运行模式: train(训练), test(测试), predict(预测), analyze(分析), data_check(数据检查)')

    parser.add_argument('--checkpoint', type=str, default=None,
                       help='检查点路径（用于测试或继续训练）')

    parser.add_argument('--epochs', type=int, default=None,
                       help='训练epoch数（覆盖配置）')

    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小（覆盖配置）')

    parser.add_argument('--learning_rate', type=float, default=None,
                       help='学习率（覆盖配置）')

    parser.add_argument('--visualize', action='store_true',
                       help='是否可视化结果')

    parser.add_argument('--save_dir', type=str, default=None,
                       help='结果保存目录（覆盖配置）')

    parser.add_argument('--data_dir', type=str, default=None,
                       help='数据目录（覆盖配置）')

    parser.add_argument('--train_data', type=str, default=None,
                       help='训练数据路径（覆盖配置）')

    parser.add_argument('--val_data', type=str, default=None,
                       help='验证数据路径（覆盖配置）')

    parser.add_argument('--test_data', type=str, default=None,
                       help='测试数据路径（覆盖配置）')

    return parser.parse_args()


def update_config_from_args(args, config):
    """根据命令行参数更新配置"""
    if args.epochs is not None:
        config.EPOCHS = args.epochs
        print(f"更新配置: EPOCHS = {args.epochs}")

    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
        print(f"更新配置: BATCH_SIZE = {args.batch_size}")

    if args.learning_rate is not None:
        config.LEARNING_RATE = args.learning_rate
        print(f"更新配置: LEARNING_RATE = {args.learning_rate}")

    if args.save_dir is not None:
        config.OUTPUT_DIR = args.save_dir
        config.MODEL_DIR = os.path.join(args.save_dir, "models")
        config.LOG_DIR = os.path.join(args.save_dir, "logs")
        config.CHECKPOINT_DIR = os.path.join(args.save_dir, "checkpoints")

        # 重新创建目录
        dirs = [config.OUTPUT_DIR, config.MODEL_DIR, config.LOG_DIR, config.CHECKPOINT_DIR]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

        print(f"更新配置: 输出目录 = {args.save_dir}")

    if args.data_dir is not None:
        config.TRAIN_FCGR_PATH = os.path.join(args.data_dir, "fcgr_output", "train", "fcgr.npz")
        config.VAL_FCGR_PATH = os.path.join(args.data_dir, "fcgr_output", "val", "fcgr.npz")
        config.TEST_FCGR_PATH = os.path.join(args.data_dir, "fcgr_output", "test", "fcgr.npz")
        print(f"更新配置: 数据目录 = {args.data_dir}")

    # 直接指定数据路径
    if args.train_data is not None:
        config.TRAIN_FCGR_PATH = args.train_data
        print(f"更新配置: 训练数据路径 = {args.train_data}")

    if args.val_data is not None:
        config.VAL_FCGR_PATH = args.val_data
        print(f"更新配置: 验证数据路径 = {args.val_data}")

    if args.test_data is not None:
        config.TEST_FCGR_PATH = args.test_data
        print(f"更新配置: 测试数据路径 = {args.test_data}")


def data_check_mode():
    """数据检查模式"""
    print("\n" + "="*60)
    print("数据检查模式")
    print("="*60)

    data_module = FCRGDataModule(config)

    print("\n数据加载器信息:")
    print(f"  训练集: {len(data_module.train_loader) if data_module.train_loader else 0} batches")
    print(f"  验证集: {len(data_module.val_loader) if data_module.val_loader else 0} batches")
    print(f"  测试集: {len(data_module.test_loader) if data_module.test_loader else 0} batches")

    if data_module.train_loader is not None:
        print("\n测试训练数据加载...")
        try:
            images, labels = next(iter(data_module.train_loader))
            print(f"  图像形状: {images.shape}")
            print(f"  标签形状: {labels.shape}")
            print(f"  图像范围: [{images.min():.3f}, {images.max():.3f}]")
        except Exception as e:
            print(f"  训练数据加载失败: {e}")

    if data_module.val_loader is not None:
        print("\n测试验证数据加载...")
        try:
            images, labels = next(iter(data_module.val_loader))
            print(f"  图像形状: {images.shape}")
            print(f"  标签形状: {labels.shape}")
            print(f"  图像范围: [{images.min():.3f}, {images.max():.3f}]")
        except Exception as e:
            print(f"  验证数据加载失败: {e}")

    if data_module.test_loader is not None:
        print("\n测试测试数据加载...")
        try:
            images, labels = next(iter(data_module.test_loader))
            print(f"  图像形状: {images.shape}")
            print(f"  标签形状: {labels.shape}")
            print(f"  图像范围: [{images.min():.3f}, {images.max():.3f}]")
        except Exception as e:
            print(f"  测试数据加载失败: {e}")

    print("\n数据检查完成!")


def train_mode(args):
    """训练模式"""
    print("\n" + "="*60)
    print("训练模式")
    print("="*60)

    data_module = FCRGDataModule(config)
    train_loader = data_module.train_loader
    val_loader = data_module.val_loader
    test_loader = data_module.test_loader

    if train_loader is None:
        print("错误: 训练数据加载失败")
        return

    print(f"  训练集: {len(train_loader)} batches")
    print(f"  验证集: {len(val_loader) if val_loader else 0} batches")
    print(f"  测试集: {len(test_loader) if test_loader else 0} batches")

    class_weights = data_module.get_class_weights()

    # 创建模型并移动到 GPU
    model = create_model(config).to(config.DEVICE)

    trainer = create_trainer(model, train_loader, class_weights)

    if args.checkpoint and os.path.exists(args.checkpoint):
        from utils import load_checkpoint
        load_checkpoint(args.checkpoint, model, trainer.optimizer, trainer.scheduler)

    history = trainer.train(train_loader, val_loader, epochs=config.EPOCHS)

    if args.visualize or config.VISUALIZE_TRAINING:
        history_path = os.path.join(config.run_dir, "training_history.png")
        plot_training_history(history, save_path=history_path)

    if test_loader is not None:
        results = trainer.evaluate(test_loader)
        if results:
            results_path = os.path.join(config.run_dir, "evaluation_results.json")
            save_results(results, results_path)
            print(f"\n测试结果:")
            print(f"  测试准确率: {results['test_acc']:.2f}%")
            print(f"  测试损失: {results['test_loss']:.4f}")
    else:
        print("\n6. 警告: 测试集为空，跳过评估")

    if (args.visualize or config.VISUALIZE_PREDICTIONS) and test_loader is not None:
        visualize_dir = os.path.join(config.run_dir, "visualizations")
        try:
            visualize_predictions(model, test_loader,
                                  num_samples=config.NUM_VISUALIZE_SAMPLES,
                                  save_dir=visualize_dir)
        except Exception as e:
            print(f"可视化预测结果失败: {e}")

        print("\n8. 分析胶囊激活模式...")
        try:
            analyze_capsule_activations(model, test_loader, save_dir=visualize_dir)
        except Exception as e:
            print(f"分析胶囊激活失败: {e}")

    print("\n训练完成!")
    print(f"所有结果保存在: {config.run_dir}")


def test_mode(args):
    """测试模式"""
    print("\n" + "="*60)
    print("测试模式")
    print("="*60)

    if args.checkpoint is None:
        args.checkpoint = config.get_model_path("best_model.pth")
        print(f"使用默认检查点: {args.checkpoint}")

    if not os.path.exists(args.checkpoint):
        print(f"错误: 检查点不存在 - {args.checkpoint}")
        return

    data_module = FCRGDataModule(config)
    test_loader = data_module.test_loader

    if test_loader is None:
        print("错误: 测试数据加载失败")
        return

    print(f"测试集: {len(test_loader)} batches")

    model = create_model(config).to(config.DEVICE)
    checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    evaluator = create_evaluator(model)
    results = evaluator.evaluate(test_loader)

    print("\n5. 详细评估报告:")
    evaluator.print_detailed_report()

    if args.visualize:
        print("\n6. 可视化结果...")
        visualize_dir = os.path.join(config.OUTPUT_DIR, "test_results")
        try:
            evaluator.visualize_results(save_dir=visualize_dir)
        except Exception as e:
            print(f"可视化结果失败: {e}")

        try:
            visualize_predictions(model, test_loader,
                                  num_samples=config.NUM_VISUALIZE_SAMPLES,
                                  save_dir=visualize_dir)
        except Exception as e:
            print(f"可视化预测样本失败: {e}")

        print("\n7. 分析错误分类...")
        try:
            evaluator.analyze_misclassifications(test_loader, top_k=20)
        except Exception as e:
            print(f"分析错误分类失败: {e}")

        print("\n8. 分析胶囊激活模式...")
        try:
            analyze_capsule_activations(model, test_loader, save_dir=visualize_dir)
        except Exception as e:
            print(f"分析胶囊激活失败: {e}")

    print("\n9. 保存结果...")
    results_dir = os.path.join(config.OUTPUT_DIR, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "evaluation_results.json")
    evaluator.save_results(results_path)
    print(f"\n测试完成! 结果保存在: {results_dir}")


def predict_mode(args):
    """预测模式"""
    print("\n" + "="*60)
    print("预测模式")
    print("="*60)

    if args.checkpoint is None:
        args.checkpoint = config.get_model_path("best_model.pth")
        print(f"使用默认检查点: {args.checkpoint}")

    if not os.path.exists(args.checkpoint):
        print(f"错误: 检查点不存在 - {args.checkpoint}")
        return

    model = create_model(config).to(config.DEVICE)
    checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    data_module = FCRGDataModule(config)
    test_loader = data_module.test_loader

    if test_loader is None:
        print("错误: 测试数据加载失败")
        return

    print(f"测试集: {len(test_loader)} batches")

    all_predictions = []
    all_probabilities = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            predictions, digit_caps, _ = model(data)
            _, predicted = predictions.max(1)
            _, target_labels = target.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.append(predictions.cpu().numpy())
            all_targets.extend(target_labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_probabilities = np.vstack(all_probabilities) if all_probabilities else np.array([])
    all_targets = np.array(all_targets)

    results_dir = os.path.join(config.OUTPUT_DIR, "predictions")
    os.makedirs(results_dir, exist_ok=True)

    import pandas as pd
    results_df = pd.DataFrame({
        'true_label': [config.CLASS_NAMES[i] for i in all_targets],
        'predicted_label': [config.CLASS_NAMES[i] for i in all_predictions],
        'is_correct': all_targets == all_predictions
    })
    for i, class_name in enumerate(config.CLASS_NAMES):
        if all_probabilities.size > 0:
            results_df[f'prob_{class_name}'] = all_probabilities[:, i]

    csv_path = os.path.join(results_dir, "predictions.csv")
    results_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"预测结果已保存: {csv_path}")

    print("\n5. 预测结果统计:")
    if len(all_targets) > 0:
        accuracy = np.mean(all_targets == all_predictions)
        print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\n各类别预测统计:")
        for i, class_name in enumerate(config.CLASS_NAMES):
            class_mask = all_targets == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(all_predictions[class_mask] == all_targets[class_mask])
                print(f"  {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%) - "
                      f"{np.sum(class_mask)} 个样本")
    else:
        print("没有预测结果")

    if args.visualize and test_loader is not None:
        print("\n6. 可视化预测结果...")
        visualize_predictions(model, test_loader,
                              num_samples=min(10, len(test_loader.dataset)),
                              save_dir=results_dir)

    print(f"\n预测完成! 结果保存在: {results_dir}")


def analyze_mode(args):
    """分析模式 - 深入分析模型和特征"""
    print("\n" + "="*60)
    print("分析模式")
    print("="*60)

    if args.checkpoint is None:
        args.checkpoint = config.get_model_path("best_model.pth")
        print(f"使用默认检查点: {args.checkpoint}")

    if not os.path.exists(args.checkpoint):
        print(f"错误: 检查点不存在 - {args.checkpoint}")
        return

    model = create_model(config).to(config.DEVICE)
    checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    data_module = FCRGDataModule(config)
    test_loader = data_module.test_loader

    if test_loader is None:
        print("错误: 测试数据加载失败")
        return

    print(f"测试集: {len(test_loader)} batches")

    analyze_dir = os.path.join(config.OUTPUT_DIR, "analysis")
    try:
        analyze_capsule_activations(model, test_loader, save_dir=analyze_dir)
    except Exception as e:
        print(f"分析胶囊激活失败: {e}")

    print(f"\n分析完成! 结果保存在: {analyze_dir}")


def main():
    """主函数"""
    args = parse_args()
    update_config_from_args(args, config)

    if args.mode == 'data_check':
        data_check_mode()
    elif args.mode == 'train':
        train_mode(args)
    elif args.mode == 'test':
        test_mode(args)
    elif args.mode == 'predict':
        predict_mode(args)
    elif args.mode == 'analyze':
        analyze_mode(args)
    else:
        print(f"未知模式: {args.mode}")
        print("可用模式: train, test, predict, analyze, data_check")


if __name__ == "__main__":
    print("\n" + "="*60)
    print(f"DNA序列胶囊网络分类器")
    print(f"项目: {config.PROJECT_NAME}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("使用CPU")

    try:
        main()
        print("\n程序执行完成!")
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()

    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)