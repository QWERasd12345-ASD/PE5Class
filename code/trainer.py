#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练器模块 - 修复版本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import numpy as np
import os
import time
import json
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import roc_auc_score

from config import config
from models import create_model
from losses import create_loss_function
from utils import Logger, save_checkpoint, load_checkpoint, EarlyStopping


class CapsuleNetTrainer:
    """胶囊网络训练器 - 修复版本"""

    def __init__(self, model, loss_fn, optimizer, device=None, logger=None):
        self.device = device if device else config.DEVICE
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.logger = logger if logger else Logger()

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'class_loss': [],
            'recon_loss': [],
            'learning_rate': []
        }

        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.best_model_state = None

        self._init_scheduler()
        self._init_early_stopping()
        self.print_trainer_info()

    def _init_scheduler(self):
        if config.SCHEDULER == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.LR_FACTOR,
                patience=config.LR_PATIENCE,
                min_lr=config.LR_MIN,
                verbose=True
            )
        elif config.SCHEDULER == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.EPOCHS,
                eta_min=config.LR_MIN
            )
        elif config.SCHEDULER == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.5
            )
        else:
            self.scheduler = None

    def _init_early_stopping(self):
        if config.EARLY_STOPPING:
            self.early_stopping = EarlyStopping(
                patience=config.EARLY_STOP_PATIENCE,
                min_delta=config.EARLY_STOP_MIN_DELTA,
                verbose=True
            )
        else:
            self.early_stopping = None

    def print_trainer_info(self):
        print("\n" + "="*50)
        print("训练器配置信息")
        print("="*50)
        print(f"设备: {self.device}")
        print(f"优化器: {self.optimizer.__class__.__name__}")
        print(f"学习率调度器: {self.scheduler.__class__.__name__ if self.scheduler else '无'}")
        print(f"早停: {'启用' if self.early_stopping else '禁用'}")
        print("="*50)

    def train_epoch(self, train_loader, epoch):
        self.model.train()

        total_loss = 0.0
        total_class_loss = 0.0
        total_recon_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_targets = []
        all_probabilities = []

        pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}/{config.EPOCHS}")

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            predictions, digit_caps, reconstruction = self.model(data, target)
            total_loss_batch, class_loss_batch, recon_loss_batch = self.loss_fn(
                target, predictions, data, reconstruction
            )
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += total_loss_batch.item()
            total_class_loss += class_loss_batch.item()
            total_recon_loss += recon_loss_batch.item()

            _, predicted = predictions.max(1)
            _, target_labels = target.max(1)
            total += target.size(0)
            correct += predicted.eq(target_labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target_labels.cpu().numpy())
            all_probabilities.append(predictions.detach().cpu().numpy())

            if batch_idx % config.LOG_INTERVAL == 0:
                batch_acc = 100. * correct / total
                pbar.set_postfix({
                    '损失': f'{total_loss/(batch_idx+1):.4f}',
                    '准确率': f'{batch_acc:.2f}%'
                })

        epoch_loss = total_loss / len(train_loader)
        epoch_class_loss = total_class_loss / len(train_loader)
        epoch_recon_loss = total_recon_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        current_lr = self.optimizer.param_groups[0]['lr']
        self.history['learning_rate'].append(current_lr)

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.vstack(all_probabilities)

        return epoch_loss, epoch_acc, epoch_class_loss, epoch_recon_loss, all_predictions, all_targets, all_probabilities

    def validate(self, val_loader, epoch):
        if val_loader is None:
            print(f"警告: 验证数据加载器为空，跳过验证")
            return 0.0, 0.0, 0.0, 0.0, None, None, None

        self.model.eval()

        total_loss = 0.0
        total_class_loss = 0.0
        total_recon_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"验证 Epoch {epoch+1}/{config.EPOCHS}")
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                predictions, digit_caps, reconstruction = self.model(data, target)
                total_loss_batch, class_loss_batch, recon_loss_batch = self.loss_fn(
                    target, predictions, data, reconstruction
                )
                total_loss += total_loss_batch.item()
                total_class_loss += class_loss_batch.item()
                total_recon_loss += recon_loss_batch.item()

                _, predicted = predictions.max(1)
                _, target_labels = target.max(1)
                total += target.size(0)
                correct += predicted.eq(target_labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target_labels.cpu().numpy())
                all_probabilities.append(predictions.cpu().numpy())

                batch_acc = 100. * correct / total
                pbar.set_postfix({
                    '损失': f'{total_loss/(len(val_loader)):.4f}',
                    '准确率': f'{batch_acc:.2f}%'
                })

        val_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_class_loss = total_class_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_recon_loss = total_recon_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_acc = 100. * correct / total if total > 0 else 0.0

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.vstack(all_probabilities) if all_probabilities else np.array([])

        return val_loss, val_acc, val_class_loss, val_recon_loss, all_predictions, all_targets, all_probabilities

    def train(self, train_loader, val_loader, epochs=None):
        if epochs is None:
            epochs = config.EPOCHS

        print(f"\n开始训练，共 {epochs} 个epoch...")
        print(f"训练集批次数: {len(train_loader) if train_loader else 0}")
        print(f"验证集批次数: {len(val_loader) if val_loader else 0}")

        start_time = time.time()

        for epoch in range(epochs):
            train_loss, train_acc, train_class_loss, train_recon_loss, train_predictions, train_targets, train_probabilities = self.train_epoch(
                train_loader, epoch
            )
            val_loss, val_acc, val_class_loss, val_recon_loss, val_predictions, val_targets, val_probabilities = self.validate(
                val_loader, epoch
            )

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['class_loss'].append(val_class_loss)
            self.history['recon_loss'].append(val_recon_loss)

            self._print_epoch_results(
                epoch, epochs, train_loss, train_acc, val_loss, val_acc,
                train_class_loss, train_recon_loss, train_predictions, train_targets, train_probabilities,
                val_predictions, val_targets, val_probabilities
            )

            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                class_loss=val_class_loss,
                recon_loss=val_recon_loss
            )

            if (epoch + 1) % config.SAVE_INTERVAL == 0:
                checkpoint_path = config.get_checkpoint_path(epoch + 1)
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    val_acc=val_acc,
                    path=checkpoint_path
                )
                print(f"检查点已保存: {checkpoint_path}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.best_model_state = self.model.state_dict().copy()
                best_model_path = config.get_model_path("best_model.pth")
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    val_acc=val_acc,
                    path=best_model_path
                )
                print(f"最佳模型已保存: {best_model_path} (准确率: {val_acc:.2f}%)")

            if self.early_stopping is not None:
                self.early_stopping(-val_acc)
                if self.early_stopping.early_stop:
                    print(f"\n早停触发！在epoch {epoch+1}停止训练")
                    break

        training_time = time.time() - start_time
        self._print_training_summary(training_time)

        final_model_path = config.get_model_path("final_model.pth")
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epochs,
            val_acc=self.best_val_acc,
            path=final_model_path
        )
        print(f"最终模型已保存: {final_model_path}")

        self._save_history()

        return self.history

    def _print_epoch_results(self, epoch, epochs, train_loss, train_acc,
                            val_loss, val_acc, class_loss, recon_loss,
                            train_predictions, train_targets, train_probabilities,
                            val_predictions, val_targets, val_probabilities):
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
        print(f"  验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")
        print(f"  分类损失: {class_loss:.4f} | 重构损失: {recon_loss:.4f}")
        print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")

        print(f"\n  训练集 - 五分类指标:")
        self._print_class_metrics(train_predictions, train_targets, train_probabilities)

        print(f"\n  验证集 - 五分类指标:")
        self._print_class_metrics(val_predictions, val_targets, val_probabilities)

    def _print_training_summary(self, training_time):
        print("\n" + "="*50)
        print("训练总结")
        print("="*50)
        print(f"训练时间: {training_time:.2f}秒 ({training_time/60:.2f}分钟)")
        print(f"最佳epoch: {self.best_epoch + 1}")
        print(f"最佳准确率: {self.best_val_acc:.2f}%")
        print("="*50)

    def _print_class_metrics(self, predictions, targets, probabilities):
        """打印五分类各个类别的指标"""
        if predictions is None or targets is None or len(predictions) == 0:
            print("    无数据")
            return

        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            from scipy.special import softmax

            probabilities_normalized = softmax(probabilities, axis=1)

            print(f"    {'类别':<20} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'AUC':<10}")
            print("    " + "-" * 70)

            class_aucs = []
            for i, class_name in enumerate(config.CLASS_NAMES):
                class_mask = targets == i
                if np.sum(class_mask) > 0:
                    class_precision = precision_score(targets, predictions, labels=[i], average=None, zero_division=0)[0]
                    class_recall = recall_score(targets, predictions, labels=[i], average=None, zero_division=0)[0]
                    class_f1 = f1_score(targets, predictions, labels=[i], average=None, zero_division=0)[0]

                    true_positives = np.sum((targets == i) & (predictions == i))
                    class_accuracy = true_positives / np.sum(targets == i)

                    try:
                        binary_targets = (targets == i).astype(int)
                        class_auc = roc_auc_score(binary_targets, probabilities_normalized[:, i])
                    except:
                        class_auc = 0.0

                    class_aucs.append(class_auc)

                    print(f"    {class_name:<20} {class_accuracy:<10.4f} {class_precision:<10.4f} "
                          f"{class_recall:<10.4f} {class_f1:<10.4f} {class_auc:<10.4f}")
                else:
                    print(f"    {class_name:<20} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

        except Exception as e:
            print(f"    计算指标失败: {e}")

    def _save_history(self):
        history_path = os.path.join(config.run_dir, "training_history.json")
        history_serializable = {}
        for key, value in self.history.items():
            if isinstance(value, list) and len(value) > 0:
                history_serializable[key] = []
                for v in value:
                    if torch.is_tensor(v):
                        history_serializable[key].append(v.item())
                    else:
                        history_serializable[key].append(v)
            else:
                history_serializable[key] = value
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        print(f"训练历史已保存: {history_path}")

    def evaluate(self, test_loader):
        if test_loader is None:
            print("警告: 测试数据加载器为空，跳过评估")
            return None

        self.model.eval()

        total_loss = 0.0
        total_class_loss = 0.0
        total_recon_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            print("\n评估模型...")
            pbar = tqdm(test_loader, desc="测试")
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                predictions, digit_caps, reconstruction = self.model(data, target)
                total_loss_batch, class_loss_batch, recon_loss_batch = self.loss_fn(
                    target, predictions, data, reconstruction
                )
                total_loss += total_loss_batch.item()
                total_class_loss += class_loss_batch.item()
                total_recon_loss += recon_loss_batch.item()

                _, predicted = predictions.max(1)
                _, target_labels = target.max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target_labels.cpu().numpy())
                all_probabilities.append(predictions.cpu().numpy())

                total += target.size(0)
                correct += predicted.eq(target_labels).sum().item()

                batch_acc = 100. * correct / total
                pbar.set_postfix({
                    '准确率': f'{batch_acc:.2f}%'
                })

        test_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        test_class_loss = total_class_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        test_recon_loss = total_recon_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        test_acc = 100. * correct / total if total > 0 else 0.0

        all_probabilities = np.vstack(all_probabilities) if all_probabilities else np.array([])

        from sklearn.metrics import precision_score, recall_score, f1_score

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        precision_macro = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
        recall_macro = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
        f1_macro = f1_score(all_targets, all_predictions, average='macro', zero_division=0)

        precision_weighted = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall_weighted = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1_weighted = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)

        print("\n" + "="*60)
        print("测试结果")
        print("="*60)
        print(f"测试准确率: {test_acc:.4f}%")
        print(f"测试损失: {test_loss:.4f}")
        print(f"分类损失: {test_class_loss:.4f}")
        print(f"重构损失: {test_recon_loss:.4f}")
        print("\n宏平均指标:")
        print(f"  精确率: {precision_macro:.4f}")
        print(f"  召回率: {recall_macro:.4f}")
        print(f"  F1分数: {f1_macro:.4f}")
        print("\n加权平均指标:")
        print(f"  精确率: {precision_weighted:.4f}")
        print(f"  召回率: {recall_weighted:.4f}")
        print(f"  F1分数: {f1_weighted:.4f}")
        print("="*60)

        results = {
            'test_loss': test_loss,
            'test_class_loss': test_class_loss,
            'test_recon_loss': test_recon_loss,
            'test_acc': test_acc,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }

        return results


def create_trainer(model, train_loader, class_weights=None):
    loss_fn = create_loss_function(config, class_weights)

    if config.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"不支持的优化器: {config.OPTIMIZER}")

    logger = Logger(log_file=config.get_log_file())

    trainer = CapsuleNetTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        logger=logger
    )

    return trainer


def test_trainer():
    print("\n" + "="*60)
    print("测试训练器")
    print("="*60)

    from data_loader import FCGRDataset
    import tempfile

    num_samples = 100
    dummy_fcgr = np.random.randn(num_samples, 64, 64, 1).astype(np.float32)
    dummy_labels = np.random.choice(config.CLASS_NAMES, num_samples)

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        np.savez(f, fcgr=dummy_fcgr, labels=dummy_labels)
        temp_path = f.name

    dataset = FCGRDataset(temp_path, augment=False, mode='train')

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = create_model(config).to(config.DEVICE)
    trainer = create_trainer(model, dataloader)

    print("\n测试训练步骤...")
    data, target = next(iter(dataloader))
    data, target = data.to(config.DEVICE), target.to(config.DEVICE)
    predictions, digit_caps, reconstruction = model(data, target)
    loss_fn = trainer.loss_fn
    total_loss, class_loss, recon_loss = loss_fn(target, predictions, data, reconstruction)
    print(f"总损失: {total_loss.item():.4f}")
    print(f"分类损失: {class_loss.item():.4f}")
    print(f"重构损失: {recon_loss.item():.4f}")

    print("\n测试验证步骤...")
    val_loss, val_acc, val_class_loss, val_recon_loss = trainer.validate(dataloader, 0)
    print(f"验证损失: {val_loss:.4f}")
    print(f"验证准确率: {val_acc:.2f}%")

    os.unlink(temp_path)
    print("\n训练器测试完成!")


if __name__ == "__main__":
    test_trainer()