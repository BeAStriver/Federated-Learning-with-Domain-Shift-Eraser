"""office_caltech10_loader.py

Data loading utilities for the Office-Caltech10 dataset
(as used in the CVPR 2025 paper: Federated Learning with Domain Shift Eraser).

Directory structure expected:
    data/OfficeCaltech10/
        ├── amazon/
        ├── caltech/
        ├── dslr/
        └── webcam/
Each subfolder contains class subdirectories (as in ImageFolder format).

This script provides per-domain train/test DataLoaders aligned with FDSE experiments.
"""

import os
import random
import numpy as np
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# Standard normalization for ImageNet pretrained AlexNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int = 224):
    """Return standard train/test transforms aligned with AlexNet settings."""
    train_t = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_t, test_t


def split_dataset_three_way(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """将数据集分割为训练集、验证集和测试集 (8:1:1) - 修复版本"""
    set_seed(seed)
    total_len = len(dataset)

    # 计算各集合大小
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len

    # 一次性生成三个互斥的索引集合
    indices = list(range(total_len))
    random.shuffle(indices)

    train_indices = indices[:train_len]
    val_indices = indices[train_len:train_len + val_len]
    test_indices = indices[train_len + val_len:]

    # 创建三个独立的Subset
    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)

    return train_set, val_set, test_set

def get_office_caltech10_dataloaders(
        root: str = "office_caltech10",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
) -> Dict[str, Tuple[DataLoader, DataLoader, DataLoader]]:

    domains = ["amazon", "caltech", "dslr", "webcam"]
    train_t, test_t = build_transforms(image_size)

    dataloaders = {}

    for domain in domains:
        domain_dir = os.path.join(root, domain)
        if not os.path.exists(domain_dir):
            raise FileNotFoundError(f"Domain folder not found: {domain_dir}")

        # 关键修复：创建一个数据集，一次性分割为三个互斥的子集
        full_dataset = datasets.ImageFolder(domain_dir)

        # 一次性分割所有索引
        total_len = len(full_dataset)
        train_len = int(total_len * train_ratio)
        val_len = int(total_len * val_ratio)
        test_len = total_len - train_len - val_len

        indices = list(range(total_len))
        random.Random(seed).shuffle(indices)

        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len]
        test_indices = indices[train_len + val_len:]

        print(f"Domain {domain}: total={total_len}, train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

        # 创建三个互斥的Subset，分别应用不同的transform
        train_set = torch.utils.data.Subset(
            datasets.ImageFolder(domain_dir, transform=train_t),
            train_indices
        )
        val_set = torch.utils.data.Subset(
            datasets.ImageFolder(domain_dir, transform=test_t),
            val_indices
        )
        test_set = torch.utils.data.Subset(
            datasets.ImageFolder(domain_dir, transform=test_t),
            test_indices
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers
        )
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers
        )

        dataloaders[domain] = (train_loader, val_loader, test_loader)

    return dataloaders

# 在office_caltech10_loader.py中添加检查代码
def strict_data_integrity_check(dataloaders):
    """严格的数据完整性检查"""
    print("\n" + "="*60)
    print("STRICT DATA INTEGRITY CHECK")
    print("="*60)

    all_good = True

    for domain, (train_loader, val_loader, test_loader) in dataloaders.items():
        print(f"\n--- 严格检查领域: {domain} ---")

        # 检查样本索引唯一性
        train_indices = set(train_loader.dataset.indices)
        val_indices = set(val_loader.dataset.indices)
        test_indices = set(test_loader.dataset.indices)

        print(f"训练集样本数: {len(train_indices)}")
        print(f"验证集样本数: {len(val_indices)}")
        print(f"测试集样本数: {len(test_indices)}")

        # 检查重叠
        train_val_overlap = len(train_indices & val_indices)
        train_test_overlap = len(train_indices & test_indices)
        val_test_overlap = len(val_indices & test_indices)

        print(f"训练∩验证重叠: {train_val_overlap}")
        print(f"训练∩测试重叠: {train_test_overlap}")
        print(f"验证∩测试重叠: {val_test_overlap}")

        # 检查transform独立性
        train_transform_id = id(train_loader.dataset.dataset.transform)
        val_transform_id = id(val_loader.dataset.dataset.transform)
        test_transform_id = id(test_loader.dataset.dataset.transform)

        print(f"训练transform ID: {train_transform_id}")
        print(f"验证transform ID: {val_transform_id}")
        print(f"测试transform ID: {test_transform_id}")

        # 验证结论
        if train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0:
            print("✅ 样本分割正确")
        else:
            print("❌ 样本有重叠！")
            all_good = False

        if train_transform_id != val_transform_id and train_transform_id != test_transform_id:
            print("✅ transform独立")
        else:
            print("❌ transform共享！")
            all_good = False

    print("\n" + "="*60)
    if all_good:
        print("✅ 所有数据完整性检查通过")
    else:
        print("❌ 数据完整性检查失败！")
    print("="*60)

    return all_good

if __name__ == "__main__":
    # 测试新的三路分割
    loaders = get_office_caltech10_dataloaders()
    for d, (tr, val, te) in loaders.items():
        print(f"{d}: train={len(tr.dataset)}, val={len(val.dataset)}, test={len(te.dataset)}")
