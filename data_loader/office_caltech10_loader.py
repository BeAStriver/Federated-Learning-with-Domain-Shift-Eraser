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


def split_dataset(dataset, split_ratio=0.8, seed=42):
    """Split dataset into train/test subsets deterministically."""
    set_seed(seed)
    total_len = len(dataset)
    train_len = int(total_len * split_ratio)
    test_len = total_len - train_len
    train_set, test_set = random_split(dataset, [train_len, test_len])
    return train_set, test_set


def get_office_caltech10_dataloaders(
        root: str = "office_caltech10",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        split_ratio: float = 0.8,
        seed: int = 42,
) -> Dict[str, Tuple[DataLoader, DataLoader]]:
    """Build per-domain dataloaders for Office-Caltech10 dataset.

    Returns a dict: {domain: (train_loader, test_loader)}
    """

    domains = ["amazon", "caltech", "dslr", "webcam"]
    train_t, test_t = build_transforms(image_size)

    dataloaders = {}

    for domain in domains:
        domain_dir = os.path.join(root, domain)
        if not os.path.exists(domain_dir):
            raise FileNotFoundError(f"Domain folder not found: {domain_dir}")

        full_dataset = datasets.ImageFolder(domain_dir, transform=train_t)
        train_set, test_set = split_dataset(full_dataset, split_ratio, seed)

        # Update transforms for test split
        test_set.dataset.transform = test_t

        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        dataloaders[domain] = (train_loader, test_loader)
        print(f"Loaded domain {domain}: train={len(train_set)}, test={len(test_set)}")

    return dataloaders


if __name__ == "__main__":
    # quick test to verify dataloader works
    loaders = get_office_caltech10_dataloaders()
    for d, (tr, te) in loaders.items():
        print(f"{d}: train batches={len(tr)}, test batches={len(te)}")
