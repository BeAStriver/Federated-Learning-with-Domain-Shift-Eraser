# diagnostic_check.py
import torch
from data_loader.office_caltech10_loader import get_office_caltech10_dataloaders

def strict_data_check():
    """严格检查数据分割独立性"""
    print("=== 严格数据分割检查 ===")

    dataloaders = get_office_caltech10_dataloaders(root="data/lry/office_caltech10", batch_size=50, seed=42)

    for domain, (train_loader, val_loader, test_loader) in dataloaders.items():
        print(f"\n--- 检查领域: {domain} ---")

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

        print(f"训练∩验证: {train_val_overlap}")
        print(f"训练∩测试: {train_test_overlap}")
        print(f"验证∩测试: {val_test_overlap}")

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

        if train_transform_id != val_transform_id and train_transform_id != test_transform_id:
            print("✅ transform独立")
        else:
            print("❌ transform共享！存在数据泄露风险")

if __name__ == "__main__":
    strict_data_check()