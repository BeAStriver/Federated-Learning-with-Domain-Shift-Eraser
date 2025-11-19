import argparse
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import csv

from model.fdse_model import build_fdse_alexnet
from data_loader.office_caltech10_loader import get_office_caltech10_dataloaders, set_seed
from federated.client import Client
from federated.server import Server


def build_clients(model, dataloaders, device):
    """构建客户端，现在每个客户端有三个数据加载器"""
    clients = []
    for cid, domain in enumerate(dataloaders.keys()):
        train_loader, val_loader, test_loader = dataloaders[domain]  # 现在解包三个loader
        clients.append(Client(
            cid=cid,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,      # 新增
            test_loader=test_loader,
            device=device
        ))
    return clients


def plot_and_save_curves(history, seed, save_dir):
    """保存训练曲线和历史记录，现在包含验证集曲线"""
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"fdse_history_seed{seed}.csv")

    # 保存CSV - 新增验证集指标
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "avg_acc", "all_acc", "avg_loss", "all_loss", "val_acc", "val_loss"])
        for i in range(len(history["rounds"])):
            writer.writerow([
                history["rounds"][i],
                history["avg_acc"][i],
                history["all_acc"][i],
                history["avg_loss"][i],
                history["all_loss"][i],
                history["val_acc"][i],      # 新增
                history["val_loss"][i],     # 新增
            ])

    # 绘制准确率曲线 - 新增验证集曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history["rounds"], history["avg_acc"], label="Test AVG Accuracy", linewidth=2)
    plt.plot(history["rounds"], history["all_acc"], label="Test ALL Accuracy", linestyle="--", linewidth=2)
    plt.plot(history["rounds"], history["val_acc"], label="Validation Accuracy", color="green", linewidth=2)
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")
    plt.title(f"FDSE Accuracy Curves (seed={seed})")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fdse_acc_curve_seed{seed}.png"), dpi=300)

    # 绘制损失曲线 - 新增验证集曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history["rounds"], history["avg_loss"], label="Test AVG Loss", color="orange", linewidth=2)
    plt.plot(history["rounds"], history["all_loss"], label="Test ALL Loss", color="red", linestyle="--", linewidth=2)
    plt.plot(history["rounds"], history["val_loss"], label="Validation Loss", color="purple", linewidth=2)
    plt.xlabel("Communication Rounds")
    plt.ylabel("Loss")
    plt.title(f"FDSE Loss Curves (seed={seed})")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fdse_loss_curve_seed{seed}.png"), dpi=300)

    print(f"[Plot] Saved CSV and curves to {save_dir}/")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.verbose:
        print("Using device:", device)
        print(f"Using 8:1:1 data split (train:val:test)")

    all_histories = []
    seeds = [args.seed + i for i in range(args.repeats)]

    for seed in seeds:
        if args.verbose:
            print("=" * 30)
            print(f"Experiment seed: {seed}")
        set_seed(seed)

        # 加载数据 - 现在返回三个数据加载器
        dataloaders = get_office_caltech10_dataloaders(
            root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=224,
            train_ratio=0.8,    # 明确指定
            val_ratio=0.1,      # 明确指定
            test_ratio=0.1,     # 明确指定
            seed=seed
        )

        print("\n" + "="*50)
        print("数据分割完整性检查")
        print("="*50)
        from data_loader.office_caltech10_loader import strict_data_integrity_check
        if not strict_data_integrity_check(dataloaders):
            print("❌ 数据完整性检查失败！程序终止")
            return
        print("="*50 + "\n")

        # 构建模型
        model = build_fdse_alexnet(num_classes=args.num_classes, G=args.G, dw=args.dw, pretrained=args.pretrained)
        model.to(device)

        # 构建客户端与服务器
        clients = build_clients(model, dataloaders, device)
        server = Server(global_model=model, clients=clients, device=device, tau=args.tau)

        # 运行联邦训练
        history = server.run(
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            lr=args.lr,
            lambda_con=args.lambda_con,
            beta=args.beta,
            personalized_tau=args.tau
        )

        # 计算最佳轮次（基于验证集）
        best_val_idx = int(np.argmax(history["val_acc"]))  # 基于验证集选择
        best_round = history["rounds"][best_val_idx]
        best_val_acc = history["val_acc"][best_val_idx]
        best_test_acc = history["avg_acc"][best_val_idx]   # 对应的测试集准确率

        if args.verbose:
            print(f"[Best] Round {best_round}: best_val_acc={best_val_acc:.4f}, corresponding_test_acc={best_test_acc:.4f}")

        # 保存结果
        result_dir = getattr(args, 'result_dir', 'results')
        plot_and_save_curves(history, seed, result_dir)
        print("[EXPORT] Results saved at:", os.path.abspath(result_dir))

        if args.verbose:
            print(f"[Checkpoint] Saved for seed {seed}.")

        # 确保目录存在
        os.makedirs(args.checkpoint_dir, exist_ok=True)

        # 保存checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, f'fdse_seed_{seed}.pth')
        torch.save({
            'global_state': server.global_model.state_dict(),
            'personalized_dse_state': server.personalized_dse_state,
            'best_global_state': getattr(server, 'best_global_state', None),
            'best_personalized_dse_state': getattr(server, 'best_personalized_dse_state', None),
            'best_val_acc': getattr(server, 'best_val_acc', 0.0),
            'best_round': getattr(server, 'best_round', -1),
            'config': {
                'lambda_con': args.lambda_con,
                'tau': args.tau,
                'beta': args.beta,
                'seed': seed
            }
        }, checkpoint_path)

        print(f"[Checkpoint] ✅ 保存成功: {checkpoint_path}")

        summary = {
            'seed': seed,
            'lambda': args.lambda_con,
            'tau': args.tau,
            'final_val_acc': history['val_acc'][-1],
            'final_test_acc': history['avg_acc'][-1],
            'best_round': best_round,
            'best_val_acc': best_val_acc,
            'best_test_acc': best_test_acc
        }
        with open(os.path.join(result_dir, f'summary_seed{seed}.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        all_histories.append(history)

    # 汇总结果
    avg_best_val_acc = np.mean([max(h['val_acc']) for h in all_histories])
    avg_best_test_acc = np.mean([h['avg_acc'][np.argmax(h['val_acc'])] for h in all_histories])
    print(f"[Summary] Average best VAL_acc over {args.repeats} runs: {avg_best_val_acc:.4f}")
    print(f"[Summary] Average best TEST_acc over {args.repeats} runs: {avg_best_test_acc:.4f}")

    # 确保返回正确的history字典
    if 'all_histories' in locals() and len(all_histories) > 0:
        return all_histories[0]  # 返回第一个种子的history
    else:
        # 返回空的history结构
        return {
            'rounds': [], 'avg_acc': [], 'all_acc': [],
            'avg_loss': [], 'all_loss': [], 'val_acc': [], 'val_loss': []
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../data/office_caltech10")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lambda_con", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--G", type=int, default=2)
    parser.add_argument("--dw", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="data/lry/FDSE/checkpoints_check")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--result_dir", type=str, default="data/lry/FDSE/results_check")

    args = parser.parse_args()
    main(args)
