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
    clients = []
    for cid, domain in enumerate(dataloaders.keys()):
        tr, te = dataloaders[domain]
        clients.append(Client(cid=cid, model=model, train_loader=tr, test_loader=te, device=device))
    return clients


def plot_and_save_curves(history, seed, save_dir):
    """保存训练曲线和历史记录"""
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"fdse_history_seed{seed}.csv")

    # === 保存 CSV ===
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "avg_acc", "all_acc", "avg_loss", "all_loss"])
        for i in range(len(history["rounds"])):
            writer.writerow([
                history["rounds"][i],
                history["avg_acc"][i],
                history["all_acc"][i],
                history["avg_loss"][i],
                history["all_loss"][i],
            ])

    # === 绘制准确率曲线 ===
    plt.figure(figsize=(8, 5))
    plt.plot(history["rounds"], history["avg_acc"], label="AVG Accuracy", linewidth=2)
    plt.plot(history["rounds"], history["all_acc"], label="ALL Accuracy", linestyle="--", linewidth=2)
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")
    plt.title(f"FDSE Accuracy Curves (seed={seed})")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fdse_acc_curve_seed{seed}.png"), dpi=300)

    # === 绘制损失曲线 ===
    plt.figure(figsize=(8, 5))
    plt.plot(history["rounds"], history["avg_loss"], label="AVG Loss", color="orange", linewidth=2)
    plt.plot(history["rounds"], history["all_loss"], label="ALL Loss", color="red", linestyle="--", linewidth=2)
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
        print(f"Using fixed batch size: {args.batch_size}")

    all_histories = []
    seeds = [args.seed + i for i in range(args.repeats)]

    for seed in seeds:
        if args.verbose:
            print("=" * 30)
            print(f"Experiment seed: {seed}")
        set_seed(seed)

        # === 加载数据 ===
        dataloaders = get_office_caltech10_dataloaders(
            root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers,
            image_size=224, split_ratio=0.8, seed=seed
        )

        # === 构建模型 ===
        model = build_fdse_alexnet(num_classes=args.num_classes, G=args.G, dw=args.dw, pretrained=args.pretrained)
        model.to(device)

        # === 构建客户端与服务器 ===
        clients = build_clients(model, dataloaders, device)
        server = Server(global_model=model, clients=clients, device=device, tau=args.tau)

        # === 运行完整联邦训练（内部已自带循环）===
        history = server.run(
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            lr=args.lr,
            lambda_con=args.lambda_con,
            beta=args.beta,
            personalized_tau=args.tau
        )

        # === 计算最佳轮次 ===
        best_idx = int(np.argmax(history["avg_acc"]))
        best_round = history["rounds"][best_idx]
        best_acc = history["avg_acc"][best_idx]
        best_all_acc = history["all_acc"][best_idx]

        if args.verbose:
            print(f"[Best] Round {best_round}: best_avg_acc={best_acc:.4f}, best_all_acc={best_all_acc:4f}")

        # === 保存曲线与结果 ===
        result_dir = getattr(args, 'result_dir', 'results')
        plot_and_save_curves(history, seed, result_dir)
        print("[EXPORT] Results saved at:", os.path.abspath(result_dir))

        summary = {
            'seed': seed,
            'lambda': args.lambda_con,
            'tau': args.tau,
            'final_avg_acc': history['avg_acc'][-1],
            'final_all_acc': history['all_acc'][-1],
            'final_avg_loss': history['avg_loss'][-1],
            'final_all_loss': history['all_loss'][-1],
            'best_round': best_round,
            'best_avg_acc': best_acc,
            'best_all_acc': best_all_acc
        }
        with open(os.path.join(result_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        all_histories.append(history)

        # === 保存 checkpoint ===
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save({
            'global_state': server.global_model.state_dict(),
            'personalized': server.personalized_dse_state
        }, os.path.join(args.checkpoint_dir, f'fdse_seed_{seed}.pth'))
        if args.verbose:
            print(f"[Checkpoint] Saved for seed {seed}.")

    # === 汇总 ===
    avg_final_acc = np.mean([h["avg_acc"][-1] for h in all_histories])
    avg_best_acc = np.mean([max(h['avg_acc']) for h in all_histories])
    avg_best_all_acc = np.mean([max(h['all_acc']) for h in all_histories])
    print(f"[Summary] Average *best* AVG_acc over {args.repeats} runs: {avg_best_acc:.4f}")
    print(f"[Summary] Average *best* ALL_acc over {args.repeats} runs: {avg_best_all_acc:.4f}")
    print(f"Average final accuracy over {args.repeats} runs: {avg_final_acc:.4f}")
    return history


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
    parser.add_argument("--checkpoint_dir", type=str, default="data/lry/checkpoints2")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--result_dir", type=str, default="data/lry/results2")

    args = parser.parse_args()
    main(args)
