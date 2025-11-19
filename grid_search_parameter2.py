import os
import itertools
import json
import pandas as pd
import numpy as np
from experiments.train_fdse import main  # 修改：从train_fdse导入main
import argparse

def run_grid_search():
    # 使用论文补充材料中的参数范围
    lambda_values = [0.01, 0.1, 1.0]
    tau_values = [0.001, 0.01, 0.1, 0.5]
    beta = 0.001

    results_csv_path = "data/lry/results_grid/grid_search_results_fixed.csv"  # 修改文件名避免覆盖
    os.makedirs("results_grid", exist_ok=True)

    results = []

    for lam, tau in itertools.product(lambda_values, tau_values):
        print(f"\n========== Running λ={lam}, τ={tau} ==========")

        result_dir = os.path.join("data/lry/results_grid", f"lambda{lam}_tau{tau}")
        checkpoint_dir = os.path.join("data/lry/checkpoints_grid", f"lambda{lam}_tau{tau}")
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 创建参数对象（不通过命令行）
        class Args:
            def __init__(self):
                self.data_root = "data/lry/office_caltech10"
                self.rounds = 500  # 增加到300轮
                self.local_epochs = 1
                self.lr = 0.01
                self.batch_size = 50
                self.num_workers = 4
                self.lambda_con = lam
                self.beta = beta
                self.tau = tau
                self.G = 2
                self.dw = 3
                self.num_classes = 10
                self.repeats = 1
                self.seed = 42
                self.checkpoint_dir = checkpoint_dir
                self.pretrained = True
                self.verbose = False  # 改为True便于监控
                self.result_dir = result_dir

        args = Args()

        # === 运行一次主训练 ===
        try:
            history = main(args)

            # === 修复：添加类型检查和错误处理 ===
            if not isinstance(history, dict):
                print(f"❌ history不是字典类型: {type(history)}")
                continue

            if "val_acc" not in history:
                print("❌ history中缺少val_acc键")
                continue

            # 获取指标 - 确保使用正确的键
            required_keys = ["rounds", "avg_acc", "all_acc", "avg_loss", "all_loss", "val_acc", "val_loss"]
            missing_keys = [key for key in required_keys if key not in history]
            if missing_keys:
                print(f"❌ history缺少键: {missing_keys}")
                print(f"可用的键: {list(history.keys())}")
                continue

            # 获取验证集和测试集指标
            rounds = np.array(history["rounds"])
            avg_acc = np.array(history["avg_acc"])
            avg_loss = np.array(history["avg_loss"])
            all_acc = np.array(history["all_acc"])
            all_loss = np.array(history["all_loss"])
            val_acc = np.array(history["val_acc"])
            val_loss = np.array(history["val_loss"])

            # 检查数据有效性
            if len(val_acc) == 0:
                print("❌ val_acc为空")
                continue

            # 基于验证集选择最佳模型
            best_idx = np.argmax(val_acc)
            if best_idx >= len(rounds):
                best_idx = len(rounds) - 1

            best_round = int(rounds[best_idx])

            summary = {
                "lambda": lam,
                "tau": tau,
                "seed": args.seed,
                "best_round": best_round,
                # 验证集指标
                "best_val_acc": float(val_acc[best_idx]),
                "best_val_loss": float(val_loss[best_idx]),
                # 对应的测试集指标
                "best_avg_acc": float(avg_acc[best_idx]),
                "best_all_acc": float(all_acc[best_idx]),
                "best_avg_loss": float(avg_loss[best_idx]),
                "best_all_loss": float(all_loss[best_idx]),
                # 最终轮次指标
                "final_val_acc": float(val_acc[-1]),
                "final_val_loss": float(val_loss[-1]),
                "final_avg_acc": float(avg_acc[-1]),
                "final_all_acc": float(all_acc[-1]),
                "final_avg_loss": float(avg_loss[-1]),
                "final_all_loss": float(all_loss[-1])
            }

            # 保存 summary.json
            with open(os.path.join(result_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

            results.append(summary)

            # === 实时写入 CSV ===
            df_partial = pd.DataFrame(results)
            df_partial.to_csv(results_csv_path, index=False)

            print(f"✅ Completed λ={lam}, τ={tau} | best_val_acc={summary['best_val_acc']:.4f}, best_test_acc={summary['best_avg_acc']:.4f}")

        except Exception as e:
            print(f"❌ Failed for λ={lam}, τ={tau}: {str(e)}")
            # 记录失败情况
            failed_summary = {
                "lambda": lam,
                "tau": tau,
                "error": str(e),
                "status": "failed"
            }
            results.append(failed_summary)
            df_partial = pd.DataFrame(results)
            df_partial.to_csv(results_csv_path, index=False)

    print("\n✅ Grid search finished. Full results saved to", results_csv_path)
    if results:
        final_df = pd.DataFrame([r for r in results if 'best_val_acc' in r])
        if not final_df.empty:
            print(final_df[['lambda', 'tau', 'best_val_acc', 'best_avg_acc']])
        else:
            print("No successful runs to display.")

if __name__ == "__main__":
    run_grid_search()