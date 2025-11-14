import os
import itertools
import json
import pandas as pd
import numpy as np
from experiments.search_train_fdse import main, argparse


def run_grid_search():
    lambda_values = [1.0, 0.01]
    tau_values = [0.01, 0.1, 0.5]
    beta = 0.001

    results_csv_path = "results/grid_search_results2.csv"
    os.makedirs("results", exist_ok=True)

    results = []

    for lam, tau in itertools.product(lambda_values, tau_values):
        print(f"\n========== Running λ={lam}, τ={tau} ==========")

        result_dir = os.path.join("results", f"lambda{lam}_tau{tau}")
        os.makedirs(result_dir, exist_ok=True)

        parser = argparse.ArgumentParser()
        parser.add_argument("--data_root", type=str, default="data/office_caltech10")
        parser.add_argument("--rounds", type=int, default=80)
        parser.add_argument("--local_epochs", type=int, default=1)
        parser.add_argument("--lr", type=float, default=0.01)
        parser.add_argument("--batch_size", type=int, default=50)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--lambda_con", type=float, default=lam)
        parser.add_argument("--beta", type=float, default=beta)
        parser.add_argument("--tau", type=float, default=tau)
        parser.add_argument("--G", type=int, default=2)
        parser.add_argument("--dw", type=int, default=3)
        parser.add_argument("--num_classes", type=int, default=10)
        parser.add_argument("--repeats", type=int, default=1)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
        parser.add_argument("--pretrained", action="store_true", default=True)
        parser.add_argument("--verbose", action="store_true", default=False)
        parser.add_argument("--result_dir", type=str, default=result_dir)
        args = parser.parse_args(args=[])

        # === 运行一次主训练 ===
        history = main(args)

        # --- 获取最终与最佳结果 ---
        rounds = np.array(history["rounds"])
        avg_acc = np.array(history["avg_acc"])
        avg_loss = np.array(history["avg_loss"])
        all_acc = np.array(history["all_acc"])
        all_loss = np.array(history["all_loss"])

        final_idx = -1
        best_idx = np.argmax(avg_acc)
        best_round = int(rounds[best_idx])

        summary = {
            "lambda": lam,
            "tau": tau,
            "seed": args.seed,
            "best_round": best_round,
            "best_avg_acc": float(avg_acc[best_idx]),
            "best_all_acc": float(all_acc[best_idx]),
            "best_avg_loss": float(avg_loss[best_idx]),
            "best_all_loss": float(all_loss[best_idx]),
            "final_avg_acc": float(avg_acc[final_idx]),
            "final_all_acc": float(all_acc[final_idx]),
            "final_avg_loss": float(avg_loss[final_idx]),
            "final_all_loss": float(all_loss[final_idx])
        }

        # 保存 summary.json
        with open(os.path.join(result_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        results.append(summary)

        # === 实时写入 CSV（即使中途终止也保留） ===
        df_partial = pd.DataFrame(results)
        df_partial.to_csv(results_csv_path, index=False)

        print(f"✅ Completed λ={lam}, τ={tau} | best_acc={summary['best_avg_acc']:.4f}, round={best_round}")

    print("\n✅ Grid search finished. Full results saved to results/grid_search_results.csv")
    print(pd.DataFrame(results))


if __name__ == "__main__":
    run_grid_search()
