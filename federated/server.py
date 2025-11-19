"""federated/server.py

Server orchestration for FDSE reproduction (strict implementation following paper).

This updated version includes:
- storage of per-client personalized DSE parameter state (personalized_dse_state)
- broadcasting per-client model copies where DSE parameters are replaced by the stored personalized copy
- aggregation that updates global DFE state and updates personalized_dse_state per client

Usage:
    server = Server(global_model, clients, device='cuda')
    server.run(...)

"""

from typing import List, Dict, Any
import copy
import torch
import torch.nn as nn
import numpy as np

from .aggregator import Aggregator
from .utils import split_dfe_dse_parameters


class Server:
    def __init__(self, global_model: nn.Module, clients: List[Any], device='cpu', tau: float = 0.1):
        self.global_model = global_model.to(device)
        self.clients = clients
        self.device = device
        self.aggregator = Aggregator(tau=tau)

        # determine dfe/dse parameter name lists
        self.dfe_names, self.dse_names = split_dfe_dse_parameters(self.global_model)
        # build layer groups for consensus aggregation: group by module prefix up to last '.' before param
        self.layer_groups = self._build_layer_groups(self.dfe_names)
        self.dse_layer_groups = self._build_layer_groups(self.dse_names)

        # storage for each client's personalized DSE parameters (mapping client_id -> {param_name: tensor})
        # initialize by copying global model's DSE params as the initial personalized params
        self.personalized_dse_state: Dict[int, Dict[str, torch.Tensor]] = {
            idx: {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items() if k in self.dse_names}
            for idx in range(len(self.clients))
        }

        # attach initial BN statistics
        self._update_global_bn_stats()

        # 新增：最佳模型状态存储
        self.best_global_state = None
        self.best_personalized_dse_state = None
        self.best_val_acc = 0.0
        self.best_round = -1

    def _build_layer_groups(self, param_names: List[str]) -> List[List[str]]:
        groups = {}
        for name in param_names:
            # module prefix = everything before the last '.'
            if '.' in name:
                prefix = name.rsplit('.', 1)[0]
            else:
                prefix = name
            groups.setdefault(prefix, []).append(name)
        # return list of groups (each group is list of full param names)
        return list(groups.values())

    def _extract_bn_dfe_stats(self) -> Dict[str, Dict[str, torch.Tensor]]:
        stats = {}
        for name, module in self.global_model.named_modules():
            if hasattr(module, 'bn_dfe') and isinstance(module.bn_dfe, (nn.BatchNorm1d, nn.BatchNorm2d)):
                stats[name] = {
                    'mean': module.bn_dfe.running_mean.detach().cpu().clone(),
                    'var': module.bn_dfe.running_var.detach().cpu().clone()
                }
        return stats

    def _update_global_bn_stats(self):
        stats = self._extract_bn_dfe_stats()
        self.global_model._bn_dfe_stats = stats

    def _aggregate_bn_dfe(self, clients_bn_stats: List[Dict[str, Dict[str, torch.Tensor]]]):
        """聚合所有客户端的BN-DFE统计信息"""
        if not clients_bn_stats or len(clients_bn_stats) == 0:
            return

        # 获取所有BN层名称
        all_bn_names = set()
        for client_stats in clients_bn_stats:
            all_bn_names.update(client_stats.keys())

        # 对每个BN层进行平均
        for bn_name in all_bn_names:
            means = []
            vars_ = []

            # 收集所有客户端该BN层的统计信息
            for client_stats in clients_bn_stats:
                if bn_name in client_stats:
                    mean_tensor = client_stats[bn_name]['running_mean']
                    var_tensor = client_stats[bn_name]['running_var']

                    # 确保张量至少有1维
                    if mean_tensor.dim() == 0:
                        mean_tensor = mean_tensor.unsqueeze(0)
                    if var_tensor.dim() == 0:
                        var_tensor = var_tensor.unsqueeze(0)

                    means.append(mean_tensor)
                    vars_.append(var_tensor)

            if means:  # 确保有统计信息可聚合
                # 找到对应的全局BN模块
                for name, module in self.global_model.named_modules():
                    if name == bn_name and hasattr(module, 'bn_dfe'):
                        # 计算平均值并更新全局BN
                        avg_mean = torch.stack(means).mean(dim=0)
                        avg_var = torch.stack(vars_).mean(dim=0)

                        module.bn_dfe.running_mean.copy_(avg_mean)
                        module.bn_dfe.running_var.copy_(avg_var)
                        break

    def broadcast(self) -> List[nn.Module]:
        """Return a list of model copies, one per client.

        For each client, produce a copy of the global model and overwrite its DSE params
        with the stored personalized DSE state for that client (if any).
        """
        models = []
        base_state = self.global_model.state_dict()
        for cid in range(len(self.clients)):
            m = copy.deepcopy(self.global_model)
            state = {k: v.detach().cpu().clone() for k, v in base_state.items()}
            # overwrite DSE keys with personalized if exist
            personalized = self.personalized_dse_state.get(cid, {})
            for k, v in personalized.items():
                if k in state:
                    state[k] = v.clone()
            m.load_state_dict(state, strict=False)
            models.append(m)
        return models

    def aggregate(self, local_states: List[Dict[str, torch.Tensor]], personalized_tau: float = None):
        """Aggregate global DFE/head parameters and compute personalized DSE updates.

        local_states: list of client state_dicts (cpu tensors) returned by clients after local update
        Returns: personalized result (list per DSE layer of per-client mappings) and updates internal storage
        """
        # aggregate global (DFE + head) by consensus-maximization per layer
        global_state = {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}
        new_global_state = self.aggregator.aggregate_global(global_state, local_states, self.layer_groups)
        # update global model
        self.global_model.load_state_dict(new_global_state)

        # personalized aggregation for DSE modules: produce per-layer per-client personalized dicts
        personalized_per_layer = self.aggregator.aggregate_personalized(local_states, self.dse_layer_groups, tau=personalized_tau)

        # personalized_per_layer is list over layers of list over clients of mapping name->tensor
        # we need to merge per-layer results into per-client personalized state dict
        num_clients = len(local_states)
        per_client_updates: List[Dict[str, torch.Tensor]] = [ {} for _ in range(num_clients) ]
        for layer_group_result in personalized_per_layer:
            # layer_group_result: list over clients of mapping name->tensor for that layer
            for cid, mapping in enumerate(layer_group_result):
                for k, v in mapping.items():
                    per_client_updates[cid][k] = v.detach().cpu().clone()

        # update server-side personalized storage
        for cid in range(num_clients):
            # overwrite the saved DSE params for client cid with the aggregated ones
            # if some param not present in per_client_updates, keep previous value
            saved = self.personalized_dse_state.get(cid, {})
            for k, v in per_client_updates[cid].items():
                saved[k] = v.clone()
            self.personalized_dse_state[cid] = saved

        self._update_global_bn_stats()
        return per_client_updates

    def run(self, rounds=100, local_epochs=1, lr=0.01, lambda_con=0.1, beta=0.001, personalized_tau=0.01):
        history = {
            'rounds': [],
            'avg_acc': [], 'avg_loss': [],
            'all_acc': [], 'all_loss': [],
            'val_acc': [], 'val_loss': []  # 新增验证集历史
        }

        for r in range(rounds):
            lr_t = lr * (0.998 ** r)
            print(f"[Server] Starting round {r:03d} | lr={lr_t:.6f}")

            # 本地训练阶段
            local_states = []
            clients_bn_stats = []
            model_copies = self.broadcast()

            for cid, c in enumerate(self.clients):
                st = c.local_update(model_copies[cid], local_epochs, lr_t, lambda_con, beta)
                local_states.append({k: v.detach().cpu().clone() for k, v in st.items()})
                bn_stats = c.get_bn_dfe_stats()  # 收集BN统计信息
                clients_bn_stats.append(bn_stats)

            # 聚合阶段
            self.aggregate(local_states, personalized_tau)

            # 聚合BN统计信息
            self._aggregate_bn_dfe(clients_bn_stats)

            # 验证和测试阶段 - 使用独立的模型副本
            val_accs, val_losses = [], []
            test_accs, test_losses, test_corrects, test_totals = [], [], [], []

            for cid, c in enumerate(self.clients):
                # 为验证集创建独立的模型副本
                val_model = copy.deepcopy(self.global_model)
                saved_dse = self.personalized_dse_state.get(cid, {})
                if saved_dse:
                    val_state = val_model.state_dict()
                    for k, v in saved_dse.items():
                        if k in val_state:
                            val_state[k] = v.clone()
                    val_model.load_state_dict(val_state, strict=False)

                # 验证集评估
                c.model.load_state_dict(val_model.state_dict(), strict=False)
                val_acc, val_loss = c.validate()
                val_accs.append(val_acc)
                val_losses.append(val_loss)

                # 为测试集创建独立的模型副本
                test_model = copy.deepcopy(self.global_model)
                if saved_dse:
                    test_state = test_model.state_dict()
                    for k, v in saved_dse.items():
                        if k in test_state:
                            test_state[k] = v.clone()
                    test_model.load_state_dict(test_state, strict=False)

                # 测试集评估
                c.model.load_state_dict(test_model.state_dict(), strict=False)
                acc, loss, correct, total = c.evaluate(return_raw=True)
                test_accs.append(acc)
                test_losses.append(loss)
                test_corrects.append(correct)
                test_totals.append(total)

                # 清理内存
                del val_model, test_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 计算平均指标
            avg_val_acc = np.mean(val_accs)
            avg_val_loss = np.mean(val_losses)
            avg_test_acc = np.mean(test_accs)
            avg_test_loss = np.mean(test_losses)
            all_test_acc = np.sum(test_corrects) / np.sum(test_totals) if np.sum(test_totals) > 0 else 0.0
            all_test_loss = np.sum(np.array(test_losses) * np.array(test_totals)) / np.sum(test_totals) if np.sum(test_totals) > 0 else 0.0

            print(f"[Server] Round {r:03d}: VAL_acc={avg_val_acc:.4f}, TEST_acc={avg_test_acc:.4f}, "
                  f"VAL-TEST_diff={avg_val_acc - avg_test_acc:.4f}")

            # 记录历史
            history['rounds'].append(r)
            history['avg_acc'].append(avg_test_acc)
            history['avg_loss'].append(avg_test_loss)
            history['all_acc'].append(all_test_acc)
            history['all_loss'].append(all_test_loss)
            history['val_acc'].append(avg_val_acc)
            history['val_loss'].append(avg_val_loss)

            # 模型选择：保存验证集上性能最佳的模型
            if avg_val_acc > self.best_val_acc:
                self.best_val_acc = avg_val_acc
                self.best_round = r
                self.best_global_state = copy.deepcopy(self.global_model.state_dict())
                self.best_personalized_dse_state = copy.deepcopy(self.personalized_dse_state)
                print(f"[Best Model] Round {r}: val_acc={avg_val_acc:.4f}, test_acc={avg_test_acc:.4f}")

        # 训练结束后加载最佳模型
        if self.best_global_state is not None:
            print(f"[Final] Loading best model from round {self.best_round} "
                  f"(val_acc={self.best_val_acc:.4f})")
            self.global_model.load_state_dict(self.best_global_state)
            self.personalized_dse_state = self.best_personalized_dse_state

            # 最终评估最佳模型在所有客户端上的性能
            print("\n=== Final Evaluation with Best Model ===")
            final_val_accs, final_test_accs = [], []
            for cid, c in enumerate(self.clients):
                # 为每个客户端加载最佳个性化模型
                best_model = copy.deepcopy(self.global_model)
                best_dse = self.best_personalized_dse_state.get(cid, {})
                if best_dse:
                    best_state = best_model.state_dict()
                    for k, v in best_dse.items():
                        if k in best_state:
                            best_state[k] = v.clone()
                    best_model.load_state_dict(best_state, strict=False)

                c.model.load_state_dict(best_model.state_dict(), strict=False)
                val_acc, _ = c.validate()
                test_acc, _ = c.evaluate()
                final_val_accs.append(val_acc)
                final_test_accs.append(test_acc)
                print(f"Client {cid}: val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")

                del best_model

            avg_final_val = np.mean(final_val_accs)
            avg_final_test = np.mean(final_test_accs)
            print(f"\n[Final Summary] Val: {avg_final_val:.4f}, Test: {avg_final_test:.4f}, "
                  f"Difference: {avg_final_val - avg_final_test:.4f}")

        return history

if __name__ == '__main__':
    print('Server module for FDSE federation with personalized DSE storage.')
