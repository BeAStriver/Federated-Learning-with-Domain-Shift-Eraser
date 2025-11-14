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

        # update BN stats attached
        self._update_global_bn_stats()
        return per_client_updates

    def run(self, rounds=100, local_epochs=1, lr=0.01, lambda_con=0.1, beta=0.001, personalized_tau=0.01):
        history = {'rounds': [], 'avg_acc': [], 'avg_loss': [], 'all_acc': [], 'all_loss': []}

        for r in range(rounds):
            lr_t = lr * (0.998 ** r)
            print(f"[Server] Starting round {r:03d} | lr={lr_t:.6f}")

            local_states = []
            accs, losses, corrects, totals = [], [], [], []

            model_copies = self.broadcast()

            for cid, c in enumerate(self.clients):
                st = c.local_update(model_copies[cid], local_epochs, lr_t, lambda_con, beta)
                local_states.append({k: v.detach().cpu().clone() for k, v in st.items()})

            # evaluate all clients after aggregation
            for cid, c in enumerate(self.clients):
                eval_model = copy.deepcopy(self.global_model)
                saved = self.personalized_dse_state.get(cid, {})
                if saved:
                    state = eval_model.state_dict()
                    for k, v in saved.items():
                        if k in state:
                            state[k] = v.clone()
                    eval_model.load_state_dict(state, strict=False)
                c.model.load_state_dict(eval_model.state_dict(), strict=False)
                acc, loss, correct, total = c.evaluate(return_raw=True)
                accs.append(acc)
                losses.append(loss)
                corrects.append(correct)
                totals.append(total)
                print(f" [Client {cid}] acc={acc:.4f}, loss={loss:.4f}")

            self.aggregate(local_states, personalized_tau)

            avg_acc = np.mean(accs)
            avg_loss = np.mean(losses)
            all_acc = np.sum(corrects) / np.sum(totals)
            all_loss = np.sum(np.array(losses) * np.array(totals)) / np.sum(totals)

            print(f"[Server] Round {r:03d}: AVG_acc={avg_acc:.4f}, ALL_acc={all_acc:.4f}, AVG_loss={avg_loss:.4f}")

            history['rounds'].append(r)
            history['avg_acc'].append(avg_acc)
            history['avg_loss'].append(avg_loss)
            history['all_acc'].append(all_acc)
            history['all_loss'].append(all_loss)

        return history


if __name__ == '__main__':
    print('Server module for FDSE federation with personalized DSE storage.')
