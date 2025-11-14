"""federated/aggregator.py

Accurate implementations of FDSE aggregation strategies following the paper:
- consensus_maximization (Eq.8): per-layer MGDA-like optimization to minimize || sum_k u_k d_k ||^2
  subject to u_k >= 0 and sum u_k = 1. We solve this QP by forming G_ij = d_i^T d_j and then
  solving the QP min_u 0.5 u^T G u s.t. u >= 0, sum u = 1 using projected gradient descent with
  projection onto the simplex (fast euclidean projection).

- similarity_aware_personalization (Eq.9): for each DSE layer, form Q = normalize(V), K=Q, compute
  A = softmax(Q K^T / tau) and output A V (attention weighted aggregation) — per-layer.

These implementations are strict (no heuristic shortcuts) and operate on flattened per-layer
parameter/gradient vectors; they return aggregated parameter deltas in a state-dict-like map.
"""

from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import flatten_params, project_onto_simplex, unflatten_to_state_dict

class Aggregator:
    def __init__(self, tau: float = 0.1, pgd_steps: int = 4000, pgd_lr: float = 3e-4):
        """Parameters:
        - tau: temperature for similarity-aware personalization
        - pgd_steps: steps for projected gradient descent solving the simplex QP
        - pgd_lr: learning rate for projected gradient descent
        """
        self.tau = tau
        self.pgd_steps = pgd_steps
        self.pgd_lr = pgd_lr

    def consensus_maximization_per_layer(self, global_state: Dict[str, torch.Tensor], locals_states: List[Dict[str, torch.Tensor]], layer_param_names: List[str]) -> Dict[str, torch.Tensor]:
        """Perform consensus maximization for a single layer (given parameter names for that layer).

        Returns aggregated parameter delta for these param names (as mapping name->tensor)
        """
        # build d_k = local - global for each client, flattened
        d_list = []
        for st in locals_states:
            vec = flatten_params([st[n] for n in layer_param_names])
            gvec = flatten_params([global_state[n] for n in layer_param_names])
            d = (vec - gvec)
            d_list.append(d)

        N = len(d_list)
        if N == 0:
            return {}

        D = d_list[0].numel()
        # form Gram matrix G where G_ij = d_i^T d_j
        G = torch.zeros((N, N), dtype=torch.float32)
        for i in range(N):
            for j in range(N):
                G[i, j] = float((d_list[i] * d_list[j]).sum())

        # QP: min 0.5 u^T G u s.t. u>=0, sum u =1
        # solve via projected gradient descent on u
        u = torch.ones(N, dtype=torch.float32) / N
        for t in range(self.pgd_steps):
            # grad = G u
            grad = G.matmul(u)
            u = u - self.pgd_lr * grad
            # project onto simplex
            u = project_onto_simplex(u)
        # aggregated delta
        agg = torch.zeros_like(d_list[0])
        for i in range(N):
            agg += u[i] * d_list[i]

        # unflatten to mapping
        template = {n: global_state[n] for n in layer_param_names}
        out = unflatten_to_state_dict(agg, template)
        return out

    def similarity_aware_per_layer(self, locals_states: List[Dict[str, torch.Tensor]], layer_param_names: List[str], tau: float=None) -> List[Dict[str, torch.Tensor]]:
        """Perform similarity-aware personalization for a single layer.

        Input: list of client state dicts; output: list (per-client) of aggregated tensors (mapping name->tensor)
        using attention A = softmax(QK^T / tau) and out = A V where V are flattened per-client vectors.
        """
        tau = self.tau if tau is None else tau
        N = len(locals_states)
        if N == 0:
            return []
        V = []
        for st in locals_states:
            V.append(flatten_params([st[n] for n in layer_param_names]))
        V = torch.stack(V, dim=0)  # (N, D)
        norms = V.norm(dim=1, keepdim=True) + 1e-12
        Q = V / norms
        K = Q
        sim = (Q @ K.t()) / tau
        A = F.softmax(sim, dim=1)  # (N, N)
        out = A @ V  # (N, D)
        results = []
        template = {n: locals_states[0][n] for n in layer_param_names}
        for i in range(N):
            mapping = unflatten_to_state_dict(out[i], template)
            results.append(mapping)
        return results

    def aggregate_global(self, global_state: Dict[str, torch.Tensor], locals_states: List[Dict[str, torch.Tensor]], layer_groups: List[List[str]]) -> Dict[str, torch.Tensor]:
        """Aggregate global (DFE & head) parameters by applying consensus_maximization per layer group.

        layer_groups: list of groups, each group is list of state_dict keys that belong to that layer.
        Returns a mapping name->new tensor (global updated state)
        """
        new_state = dict(global_state)
        for group in layer_groups:
            upd = self.consensus_maximization_per_layer(global_state, locals_states, group)
            # apply: new param = global + upd (paper sums normalized updates; here upd already is delta)
            for k, v in upd.items():
                new_state[k] = (new_state[k].to(v.device) + v).clone()
        return new_state

    def aggregate_personalized(self, locals_states: List[Dict[str, torch.Tensor]], layer_groups: List[List[str]], tau: float=None) -> List[List[Dict[str, torch.Tensor]]]:
        """For each layer group, compute per-client personalized aggregation results.

        Returns: list over layers of list over clients of mapping name->tensor (personalized copy for each client)
        """
        res = []
        for group in layer_groups:
            per_layer = self.similarity_aware_per_layer(locals_states, group, tau=tau)
            res.append(per_layer)
        return res
