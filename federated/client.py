"""federated/client.py

Client-side local training for FDSE reproduction (strict implementation aligned with the paper).
- maintains per-DSE-module local running statistics with momentum gamma (matching BN momentum)
- computes consistency regularization LCon per Eq.(6) and LCon aggregated per Eq.(7) with layer weights wl

Client.local_update performs E local epochs and returns the updated model state_dict.
"""

from typing import Dict, List
import copy
import torch
import torch.nn as nn
import torch.optim as optim


class Client:
    def __init__(self, cid: int, model: nn.Module, train_loader, val_loader, test_loader, device='cpu', gamma: float = 0.9):
        self.cid = cid
        self.device = device
        self.model = copy.deepcopy(model).to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.gamma = gamma  # momentum for local running stats (match BN momentum)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None

        # prepare storage for local running stats of each DSE module by module name
        self._dse_module_names = []
        self._local_running = {}  # name -> {'mean': tensor, 'var': tensor}
        self._register_dse_modules()

    def _register_dse_modules(self):
        for name, module in self.model.named_modules():
            if hasattr(module, 'bn_dfe'):
                # initialize running stats (empty until batch seen)
                self._dse_module_names.append(name)
                self._local_running[name] = {'mean': None, 'var': None}

    def set_optimizer(self, lr=0.01):
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

    def _update_local_running(self, name: str, batch_mean: torch.Tensor, batch_var: torch.Tensor):
        st = self._local_running[name]
        if st['mean'] is None:
            st['mean'] = batch_mean.detach().clone().to(self.device)
            st['var'] = batch_var.detach().clone().to(self.device)
        else:
            st['mean'] = self.gamma * st['mean'] + (1.0 - self.gamma) * batch_mean.detach().to(self.device)
            st['var'] = self.gamma * st['var'] + (1.0 - self.gamma) * batch_var.detach().to(self.device)

    def compute_LCon(self, global_bn_stats: Dict[str, Dict[str, torch.Tensor]], beta: float = 0.001) -> torch.Tensor:
        """Compute LCon following Eq.(6)-(7) in the paper.

        - For each DSE module l: compute L^(l)_Con = 1/d ||mu_hat - mu_g||^2 + (||sigma_hat^2||_1 - ||sigma_g^2||_1 / d)^2
        - Then LCon = sum_l w_l * L^(l)_Con with wl = softmax(beta * l)

        global_bn_stats: mapping module_name -> {'mean': tensor, 'var': tensor} extracted from global model BNDFE
        """
        losses = []
        d_list = []
        for idx, name in enumerate(self._dse_module_names):
            local = self._local_running.get(name, None)
            if local is None or local['mean'] is None:
                continue
            if name not in global_bn_stats:
                continue
            mu_hat = local['mean']
            sigma_hat = local['var']
            mu_g = global_bn_stats[name]['mean'].to(self.device)
            sigma_g = global_bn_stats[name]['var'].to(self.device)
            d = mu_hat.numel()
            term1 = ((mu_hat - mu_g)**2).sum() / d
            term2 = ((sigma_hat.sum() - sigma_g.sum()) / d)**2
            losses.append((idx, term1 + term2))
            d_list.append(d)

        if not losses:
            return torch.tensor(0.0, device=self.device)

        # compute layer weights w_l via softmax(beta * l)
        betasz = beta
        idxs = torch.tensor([i for (i, _) in losses], dtype=torch.float32, device=self.device)
        w_raw = torch.softmax(betasz * idxs, dim=0)
        total = torch.tensor(0.0, device=self.device)
        for (i, term), w in zip(losses, w_raw):
            total = total + w * term
        return total

    def local_update(self, global_model: nn.Module, local_epochs: int = 1, lr: float = 0.01, lambda_con: float = 0.1, beta: float = 0.001):
        # load global parameters
        self.model.load_state_dict(global_model.state_dict(), strict=False)
        self.set_optimizer(lr)
        self.model.train()

        # attach forward hooks to collect per-batch stats from DSE modules
        handles = []
        def make_hook(name):
            def hook(module, inp, outp):
                t = outp.detach()
                if t.dim() >= 4:
                    # compute per-channel mean/var over batch+spatial
                    mean = t.mean(dim=[0,2,3])
                    var = t.var(dim=[0,2,3], unbiased=False)
                else:
                    mean = t.mean(dim=0)
                    var = t.var(dim=0, unbiased=False)
                # update local running stats via exponential moving average with momentum gamma
                self._update_local_running(name, mean, var)
            return hook

        for name, module in self.model.named_modules():
            if hasattr(module, 'bn_dfe'):
                handles.append(module.register_forward_hook(make_hook(name)))

        for epoch in range(local_epochs):
            # iterate batches
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(x)
                loss_task = self.criterion(out, y)
                # global BN stats should be attached on the passed global_model
                global_stats = getattr(global_model, '_bn_dfe_stats', {})
                loss_con = self.compute_LCon(global_stats, beta=beta)
                loss = loss_task + lambda_con * loss_con
                loss.backward()
                # 添加梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()

        # remove hooks
        for h in handles:
            h.remove()

        return copy.deepcopy(self.model.state_dict())

    def validate(self, return_raw: bool = False):
        """在验证集上评估模型性能"""
        return self._evaluate_on_loader(self.val_loader, return_raw)

    def evaluate(self, return_raw: bool = False):
        """在测试集上评估模型性能"""
        return self._evaluate_on_loader(self.test_loader, return_raw)

    def _evaluate_on_loader(self, loader, return_raw: bool = False):
        """在指定数据加载器上评估模型的通用方法"""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                out = self.model(x)
                loss = criterion(out, y)
                total_loss += loss.item() * x.size(0)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        avg_loss = total_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        if return_raw:
            return acc, avg_loss, correct, total
        else:
            return acc, avg_loss

    def get_bn_dfe_stats(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """获取当前客户端的BN-DFE统计信息"""
        bn_stats = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'bn_dfe') and isinstance(module.bn_dfe, (nn.BatchNorm1d, nn.BatchNorm2d)):
                bn_stats[name] = {
                    'running_mean': module.bn_dfe.running_mean.detach().cpu().clone(),
                    'running_var': module.bn_dfe.running_var.detach().cpu().clone()
                }
        return bn_stats