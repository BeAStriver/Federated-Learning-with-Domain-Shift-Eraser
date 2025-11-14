"""federated/utils.py

Utilities for FDSE federated reproduction.
- parameter grouping: identify DFE (shared) vs DSE (personalized) parameters by module attributes
- flatten / unflatten helpers for param lists
- projection onto simplex (for solving MGDA weights)

This module is designed to be used by aggregator.py, client.py, server.py.
"""

from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import math


def flatten_params(params: List[nn.Parameter]) -> torch.Tensor:
    """Flatten a list of parameters to a single 1D tensor (CPU tensor)."""
    vecs = [p.detach().cpu().view(-1) for p in params]
    if len(vecs) == 0:
        return torch.tensor([], dtype=torch.float32)
    return torch.cat(vecs, dim=0)


def unflatten_to_state_dict(flat: torch.Tensor, template_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Unflatten a 1D tensor into a state_dict-like mapping with shapes from template_state_dict.

    `template_state_dict` should be an ordered mapping of name->tensor whose shapes will be used.
    """
    out = {}
    idx = 0
    for k, v in template_state_dict.items():
        n = v.numel()
        out[k] = flat[idx:idx+n].view_as(v).clone()
        idx += n
    return out


def split_dfe_dse_parameters(model: nn.Module) -> Tuple[List[str], List[str]]:
    """Return two lists of parameter names: (dfe_param_names, dse_param_names).

    Heuristic (matching paper Table A3 and supplement): for modules that are DSEBlock-like
    (we expect them to expose attributes `conv_core`/`lin_core` and `bn_dfe`), treat `conv_core`/`lin_core`
    and `bn_dfe` as DFE (shared); treat the remaining submodules under the block as DSE (personalized).

    Additionally, classifier final FC (the last Linear layer) is shared (DFE).

    This function returns lists of parameter keys as present in model.state_dict().
    """
    dfe_names = []
    dse_names = []
    sd = model.state_dict()

    # find DSE-like modules
    dse_module_names = []
    for name, module in model.named_modules():
        if hasattr(module, 'bn_dfe'):
            dse_module_names.append(name)

    # assign parameters
    for key in sd.keys():
        assigned = False
        for mod_name in dse_module_names:
            # parameter keys are like 'layer1.0.conv_core.weight' or similar; match by prefix
            if key.startswith(mod_name + '.'):
                # if contains conv_core or bn_dfe in the suffix => DFE
                suffix = key[len(mod_name) + 1 :]
                if suffix.startswith('conv_core') or suffix.startswith('lin_core') or suffix.startswith('bn_dfe'):
                    dfe_names.append(key)
                else:
                    dse_names.append(key)
                assigned = True
                break
        if not assigned:
            # default: classifier and others -> DFE
            dfe_names.append(key)

    # ensure disjoint
    dfe_names = list(dict.fromkeys(dfe_names))
    dse_names = list(dict.fromkeys(dse_names))
    return dfe_names, dse_names


def project_onto_simplex(v: torch.Tensor, s: float = 1.0) -> torch.Tensor:
    """Euclidean projection of vector v onto the probability simplex {x : x>=0, sum x = s}.
    Implementation based on: Wang and Carreira-Perpinan (2013) - fast projection.
    """
    if v.numel() == 0:
        return v
    u = torch.sort(v, descending=True)[0]
    sv = torch.cumsum(u, dim=0)
    rho = torch.nonzero(u * torch.arange(1, v.numel()+1, device=v.device) > (sv - s), as_tuple=False)
    if rho.numel() == 0:
        theta = 0.0
    else:
        rho = rho[-1].item()
        theta = (sv[rho] - s) / (rho + 1)
    w = torch.clamp(v - theta, min=0.0)
    return w


# small utility to compute squared norm across params by name
def compute_flat_param_from_state(state: Dict[str, torch.Tensor], param_names: List[str]) -> torch.Tensor:
    vecs = [state[n].detach().cpu().view(-1) for n in param_names]
    if not vecs:
        return torch.tensor([], dtype=torch.float32)
    return torch.cat(vecs, dim=0)
