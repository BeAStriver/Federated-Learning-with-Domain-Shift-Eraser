"""fdse_model.py

Implementation of the FDSE model architecture (DSEBlock + AlexNet backbone)
based on "Federated Learning with Domain Shift Eraser" (Wang et al., CVPR 2025)
Supplemental: Table A2 and Table A3.

This file provides:
- DSEBlockConv: convolutional DSEBlock as described in Table A3
- DSEBlockLinear: a linear (FC) variant used for fully-connected layers
- FDSEAlexNet: AlexNet-like backbone where each original layer is replaced by a DSEBlock
- build_fdse_alexnet: convenience constructor

Notes:
- This file focuses on model structure / forward pass only (PyTorch).
- It preserves the decomposition idea: each original layer -> DFE (final BN+ReLU) and a light-weight DSE expansion.
- Hyperparameters: G (expansion factor groups), dw (kernel size for cheap conv branch) are exposed.

Author: generated for reproduction by assistant
"""

from math import ceil, sqrt
from typing import Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models


class DSEBlockConv(nn.Module):
    """DSEBlock for 2D convolutional layers.

    Implements the decomposition in Table A3:
      1) Conv2d(S, ceil(T/G), kernel, stride, padding), BNDSE, ReLU
      2) Conv2d(ceil(T/G), T - ceil(T/G), dw, 1, dw//2)  # cheap op branch
      3) Concat(out1, out2)
      4) BNDFE(T), ReLU

    Parameters
    ----------
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    G: int expansion param (default 2)
    dw: int kernel size for cheap branch (default 3)
    bias: bool for convolutions
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int,
            G: int = 2,
            dw: int = 3,
            bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.G = max(1, int(G))
        # number of channels produced by the first (DFE) conv
        self.mid_channels = int(ceil(out_channels / self.G))
        self.dw = dw

        # First branch: core extractor -> small number of channels
        self.conv_core = nn.Conv2d(
            in_channels, self.mid_channels, kernel_size, stride, padding, bias=bias
        )
        self.bn_dse = nn.BatchNorm2d(self.mid_channels)
        # cheap expansion branch: from mid_channels -> remaining channels
        cheap_out = out_channels - self.mid_channels
        # if cheap_out == 0, we still create an identity-like conv (1x1)
        if cheap_out > 0:
            self.conv_cheap = nn.Conv2d(
                self.mid_channels, cheap_out, kernel_size=self.dw, stride=1, padding=self.dw // 2, bias=bias
            )
        else:
            # placeholder conv that outputs zero channels (won't be used)
            self.conv_cheap = None

        # Final BN + ReLU (the DFE module's BN)
        self.bn_dfe = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Initialize weights with standard practice
        self._init_weights()

    def _init_weights(self):
        # follow common conv init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # core path
        core = self.conv_core(x)
        core = self.bn_dse(core)
        core = F.relu(core, inplace=True)

        # cheap expansion from each core channel to multiple variants
        if self.conv_cheap is not None:
            cheap = self.conv_cheap(core)
        else:
            # produce a zero-tensor of appropriate shape to concat (shouldn't usually happen)
            B, C, H, W = core.shape
            cheap = core.new_zeros((B, self.out_channels - self.mid_channels, H, W))

        # concat the core and the cheap branch to reconstruct out_channels
        out = torch.cat([core, cheap], dim=1)
        # if concatenation didn't reach out_channels (numerical), pad/truncate
        if out.shape[1] != self.out_channels:
            # adjust by slicing or padding
            if out.shape[1] > self.out_channels:
                out = out[:, : self.out_channels, :, :]
            else:
                pad_ch = self.out_channels - out.shape[1]
                pad_tensor = out.new_zeros((out.shape[0], pad_ch, out.shape[2], out.shape[3]))
                out = torch.cat([out, pad_tensor], dim=1)

        out = self.bn_dfe(out)
        out = F.relu(out, inplace=True)
        return out


class DSEBlockLinear(nn.Module):
    """DSEBlock variant for fully-connected layers.

    Decomposes a Linear(in, out) into a light DFE+DSE style module.
    We implement a simple decomposition:
        1) Linear(in, mid), BN, ReLU
        2) Linear(mid, out-mid) as cheap expansion
        3) Concat -> BN(out) -> ReLU

    This matches the convolutional idea but in a linear form and is used
    for FC layers 7 and 8 in the FDSE AlexNet from the supplement.
    """

    def __init__(self, in_features: int, out_features: int, G: int = 2, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.G = max(1, int(G))
        self.mid = int(ceil(out_features / self.G))

        self.lin_core = nn.Linear(in_features, self.mid, bias=bias)
        self.bn_dse = nn.BatchNorm1d(self.mid)
        cheap_out = out_features - self.mid
        if cheap_out > 0:
            self.lin_cheap = nn.Linear(self.mid, cheap_out, bias=bias)
        else:
            self.lin_cheap = None

        self.bn_dfe = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=sqrt(5))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, in_features)
        core = self.lin_core(x)
        # BatchNorm1d expects (B, C)
        core = self.bn_dse(core)
        core = F.relu(core, inplace=True)

        if self.lin_cheap is not None:
            cheap = self.lin_cheap(core)
        else:
            cheap = x.new_zeros((x.shape[0], self.out_features - self.mid))

        out = torch.cat([core, cheap], dim=1)
        if out.shape[1] != self.out_features:
            if out.shape[1] > self.out_features:
                out = out[:, : self.out_features]
            else:
                pad_ch = self.out_features - out.shape[1]
                pad_tensor = out.new_zeros((out.shape[0], pad_ch))
                out = torch.cat([out, pad_tensor], dim=1)

        out = self.bn_dfe(out)
        out = F.relu(out, inplace=True)
        return out


class FDSEAlexNet(nn.Module):
    """AlexNet backbone where each layer is replaced by a DSEBlock as in Table A2.

    Structure follows the supplemental material Table A2:
    1: DSEBlock(3, 64, 11, 4, 2, G=2, dw=3), MaxPool2D(3,2)
    2: DSEBlock(64,192,5,1,2,G=2,dw=3), MaxPool2D(3,2)
    3: DSEBlock(192,384,3,1,1,G=2,dw=3)
    4: DSEBlock(384,256,3,1,1,G=2,dw=3)
    5: DSEBlock(256,256,3,1,1,G=2,dw=3), MaxPool2D(3,2)
    6: AdaptiveAvgPool2D(6,6)
    7: DSEBlockLinear(9216,1024,G=2)
    8: DSEBlockLinear(1024,1024,G=2)
    9: FC(1024, num_classes)
    """

    def __init__(self, num_classes: int = 1000, G: int = 2, dw: int = 3):
        super().__init__()
        self.G = G
        self.dw = dw

        # conv DSE blocks
        self.layer1 = nn.Sequential(
            DSEBlockConv(3, 64, kernel_size=11, stride=4, padding=2, G=self.G, dw=self.dw),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer2 = nn.Sequential(
            DSEBlockConv(64, 192, kernel_size=5, stride=1, padding=2, G=self.G, dw=self.dw),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer3 = DSEBlockConv(192, 384, kernel_size=3, stride=1, padding=1, G=self.G, dw=self.dw)
        self.layer4 = DSEBlockConv(384, 256, kernel_size=3, stride=1, padding=1, G=self.G, dw=self.dw)
        self.layer5 = nn.Sequential(
            DSEBlockConv(256, 256, kernel_size=3, stride=1, padding=1, G=self.G, dw=self.dw),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # fully connected / classifier part implemented by linear DSE blocks
        # Note: after avgpool the flattened vector has size 256 * 6 * 6 = 9216 (as in supplement)
        self.fc7 = DSEBlockLinear(256 * 6 * 6, 1024, G=self.G)
        self.fc8 = DSEBlockLinear(1024, 1024, G=self.G)
        self.classifier = nn.Linear(1024, num_classes)

        # initialize classifier
        nn.init.normal_(self.classifier.weight, 0, 0.01)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv feature extraction
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc7(x)
        x = self.fc8(x)
        x = self.classifier(x)
        return x

# --- helper to copy conv feature weights from pretrained alexnet to FDSE model ---
def load_pretrained_alexnet_features(pretrained_alex: nn.Module, fdse_model: nn.Module):
    """
    Copy Conv2d weights from pretrained_alex.features to fdse_model's DSEBlocks conv_core.
    Matching is done by traversal order: the i-th Conv2d in pretrained.features -> i-th DSEBlock.conv_core.
    """
    # collect conv modules from pretrained alexnet.features in order
    pretrained_convs = []
    for m in pretrained_alex.features.modules():
        if isinstance(m, nn.Conv2d):
            pretrained_convs.append(m)

    # collect target conv_core modules from fdse_model in order
    target_convs = []
    # we expect DSEBlockConv instances to have attribute conv_core
    for name, module in fdse_model.named_modules():
        # DSEBlockConv: has attribute 'conv_core'
        if hasattr(module, "conv_core") and isinstance(getattr(module, "conv_core"), nn.Conv2d):
            target_convs.append((name, module.conv_core))

    if len(pretrained_convs) != len(target_convs):
        # It's possible the number differs because FC DSEBlocks are linear; only copy as many as match
        # We'll copy min(len(pretrained_convs), len(target_convs))
        n_copy = min(len(pretrained_convs), len(target_convs))
    else:
        n_copy = len(pretrained_convs)

    # perform copying
    for i in range(n_copy):
        src = pretrained_convs[i]
        tgt_name, tgt = target_convs[i]
        # check shape compatibility (out_channels, in_channels, kH, kW)
        if src.weight.shape == tgt.weight.shape:
            tgt.weight.data.copy_(src.weight.data)
            if src.bias is not None and tgt.bias is not None:
                tgt.bias.data.copy_(src.bias.data)
            else:
                # if target has bias but src doesn't or vice versa, skip bias copy
                pass
        else:
            # if shapes mismatch, try compatible partial copy if possible (rare). Otherwise skip and warn.
            if src.weight.shape == tgt.weight.shape:
                tgt.weight.data.copy_(src.weight.data)
                if src.bias is not None and tgt.bias is not None:
                    tgt.bias.data.copy_(src.bias.data)
            else:
                # partial copy: copy first tgt.out_channels channels from src
                so = src.weight.shape  # (Out_src, In, kH, kW)
                to = tgt.weight.shape  # (Out_tgt, In, kH, kW)
                ocopy = min(so[0], to[0])
                icopy = min(so[1], to[1])
                # copy weight[:ocopy, :icopy, :, :] -> tgt.weight[:ocopy, :icopy, :, :]
                tgt.weight.data[:ocopy, :icopy, :, :].copy_(src.weight.data[:ocopy, :icopy, :, :])
                if src.bias is not None and tgt.bias is not None:
                    tgt.bias.data[:ocopy].copy_(src.bias.data[:ocopy])
                print(f"[Model] Partial-copied {ocopy}/{to[0]} out-channels for layer {i} (src->{tuple(so)} -> tgt->{tuple(to)})")

    print(f"[Model] Copied {n_copy} conv layers from pretrained AlexNet to FDSE model.")


# --- updated build function ---
def build_fdse_alexnet(num_classes: int = 10, G: int = 2, dw: int = 3, pretrained: bool = False) -> FDSEAlexNet:
    """
    Build FDSE AlexNet.
    If pretrained==True, load torchvision alexnet pretrained weights and copy conv features into FDSE conv_core modules.
    """
    if pretrained:
        print("[Model] Loading ImageNet-pretrained AlexNet backbone...")
        try:
            base_alex = tv_models.alexnet(weights=tv_models.AlexNet_Weights.IMAGENET1K_V1)
        except Exception:
            # fallback for older torchvision
            base_alex = tv_models.alexnet(pretrained=True)
    else:
        print("[Model] Using randomly initialized AlexNet backbone...")
        base_alex = tv_models.alexnet(weights=None)

    # build FDSE model (this uses the DSEBlockConv / DSEBlockLinear implementation you already have)
    model = FDSEAlexNet(num_classes=num_classes, G=G, dw=dw)

    # if requested, copy conv features from base_alex -> fdse model
    if pretrained:
        load_pretrained_alexnet_features(base_alex, model)

    return model


if __name__ == "__main__":
    # quick smoke test
    model = build_fdse_alexnet(num_classes=10, G=2, dw=3)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)
