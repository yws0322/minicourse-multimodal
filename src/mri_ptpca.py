"""MRI-PTPCa model architecture, adapted from HIMF-Surv's `feature_extractors/mri.py`
(itself adapted from https://github.com/StandWisdom/MRI-based-Predicted-Transformer-for-Prostate-cancer).

This is exactly the kind of thing the project convention keeps out of notebooks: a large,
third-party CNN+ViT architecture with input-reshaping plumbing that isn't itself the lesson —
notebook 02 explains and visualizes *preprocessing* and *what gets extracted*, and imports the
model from here to do it.

HIMF-Surv registers forward hooks on every transformer block, mean-pools tokens per block, and
aggregates layers with a `[0.5, 0.75, 1.0]` grouping scheme. This course skips that: `extract_embedding`
hooks only the *last* transformer block, matching the "no layer-wise aggregation" simplification
used for the WSI (H-optimus-0) side too.
"""
import typing as tp
from pathlib import Path

import torch
import torchvision.models as models
from torch import nn
from vit_pytorch import SimpleViT
import einops

# Matches CHIMERA's mpMRI layout: 16 slices per scan, cropped/resized to 200x200.
IMG_SIZE = (16, 200, 200)  # (slices, height, width)
# Internal token dim of the ViT fusion transformer -- the embedding dimension notebook 02 saves.
EMBEDDING_DIM = 2048


class VisionNet(nn.Module):
    """Single-modality (T2W or ADC) CNN backbone: a MobileNetV3-Small feature extractor."""

    def __init__(self, batchsize: int = 1):
        super().__init__()
        self.myCNN = models.mobilenet_v3_small().features
        self.relu = nn.ReLU()
        self.batchsize = batchsize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "b z c w h -> (b z) c w h")
        x = self.relu(self.myCNN(x))
        return einops.rearrange(x, "(b z) c w h -> b z c w h", b=self.batchsize)


class CNNViTMM(nn.Module):
    """Multi-modality CNN-ViT model: fuses T2W + ADC + (zeroed) DWI via per-modality CNNs
    feeding a shared ViT transformer."""

    def __init__(self, batchsize: int = 1):
        super().__init__()
        self.batchsize = batchsize
        self.z_num = 16
        self.patch_size = 7

        self.conv_0 = nn.Conv2d(576, 3, kernel_size=1)
        self.conv_1 = nn.Conv2d(576, 3, kernel_size=1)
        self.conv_2 = nn.Conv2d(576, 3, kernel_size=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(3)

        self.net_t2 = VisionNet(batchsize)
        self.net_adc = VisionNet(batchsize)
        self.net_dwi = VisionNet(batchsize)

        self.myViT_MM = SimpleViT(
            image_size=(self.patch_size * 4, self.patch_size * 12),
            patch_size=self.patch_size,
            num_classes=6,  # unused -- embeddings are read from a transformer hook, not this head
            dim=EMBEDDING_DIM,
            depth=24,
            heads=12,
            mlp_dim=EMBEDDING_DIM,
        )

    def _encode_modalities(self, x: torch.Tensor):
        z = self.z_num
        t2, adc, dwi = x[:, 0 * z:1 * z], x[:, 1 * z:2 * z], x[:, 2 * z:3 * z]
        return self.net_t2(t2), self.net_adc(adc), self.net_dwi(dwi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of shape (B, 48, 3, H, W) -- T2(16) + ADC(16) + DWI(16) concatenated on dim 1."""
        x0, x1, x2 = self._encode_modalities(x)

        def project(xi, conv):
            xi = einops.rearrange(xi, "b z c w h -> (b z) c w h")
            xi = self.relu(conv(xi))
            xi = self.bn(xi)
            return einops.rearrange(xi, "(b z) c w h -> b z c w h", b=self.batchsize)

        x0, x1, x2 = project(x0, self.conv_0), project(x1, self.conv_1), project(x2, self.conv_2)
        combined = torch.cat((x0, x1, x2), dim=1)
        combined = einops.rearrange(
            combined, "b (z1 z2) c w h -> b c (z1 w) (z2 h)",
            z1=4, h=self.patch_size, w=self.patch_size,
        )
        return self.myViT_MM(combined)


def load_mri_ptpca_model(
    t2_model_path: tp.Optional[str] = None,
    adc_model_path: tp.Optional[str] = None,
    vit_model_path: tp.Optional[str] = None,
    device: str = "cpu",
) -> CNNViTMM:
    """Build the model (batch size fixed at 1, for per-patient inference) and load any provided
    pretrained weights. Weights are downloadable from the original MRI-PTPCa repo:
    https://github.com/StandWisdom/MRI-based-Predicted-Transformer-for-Prostate-cancer/tree/main/pretrained_weights

    Without weights, the CNN backbones fall back to randomly initialized MobileNetV3 -- fine for
    exercising the pipeline, but not for meaningful embeddings.
    """
    model = CNNViTMM(batchsize=1)

    if t2_model_path and Path(t2_model_path).exists():
        model.net_t2.load_state_dict(torch.load(t2_model_path, map_location=device))
    if adc_model_path and Path(adc_model_path).exists():
        model.net_adc.myCNN.load_state_dict(torch.load(adc_model_path, map_location=device))
    if vit_model_path and Path(vit_model_path).exists():
        model.myViT_MM.load_state_dict(torch.load(vit_model_path, map_location=device), strict=False)

    return model.to(device).eval()


def extract_embedding(
    model: CNNViTMM,
    t2_tensor: torch.Tensor,
    adc_tensor: torch.Tensor,
    dwi_tensor: tp.Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Run the model and return the last transformer block's mean-pooled token output.

    `dwi_tensor` is the third (high-b-value/DWI) modality the model was pretrained expecting --
    CHIMERA provides this as the `_hbv.mha` series. If it's not available for a given scan, pass
    `None` to zero-fill instead (matching HIMF-Surv's fallback when a series is missing).

    Returns a single `EMBEDDING_DIM`-length vector (no per-layer aggregation).
    """
    t2_tensor, adc_tensor = t2_tensor.to(device), adc_tensor.to(device)
    dwi_tensor = dwi_tensor.to(device) if dwi_tensor is not None else torch.zeros_like(adc_tensor)
    combined_input = torch.cat((t2_tensor, adc_tensor, dwi_tensor), dim=1)

    captured = {}

    def hook_fn(module, inp, output):
        captured["last_layer"] = output.detach()

    last_attention = model.myViT_MM.transformer.layers[-1][0]
    handle = last_attention.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(combined_input)
    finally:
        handle.remove()

    return captured["last_layer"].mean(dim=1).squeeze(0)  # (1, N, D) -> (D,)
