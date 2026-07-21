# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Large Selective Kernel Network backbone.

This is an Ultralytics-compatible adaptation of the official LSKNet backbone:
https://github.com/zcablii/LSKNet

The upstream implementation is licensed CC BY-NC 4.0. This adapted module is
therefore for non-commercial use unless separate permission is obtained from
the LSKNet authors.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop residual paths independently per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Stochastic depth applied to residual branches."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth."""
        return drop_path(x, self.drop_prob, self.training)


class LSKMlp(nn.Module):
    """Convolutional MLP used by an LSK block."""

    def __init__(self, channels: int, hidden_channels: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(channels, hidden_channels, 1)
        self.dwconv = LSKDWConv(hidden_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_channels, channels, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pointwise, depthwise, and pointwise projections."""
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class LSKDWConv(nn.Module):
    """Depthwise convolution with official LSKNet checkpoint key names."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply depthwise convolution."""
        return self.dwconv(x)


class LSKSpatialGate(nn.Module):
    """Select between local and dilated large-kernel spatial features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = channels // 2
        self.conv0 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.conv_spatial = nn.Conv2d(channels, channels, 7, padding=9, groups=channels, dilation=3)
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.conv2 = nn.Conv2d(channels, hidden, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply selective large-kernel spatial gating."""
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        features = torch.cat((attn1, attn2), dim=1)
        pooled = torch.cat(
            (torch.mean(features, dim=1, keepdim=True), torch.max(features, dim=1, keepdim=True).values),
            dim=1,
        )
        weights = self.conv_squeeze(pooled).sigmoid()
        selected = attn1 * weights[:, 0:1] + attn2 * weights[:, 1:2]
        return x * self.conv(selected)


class LSKAttention(nn.Module):
    """Residual large selective kernel attention."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj_1 = nn.Conv2d(channels, channels, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKSpatialGate(channels)
        self.proj_2 = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial gating between pointwise projections."""
        shortcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        return self.proj_2(x) + shortcut


class LSKBlock(nn.Module):
    """One LSKNet attention and MLP block."""

    def __init__(self, channels: int, mlp_ratio: float, drop: float, drop_path_rate: float) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.attn = LSKAttention(channels)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.mlp = LSKMlp(channels, int(channels * mlp_ratio), drop)
        self.layer_scale_1 = nn.Parameter(1e-2 * torch.ones(channels))
        self.layer_scale_2 = nn.Parameter(1e-2 * torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and MLP residual branches."""
        scale1 = self.layer_scale_1[:, None, None]
        scale2 = self.layer_scale_2[:, None, None]
        x = x + self.drop_path(scale1 * self.attn(self.norm1(x)))
        return x + self.drop_path(scale2 * self.mlp(self.norm2(x)))


class OverlapPatchEmbed(nn.Module):
    """Overlapping convolutional patch embedding."""

    def __init__(self, in_channels: int, out_channels: int, patch_size: int, stride: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and normalize an image or feature map."""
        return self.norm(self.proj(x))


class LSKNet(nn.Module):
    """Four-stage LSKNet backbone returning P2/4 through P5/32 feature maps."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: list[int] = (32, 64, 160, 256),
        mlp_ratios: list[float] = (8.0, 8.0, 4.0, 4.0),
        depths: list[int] = (3, 3, 5, 2),
        drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if not (len(embed_dims) == len(mlp_ratios) == len(depths) == 4):
            raise ValueError("LSKNet requires four embed_dims, mlp_ratios, and depths values")

        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        offset = 0
        for stage in range(4):
            patch_embed = OverlapPatchEmbed(
                in_channels if stage == 0 else embed_dims[stage - 1],
                embed_dims[stage],
                patch_size=7 if stage == 0 else 3,
                stride=4 if stage == 0 else 2,
            )
            blocks = nn.ModuleList(
                LSKBlock(
                    embed_dims[stage],
                    mlp_ratios[stage],
                    drop_rate,
                    rates[offset + index],
                )
                for index in range(depths[stage])
            )
            offset += depths[stage]
            setattr(self, f"patch_embed{stage + 1}", patch_embed)
            setattr(self, f"block{stage + 1}", blocks)
            setattr(self, f"norm{stage + 1}", nn.LayerNorm(embed_dims[stage]))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return the four multiscale backbone features."""
        outputs = []
        for stage in range(1, 5):
            x = getattr(self, f"patch_embed{stage}")(x)
            for block in getattr(self, f"block{stage}"):
                x = block(x)
            batch, _, height, width = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = getattr(self, f"norm{stage}")(x)
            x = x.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
            outputs.append(x)
        return outputs
