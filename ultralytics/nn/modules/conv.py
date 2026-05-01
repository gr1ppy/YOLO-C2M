# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

from __future__ import annotations

import math
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import torchvision.ops
from torchvision.ops import DeformConv2d
from torch import einsum
from einops import rearrange, repeat
from typing import List, Optional
#from DCNv4.modules import DCNv4

__all__ = (
    "CBAM",
    "ChannelAttention",
    "Concat",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DWConv",
    "DWConvTranspose2d",
    "Focus",
    "GhostConv",
    "Index",
    "LightConv",
    "RepConv",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "Index",
    "h_sigmoid",
    "h_swish",
    "CoordAtt",
    "GaborConv",
    "AAttn",
    "GaborConvMax",
    "UP",
    "Masked_Downsample",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class UP(nn.Module):
    def __init__(self, c1, c2, n: int=2):
        super().__init__()
        self.up = nn.Upsample(None, n, 'nearest')
    def forward(self, x):
        return self.up(x)






class Masked_Downsample(nn.Module):
    def __init__(self, in_channels, out_channels=None, pool_kernel=3, pool_stride=2, pos=False, act=nn.SiLU()):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pos = pos
        self.mask_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, groups=in_channels, bias=True)

        # learnable scale (global)
        self.scale = nn.Parameter(torch.zeros(in_channels,1,1))  # init 0 => no effect initially
        # If you prefer per-channel: self.scale = nn.Parameter(torch.zeros(in_channels,1,1))

        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride, padding=1)
        self.down = Conv(in_channels, out_channels, k=1, s=1)
        # self.attn = SEBlock(out_channels)

    def forward(self, x):
        mask = F.adaptive_avg_pool2d(x, (1, 1))  # shape (B, C, 1, 1)
        mask_out = self.mask_conv(mask)
        scaled = mask_out * self.scale
        if self.pos:
            y = x + scaled
        else:
            y = x - scaled

        y = self.pool(y)
        y = self.down(y)
        return y
        # return self.attn(y)


class SEBlock(nn.Module):
    def __init__(self, c_in, reduction=32):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.Sigmoid()
        )


class AAttn(nn.Module):
    """
    Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """
        Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class GaborConvold(nn.Module):
    """
    Depthwise Gabor: one Gabor kernel per input channel (unique orientation per channel).
    - sigma: single sigma value (float) used for all channels (no multiple sigmas).
    - lambd: wavelength passed to cv2.getGaborKernel.
    - Each input channel gets exactly one Gabor kernel with orientation theta_i = i * pi / c_in.
    - This is equivalent to a depthwise conv with one kernel per input channel.
    - No reduction (max/avg/sum) needed because filters_per_channel == 1.
    """

    def __init__(
            self,
            c_in: int,
            target_c_out: int,
            kernel_size: int = 5,
            sigma: float = 0.1,  # single sigma
            gamma: float = 0.5,
            lambd: float = 10.0,  # Gabor wavelength
            psi: float = 0.0,
            use_bias: bool = False,
            trainable_gabor: bool = False,
    ):
        super().__init__()
        self.c_in = int(c_in)
        self.target_c_out = int(target_c_out)
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.lambd = float(lambd)
        self.psi = float(psi)
        self.use_bias = use_bias
        self.trainable_gabor = bool(trainable_gabor)

        self.groups = self.c_in

        # Depthwise conv: groups = c_in, out_channels = c_in
        self.gabor_dw_conv = nn.Conv2d(
            in_channels=self.c_in,
            out_channels=self.c_in,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.groups,
            bias=self.use_bias
        )

        # Build one Gabor kernel per input channel with distinct orientation
        # Orientation for channel i: theta = i * pi / c_in (evenly spaced in [0, pi))
        gabor_kernels = []
        for i in range(self.c_in):
            theta = np.pi
            k = cv2.getGaborKernel(
                (self.kernel_size, self.kernel_size),
                self.sigma,
                theta,
                self.lambd,
                self.gamma,
                self.psi,
                ktype=cv2.CV_32F
            )
            gabor_kernels.append(k.astype(np.float32))

        # gabor_kernels shape -> (c_in, k, k)
        # For grouped conv with groups=c_in we need weight shape (out_channels, 1, k, k) where out_channels == c_in
        weight_np = np.stack(gabor_kernels, axis=0).reshape(self.c_in, 1, self.kernel_size, self.kernel_size)

        # Assign to conv weight
        weight_tensor = torch.from_numpy(weight_np)
        with torch.no_grad():
            self.gabor_dw_conv.weight.copy_(weight_tensor)
        self.gabor_dw_conv.weight.requires_grad = bool(self.trainable_gabor)

        # Post-conv BN + activation
        # self.bn_after = nn.BatchNorm2d(self.target_c_out)
        # self.act_after = nn.SiLU()
        # self.con = Conv(self.c_in, self.c_in, 1, 1)

        # Optional projection from c_in -> target_c_out
        # if self.target_c_out != self.c_in:
        self.proj = nn.Conv2d(self.c_in, self.c_in, kernel_size=1, stride=1, padding=0, groups=self.c_in, bias=False)
        # self.bn_proj = nn.BatchNorm2d(self.target_c_out)
        self.act_proj = nn.SiLU()
        # else:
        # self.proj = nn.Identity()
        # self.bn_proj = nn.Identity()
        # self.act_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, c_in, H, W)
        returns: (B, target_c_out, H, W)
        """
        # dw = self.con(x)

        # Depthwise conv -> (B, c_in, H, W) because filters_per_channel == 1
        dw = self.gabor_dw_conv(x)

        # Projection to desired output channels
        dw = self.proj(dw)
        # out = self.bn_proj(out)
        dw = self.act_proj(dw)
        return dw


class GaborConvMax(nn.Module):
    """
    Depthwise Gabor bank per input channel -> channel-wise reduction -> optional projection.

    - `scales` are treated as SIGMA values for cv2.getGaborKernel.
    - `lambd` is the wavelength (lambda) passed to cv2.getGaborKernel.
    - For each input channel we build (orientations * num_sigmas) kernels.
    - Uses grouped conv with groups=c_in and out_channels = c_in * filters_per_channel.
    - After conv: reshape (B, c_in, filters_per_channel, H, W) and reduce over filters to (B, c_in, H, W).
    - Optional 1x1 projection to target_c_out.
    - `reduce` controls reduction method: "max" (default), "avg", "sum", or "learned".
    """

    def __init__(
        self,
        c_in: int,
        target_c_out: int,
        kernel_size: int = 5,
        orientations: int = 8,
        scales=(0.5, 1.0),     # SIGMA values
        gamma: float = 0.5,
        lambd: float = 10.0,    # wavelength (lambda) for Gabor
        psi: float = 0.0,
        use_bias: bool = False,
        trainable_gabor: bool = False,
        reduce: str = "max",
        eps_sigma: float = 1e-4,
    ):
        super().__init__()
        assert reduce in ("max", "avg", "sum")

        self.c_in = int(c_in)
        self.target_c_out = int(target_c_out)
        self.kernel_size = int(kernel_size)
        self.orientations = int(orientations)
        # treat 'scales' as sigma values; avoid zeros
        self.sigmas = [float(s) if float(s) > eps_sigma else eps_sigma for s in scales]
        self.num_sigmas = len(self.sigmas)
        self.filters_per_channel = self.orientations * self.num_sigmas

        # wavelength (lambda in Gabor)
        self.lambd = float(lambd)
        self.gamma = float(gamma)
        self.psi = float(psi)

        self.use_bias = use_bias
        self.trainable_gabor = bool(trainable_gabor)
        self.reduce = reduce

        # Depthwise conv parameters
        self.dw_out_channels = self.c_in * self.filters_per_channel
        self.groups = self.c_in

        # Depthwise conv: each group corresponds to one input channel
        self.gabor_dw_conv = nn.Conv2d(
            in_channels=self.c_in,
            out_channels=self.dw_out_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.groups,
            bias=self.use_bias
        )

        # Build Gabor kernels for one input channel (filters_per_channel, k, k)
        theta_values = [i * np.pi / self.orientations for i in range(self.orientations)]
        gabor_kernels = []
        for sigma in self.sigmas:
            for theta in theta_values:
                # cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, psi)
                k = cv2.getGaborKernel(
                    (self.kernel_size, self.kernel_size),
                    sigma,
                    theta,
                    self.lambd,
                    self.gamma,
                    self.psi,
                    ktype=cv2.CV_32F
                )
                gabor_kernels.append(k.astype(np.float32))

        single_channel_kernels = np.stack(gabor_kernels, axis=0)  # (filters_per_channel, k, k)

        # Expand per input channel -> shape (c_in * filters_per_channel, 1, k, k)
        weight_np = np.repeat(single_channel_kernels[np.newaxis, ...], repeats=self.c_in, axis=0)
        weight_np = weight_np.reshape(self.dw_out_channels, 1, self.kernel_size, self.kernel_size)

        # Convert to tensor and assign to conv weight (PyTorch will move it when module.to(device) is called)
        weight_tensor = torch.from_numpy(weight_np)
        with torch.no_grad():
            self.gabor_dw_conv.weight.copy_(weight_tensor)
        self.gabor_dw_conv.weight.requires_grad = bool(self.trainable_gabor)

        # Post-reduction BN + activation
        self.bn_after = nn.BatchNorm2d(self.c_in)
        self.act_after = nn.SiLU()

        # Optional projection from c_in -> target_c_out
        if self.target_c_out != self.c_in:
            self.proj = nn.Conv2d(self.c_in, self.target_c_out, kernel_size=1, bias=self.use_bias)
            self.bn_proj = nn.BatchNorm2d(self.target_c_out)
            self.act_proj = nn.SiLU()
        else:
            self.proj = nn.Identity()
            self.bn_proj = nn.Identity()
            self.act_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, c_in, H, W)
        returns: (B, target_c_out, H, W)
        """
        B, C, H, W = x.shape
        assert C == self.c_in, f"Expected input channels {self.c_in}, got {C}"

        # Depthwise conv -> (B, c_in * filters_per_channel, H, W)
        dw = self.gabor_dw_conv(x)

        # Reshape -> (B, c_in, filters_per_channel, H, W)
        dw_view = dw.view(B, self.c_in, self.filters_per_channel, H, W)

        # Reduce across filters per-channel
        if self.reduce == "max":
            per_channel, _ = torch.max(dw_view, dim=2)   # (B, c_in, H, W)
        elif self.reduce == "avg":
            per_channel = dw_view.mean(dim=2)
        elif self.reduce == "sum":
            per_channel = dw_view.sum(dim=2)

        # BN + activation
        out = self.bn_after(per_channel)
        out = self.act_after(out)

        # Projection (if any)
        out = self.proj(out)
        out = self.bn_proj(out)
        out = self.act_proj(out)

        return out


class GaborConv(nn.Module):
    def __init__(
            self,
            c_in: int,
            target_c_out: int,
            kernel_size: int = 5,
            sigma: float = 0.1,  # single sigma
            gamma: float = 0.5,
            lambd: float = 10.0,  # Gabor wavelength
            psi: float = 0.0,
            use_bias: bool = False,
            trainable_gabor: bool = False,
    ):
        super().__init__()
        self.c_in = int(c_in)
        self.target_c_out = int(target_c_out)
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.lambd = float(lambd)
        self.psi = float(psi)
        self.use_bias = use_bias
        self.trainable_gabor = bool(trainable_gabor)

        self.groups = self.c_in

        # Depthwise conv: groups = c_in, out_channels = c_in
        self.gabor_dw_conv = nn.Conv2d(
            in_channels=self.c_in,
            out_channels=self.c_in,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.groups,
            bias=self.use_bias
        )

        gabor_kernels = []
        for i in range(self.c_in):
            theta = np.pi
            k = cv2.getGaborKernel(
                (self.kernel_size, self.kernel_size),
                self.sigma,
                theta,
                self.lambd,
                self.gamma,
                self.psi,
                ktype=cv2.CV_32F
            )
            gabor_kernels.append(k.astype(np.float32))

        weight_np = np.stack(gabor_kernels, axis=0).reshape(self.c_in, 1, self.kernel_size, self.kernel_size)

        # Assign to conv weight
        weight_tensor = torch.from_numpy(weight_np)
        with torch.no_grad():
            self.gabor_dw_conv.weight.copy_(weight_tensor)
        self.gabor_dw_conv.weight.requires_grad = bool(self.trainable_gabor)

        self.proj = nn.Conv2d(self.c_in, self.c_in, kernel_size=1, stride=1, padding=0, groups=self.c_in, bias=False)
        self.act_proj = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dw = self.gabor_dw_conv(x)
        dw = self.proj(dw)
        dw = self.act_proj(dw)
        return dw


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d( in_tensor, in_tensor.size(2), stride=in_tensor.size(2) )
        return self.gate_c( avg_pool ).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)

class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module( 'gate_s_bn_reduce0',	nn.BatchNorm2d(gate_channel//reduction_ratio) )
        self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
						padding=dilation_val, dilation=dilation_val) )
            self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
            self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
    def forward(self, in_tensor):
        return self.gate_s( in_tensor ).expand_as(in_tensor)
class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)
    def forward(self,x):
        att = 1 + F.sigmoid( self.channel_att(x) * self.spatial_att(x) )
        return att * x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu6(x + 3) / 6.0

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.act = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.act(x)

class CoordAtt(nn.Module):
    """
    Robust CoordAtt:
      - inp: input channels
      - oup: output channels. If oup != inp, a 1x1 conv projects identity before multiplication.
      - reduction: bottleneck factor
    """
    def __init__(self, inp, oup=None, reduction=32):
        super().__init__()
        if oup is None:
            oup = inp
        self.inp = inp
        self.oup = oup

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        # convs producing attention maps for height and width
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=True)

        # if oup != inp, project identity to oup channels before multiplication
        if oup != inp:
            self.project = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.project = None

    def forward(self, x):
        # x: (N, C, H, W)
        identity = x
        n, c, h, w = x.size()

        # adaptive pooling with explicit sizes (safe)
        x_h = F.adaptive_avg_pool2d(x, (h, 1))       # (N, C, H, 1)
        x_w = F.adaptive_avg_pool2d(x, (1, w))       # (N, C, 1, W)
        x_w = x_w.permute(0, 1, 3, 2)                # (N, C, W, 1)

        # concat on spatial dim (height axis)
        y = torch.cat([x_h, x_w], dim=2)             # (N, C, H+W, 1)

        y = self.conv1(y)                            # (N, mip, H+W, 1)
        y = self.bn1(y)
        y = self.act(y)

        # split back
        x_h, x_w = torch.split(y, [h, w], dim=2)     # x_h: (N, mip, H, 1), x_w: (N, mip, W, 1)
        x_w = x_w.permute(0, 1, 3, 2)                # (N, mip, 1, W)

        a_h = self.conv_h(x_h).sigmoid()             # (N, oup, H, 1)
        a_w = self.conv_w(x_w).sigmoid()             # (N, oup, 1, W)

        # project identity if needed
        if self.project is not None:
            identity = self.project(identity)        # (N, oup, H, W)

        out = identity * a_h * a_w                   # broadcasting -> (N, oup, H, W)
        return out


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class UP(nn.Module):
    def __init__(self, c1, c2, n: int=2):
        super().__init__()
        self.up = nn.Upsample(None, n, 'nearest')
    def forward(self, x):
        return self.up(x)






class Masked_Downsample(nn.Module):
    def __init__(self, in_channels, out_channels=None, pool_kernel=3, pool_stride=2, pos=False, act=nn.SiLU()):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pos = pos
        self.mask_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, groups=in_channels, bias=True)

        # learnable scale (global)
        self.scale = nn.Parameter(torch.zeros(in_channels,1,1))  # init 0 => no effect initially
        # If you prefer per-channel: self.scale = nn.Parameter(torch.zeros(in_channels,1,1))

        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride, padding=1)
        self.down = Conv(in_channels, out_channels, k=1, s=1)
        # self.attn = SEBlock(out_channels)

    def forward(self, x):
        mask = F.adaptive_avg_pool2d(x, (1, 1))  # shape (B, C, 1, 1)
        mask_out = self.mask_conv(mask)
        scaled = mask_out * self.scale
        if self.pos:
            y = x + scaled
        else:
            y = x - scaled

        y = self.pool(y)
        y = self.down(y)
        return y
        # return self.attn(y)


class SEBlock(nn.Module):
    def __init__(self, c_in, reduction=32):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.Sigmoid()
        )


class AAttn(nn.Module):
    """
    Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """
        Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class GaborConvold(nn.Module):
    """
    Depthwise Gabor: one Gabor kernel per input channel (unique orientation per channel).
    - sigma: single sigma value (float) used for all channels (no multiple sigmas).
    - lambd: wavelength passed to cv2.getGaborKernel.
    - Each input channel gets exactly one Gabor kernel with orientation theta_i = i * pi / c_in.
    - This is equivalent to a depthwise conv with one kernel per input channel.
    - No reduction (max/avg/sum) needed because filters_per_channel == 1.
    """

    def __init__(
            self,
            c_in: int,
            target_c_out: int,
            kernel_size: int = 5,
            sigma: float = 0.1,  # single sigma
            gamma: float = 0.5,
            lambd: float = 10.0,  # Gabor wavelength
            psi: float = 0.0,
            use_bias: bool = False,
            trainable_gabor: bool = False,
    ):
        super().__init__()
        self.c_in = int(c_in)
        self.target_c_out = int(target_c_out)
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.lambd = float(lambd)
        self.psi = float(psi)
        self.use_bias = use_bias
        self.trainable_gabor = bool(trainable_gabor)

        self.groups = self.c_in

        # Depthwise conv: groups = c_in, out_channels = c_in
        self.gabor_dw_conv = nn.Conv2d(
            in_channels=self.c_in,
            out_channels=self.c_in,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.groups,
            bias=self.use_bias
        )

        # Build one Gabor kernel per input channel with distinct orientation
        # Orientation for channel i: theta = i * pi / c_in (evenly spaced in [0, pi))
        gabor_kernels = []
        for i in range(self.c_in):
            theta = np.pi
            k = cv2.getGaborKernel(
                (self.kernel_size, self.kernel_size),
                self.sigma,
                theta,
                self.lambd,
                self.gamma,
                self.psi,
                ktype=cv2.CV_32F
            )
            gabor_kernels.append(k.astype(np.float32))

        # gabor_kernels shape -> (c_in, k, k)
        # For grouped conv with groups=c_in we need weight shape (out_channels, 1, k, k) where out_channels == c_in
        weight_np = np.stack(gabor_kernels, axis=0).reshape(self.c_in, 1, self.kernel_size, self.kernel_size)

        # Assign to conv weight
        weight_tensor = torch.from_numpy(weight_np)
        with torch.no_grad():
            self.gabor_dw_conv.weight.copy_(weight_tensor)
        self.gabor_dw_conv.weight.requires_grad = bool(self.trainable_gabor)

        # Post-conv BN + activation
        # self.bn_after = nn.BatchNorm2d(self.target_c_out)
        # self.act_after = nn.SiLU()
        # self.con = Conv(self.c_in, self.c_in, 1, 1)

        # Optional projection from c_in -> target_c_out
        # if self.target_c_out != self.c_in:
        self.proj = nn.Conv2d(self.c_in, self.c_in, kernel_size=1, stride=1, padding=0, groups=self.c_in, bias=False)
        # self.bn_proj = nn.BatchNorm2d(self.target_c_out)
        self.act_proj = nn.SiLU()
        # else:
        # self.proj = nn.Identity()
        # self.bn_proj = nn.Identity()
        # self.act_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, c_in, H, W)
        returns: (B, target_c_out, H, W)
        """
        # dw = self.con(x)

        # Depthwise conv -> (B, c_in, H, W) because filters_per_channel == 1
        dw = self.gabor_dw_conv(x)

        # Projection to desired output channels
        dw = self.proj(dw)
        # out = self.bn_proj(out)
        dw = self.act_proj(dw)
        return dw


class GaborConvMax(nn.Module):
    """
    Depthwise Gabor bank per input channel -> channel-wise reduction -> optional projection.

    - `scales` are treated as SIGMA values for cv2.getGaborKernel.
    - `lambd` is the wavelength (lambda) passed to cv2.getGaborKernel.
    - For each input channel we build (orientations * num_sigmas) kernels.
    - Uses grouped conv with groups=c_in and out_channels = c_in * filters_per_channel.
    - After conv: reshape (B, c_in, filters_per_channel, H, W) and reduce over filters to (B, c_in, H, W).
    - Optional 1x1 projection to target_c_out.
    - `reduce` controls reduction method: "max" (default), "avg", "sum", or "learned".
    """

    def __init__(
        self,
        c_in: int,
        target_c_out: int,
        kernel_size: int = 5,
        orientations: int = 8,
        scales=(0.5, 1.0),     # SIGMA values
        gamma: float = 0.5,
        lambd: float = 10.0,    # wavelength (lambda) for Gabor
        psi: float = 0.0,
        use_bias: bool = False,
        trainable_gabor: bool = False,
        reduce: str = "max",
        eps_sigma: float = 1e-4,
    ):
        super().__init__()
        assert reduce in ("max", "avg", "sum")

        self.c_in = int(c_in)
        self.target_c_out = int(target_c_out)
        self.kernel_size = int(kernel_size)
        self.orientations = int(orientations)
        # treat 'scales' as sigma values; avoid zeros
        self.sigmas = [float(s) if float(s) > eps_sigma else eps_sigma for s in scales]
        self.num_sigmas = len(self.sigmas)
        self.filters_per_channel = self.orientations * self.num_sigmas

        # wavelength (lambda in Gabor)
        self.lambd = float(lambd)
        self.gamma = float(gamma)
        self.psi = float(psi)

        self.use_bias = use_bias
        self.trainable_gabor = bool(trainable_gabor)
        self.reduce = reduce

        # Depthwise conv parameters
        self.dw_out_channels = self.c_in * self.filters_per_channel
        self.groups = self.c_in

        # Depthwise conv: each group corresponds to one input channel
        self.gabor_dw_conv = nn.Conv2d(
            in_channels=self.c_in,
            out_channels=self.dw_out_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.groups,
            bias=self.use_bias
        )

        # Build Gabor kernels for one input channel (filters_per_channel, k, k)
        theta_values = [i * np.pi / self.orientations for i in range(self.orientations)]
        gabor_kernels = []
        for sigma in self.sigmas:
            for theta in theta_values:
                # cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, psi)
                k = cv2.getGaborKernel(
                    (self.kernel_size, self.kernel_size),
                    sigma,
                    theta,
                    self.lambd,
                    self.gamma,
                    self.psi,
                    ktype=cv2.CV_32F
                )
                gabor_kernels.append(k.astype(np.float32))

        single_channel_kernels = np.stack(gabor_kernels, axis=0)  # (filters_per_channel, k, k)

        # Expand per input channel -> shape (c_in * filters_per_channel, 1, k, k)
        weight_np = np.repeat(single_channel_kernels[np.newaxis, ...], repeats=self.c_in, axis=0)
        weight_np = weight_np.reshape(self.dw_out_channels, 1, self.kernel_size, self.kernel_size)

        # Convert to tensor and assign to conv weight (PyTorch will move it when module.to(device) is called)
        weight_tensor = torch.from_numpy(weight_np)
        with torch.no_grad():
            self.gabor_dw_conv.weight.copy_(weight_tensor)
        self.gabor_dw_conv.weight.requires_grad = bool(self.trainable_gabor)

        # Post-reduction BN + activation
        self.bn_after = nn.BatchNorm2d(self.c_in)
        self.act_after = nn.SiLU()

        # Optional projection from c_in -> target_c_out
        if self.target_c_out != self.c_in:
            self.proj = nn.Conv2d(self.c_in, self.target_c_out, kernel_size=1, bias=self.use_bias)
            self.bn_proj = nn.BatchNorm2d(self.target_c_out)
            self.act_proj = nn.SiLU()
        else:
            self.proj = nn.Identity()
            self.bn_proj = nn.Identity()
            self.act_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, c_in, H, W)
        returns: (B, target_c_out, H, W)
        """
        B, C, H, W = x.shape
        assert C == self.c_in, f"Expected input channels {self.c_in}, got {C}"

        # Depthwise conv -> (B, c_in * filters_per_channel, H, W)
        dw = self.gabor_dw_conv(x)

        # Reshape -> (B, c_in, filters_per_channel, H, W)
        dw_view = dw.view(B, self.c_in, self.filters_per_channel, H, W)

        # Reduce across filters per-channel
        if self.reduce == "max":
            per_channel, _ = torch.max(dw_view, dim=2)   # (B, c_in, H, W)
        elif self.reduce == "avg":
            per_channel = dw_view.mean(dim=2)
        elif self.reduce == "sum":
            per_channel = dw_view.sum(dim=2)

        # BN + activation
        out = self.bn_after(per_channel)
        out = self.act_after(out)

        # Projection (if any)
        out = self.proj(out)
        out = self.bn_proj(out)
        out = self.act_proj(out)

        return out


class GaborConv(nn.Module):
    def __init__(
            self,
            c_in: int,
            target_c_out: int,
            kernel_size: int = 5,
            sigma: float = 0.1,  # single sigma
            gamma: float = 0.5,
            lambd: float = 10.0,  # Gabor wavelength
            psi: float = 0.0,
            use_bias: bool = False,
            trainable_gabor: bool = False,
    ):
        super().__init__()
        self.c_in = int(c_in)
        self.target_c_out = int(target_c_out)
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.lambd = float(lambd)
        self.psi = float(psi)
        self.use_bias = use_bias
        self.trainable_gabor = bool(trainable_gabor)

        self.groups = self.c_in

        # Depthwise conv: groups = c_in, out_channels = c_in
        self.gabor_dw_conv = nn.Conv2d(
            in_channels=self.c_in,
            out_channels=self.c_in,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.groups,
            bias=self.use_bias
        )

        gabor_kernels = []
        for i in range(self.c_in):
            theta = np.pi
            k = cv2.getGaborKernel(
                (self.kernel_size, self.kernel_size),
                self.sigma,
                theta,
                self.lambd,
                self.gamma,
                self.psi,
                ktype=cv2.CV_32F
            )
            gabor_kernels.append(k.astype(np.float32))

        weight_np = np.stack(gabor_kernels, axis=0).reshape(self.c_in, 1, self.kernel_size, self.kernel_size)

        # Assign to conv weight
        weight_tensor = torch.from_numpy(weight_np)
        with torch.no_grad():
            self.gabor_dw_conv.weight.copy_(weight_tensor)
        self.gabor_dw_conv.weight.requires_grad = bool(self.trainable_gabor)

        self.proj = nn.Conv2d(self.c_in, self.c_in, kernel_size=1, stride=1, padding=0, groups=self.c_in, bias=False)
        self.act_proj = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dw = self.gabor_dw_conv(x)
        dw = self.proj(dw)
        dw = self.act_proj(dw)
        return dw


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d( in_tensor, in_tensor.size(2), stride=in_tensor.size(2) )
        return self.gate_c( avg_pool ).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)

class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module( 'gate_s_bn_reduce0',	nn.BatchNorm2d(gate_channel//reduction_ratio) )
        self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
						padding=dilation_val, dilation=dilation_val) )
            self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
            self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
    def forward(self, in_tensor):
        return self.gate_s( in_tensor ).expand_as(in_tensor)
class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)
    def forward(self,x):
        att = 1 + F.sigmoid( self.channel_att(x) * self.spatial_att(x) )
        return att * x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu6(x + 3) / 6.0

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.act = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.act(x)

class CoordAtt(nn.Module):
    """
    Robust CoordAtt:
      - inp: input channels
      - oup: output channels. If oup != inp, a 1x1 conv projects identity before multiplication.
      - reduction: bottleneck factor
    """
    def __init__(self, inp, oup=None, reduction=32):
        super().__init__()
        if oup is None:
            oup = inp
        self.inp = inp
        self.oup = oup

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        # convs producing attention maps for height and width
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=True)

        # if oup != inp, project identity to oup channels before multiplication
        if oup != inp:
            self.project = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.project = None

    def forward(self, x):
        # x: (N, C, H, W)
        identity = x
        n, c, h, w = x.size()

        # adaptive pooling with explicit sizes (safe)
        x_h = F.adaptive_avg_pool2d(x, (h, 1))       # (N, C, H, 1)
        x_w = F.adaptive_avg_pool2d(x, (1, w))       # (N, C, 1, W)
        x_w = x_w.permute(0, 1, 3, 2)                # (N, C, W, 1)

        # concat on spatial dim (height axis)
        y = torch.cat([x_h, x_w], dim=2)             # (N, C, H+W, 1)

        y = self.conv1(y)                            # (N, mip, H+W, 1)
        y = self.bn1(y)
        y = self.act(y)

        # split back
        x_h, x_w = torch.split(y, [h, w], dim=2)     # x_h: (N, mip, H, 1), x_w: (N, mip, W, 1)
        x_w = x_w.permute(0, 1, 3, 2)                # (N, mip, 1, W)

        a_h = self.conv_h(x_h).sigmoid()             # (N, oup, H, 1)
        a_w = self.conv_w(x_w).sigmoid()             # (N, oup, 1, W)

        # project identity if needed
        if self.project is not None:
            identity = self.project(identity)        # (N, oup, H, W)

        out = identity * a_h * a_w                   # broadcasting -> (N, oup, H, W)
        return out


class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU() # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Apply convolution transpose and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, H, W) and output shape is (B, c2, H/2, W/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=32, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class Concat(nn.Module):
    """Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """Select and return a particular index from input.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]
