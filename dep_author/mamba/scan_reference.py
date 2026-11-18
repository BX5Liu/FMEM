# Copyright (c) 2023, Tri Dao, Albert Gu. Apache-2.0.
import torch
import torch.nn.functional as F
from einops import rearrange
from vendor.selective_scan_ref import selective_scan_ref as selective_scan_fn


def mamba_inner_fn_no_out_proj(xz, conv1d_weight, conv1d_bias, x_proj_weight,
        delta_proj_weight, A, B=None, C=None, D=None, delta_bias=None,
        B_proj_bias=None, C_proj_bias=None, delta_softplus=True):
    length = xz.shape[-1]
    rank = delta_proj_weight.shape[1]
    states = A.shape[-1] * (2 if A.is_complex() else 1)
    x, z = xz.chunk(2, dim=1)
    x = F.silu(F.conv1d(x, conv1d_weight, conv1d_bias,
                        padding=conv1d_weight.shape[-1] - 1, groups=x.shape[1])[..., :length])
    projected = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)
    delta = rearrange(delta_proj_weight @ projected[:, :rank].t(), 'd (b l) -> b d l', l=length)
    if B is None:
        B = projected[:, rank:rank + states]
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(B.dtype)
        B = rearrange(B, '(b l) dstate -> b dstate l', l=length).contiguous()
    if C is None:
        C = projected[:, -states:]
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(C.dtype)
        C = rearrange(C, '(b l) dstate -> b dstate l', l=length).contiguous()
    if A.is_complex():
        raise NotImplementedError('This experiment uses real-valued Mamba states')
    return selective_scan_fn(x, delta, A, B, C, D, z=z, delta_bias=delta_bias,
                             delta_softplus=delta_softplus)


def mamba_inner_fn(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
        out_proj_weight, out_proj_bias, A, B=None, C=None, D=None, delta_bias=None,
        B_proj_bias=None, C_proj_bias=None, delta_softplus=True):
    result = mamba_inner_fn_no_out_proj(xz, conv1d_weight, conv1d_bias, x_proj_weight,
        delta_proj_weight, A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus)
    return F.linear(result.transpose(1, 2), out_proj_weight, out_proj_bias)


def bimamba_inner_fn(*args, **kwargs):
    raise NotImplementedError('The author configuration uses bidirectional v2')
