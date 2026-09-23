"""Hand-written ops on torch primitives. No torch.nn, no torch.nn.functional model ops."""
import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm: scale each token's vector to unit root-mean-square, then per-channel weight.

    x:      [..., D]   the residual stream (one row per token)
    weight: [D]        learned per-channel volume knobs
    """
    x32 = x.float()                                     # do the math in fp32 for accuracy
    rms = torch.sqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + eps)
    return (x32 / rms) * weight.float()                 # broadcast weight across tokens
