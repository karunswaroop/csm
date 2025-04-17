"""
Dummy bitsandbytes module to satisfy moshi quantization on non-NVIDIA platforms.
Implements MatmulLtState and matmul fallback to torch.nn.functional.linear.
"""
import torch
import torch.nn.functional as F

__all__ = ["MatmulLtState", "matmul"]

class MatmulLtState:
    """
    Dummy state for linear quantized matmul.
    """
    def __init__(self):
        self.CB = None
        self.SCB = None
        self.has_fp16_weights = False

def matmul(x: torch.Tensor, CB: torch.Tensor, state=None) -> torch.Tensor:
    """
    Fallback matmul for quantized weights: use a standard linear.
    """
    # x: (*, in_features), CB: (out_features, in_features)
    return F.linear(x, CB)