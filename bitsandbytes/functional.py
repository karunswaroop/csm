"""
Dummy bitsandbytes.functional module to satisfy moshi quantization on non-NVIDIA platforms.
Provides int8_vectorwise_quant fallback (identity quantization).
"""
import torch

__all__ = ["int8_vectorwise_quant"]

def int8_vectorwise_quant(w: torch.Tensor):
    """
    Identity fallback for int8 vectorwise quantization.
    Returns:
        CB: torch.Tensor -- same as w but cast to float32
        SCB: torch.Tensor -- scaling factors (all ones)
        None
    """
    # w is expected as float16
    CB = w.to(torch.float32)
    # SCB: one scale per output channel
    out_channels = CB.shape[0]
    SCB = torch.ones(out_channels, dtype=torch.float32)
    return CB, SCB, None