"""
Quantization Module for Faster R-CNN
"""

from .int8_ipex import quantize_ipex, evaluate_ipex_model, check_ipex, get_ipex_version


__all__ = [
    'quantize_to_fp16',
    'evaluate_fp16_model',
]