"""
FP16 Quantization Module
Universal quantization compatible with NVIDIA CUDA and Intel Arc XPU
"""

import torch
import time
import numpy as np
from tqdm import tqdm


def quantize_to_fp16(model):
    """
    Convert model to Mixed Precision FP16
    - Backbone: FP16 (convolutions are stable)
    - RPN + ROI Heads: FP32 (keep precision for detection)
    
    Args:
        model: PyTorch model
    
    Returns:
        mixed_fp16_model: Model with selective FP16 conversion
    """
    print("\nApplying Mixed Precision FP16 quantization...")
    print("Strategy:")
    print("  - Backbone → FP16 (safe for convolutions)")
    print("  - RPN → FP32 (anchor generation needs precision)")
    print("  - ROI Heads → FP32 (detection needs precision)")
    
    # Convert backbone to FP16
    if hasattr(model, 'backbone'):
        print("\n  Converting backbone to FP16...")
        model.backbone = model.backbone.half()
    
    # Keep RPN in FP32
    if hasattr(model, 'rpn'):
        print("  Keeping RPN in FP32...")
        model.rpn = model.rpn.float()
    
    # Keep ROI heads in FP32
    if hasattr(model, 'roi_heads'):
        print("  Keeping ROI heads in FP32...")
        model.roi_heads = model.roi_heads.float()
    
    # Keep transform in FP32 (important for preprocessing)
    if hasattr(model, 'transform'):
        print("  Keeping transform in FP32...")
        model.transform = model.transform.float()
    
    print("✓ Mixed Precision FP16 conversion complete")
    print("  Expected: ~70-80MB model size, ~1.5x speedup, minimal accuracy loss")
    
    return model


def evaluate_fp16_model(model, val_loader, device, original_metrics):
    """
    Evaluate Mixed Precision FP16 quantized model
    
    Args:
        model: Mixed FP16 quantized model
        val_loader: Validation data loader
        device: torch.device
        original_metrics: Original model metrics for comparison
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    from utils.evaluation import evaluate_detection
    
    print("\nEvaluating Mixed Precision FP16 model...")
    model.eval()
    
    predictions_all = []
    targets_all = []
    inference_times = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating Mixed FP16"):
            # Convert images to FP16 for backbone, model will handle internal conversions
            images_gpu = [img.to(device).half() for img in images]
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            
            preds = model(images_gpu)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_times.append(time.time() - start)
            
            predictions_all.extend([{k: v.cpu() for k, v in p.items()} for p in preds])
            targets_all.extend(targets)
    
    # Calculate metrics
    metrics = evaluate_detection(predictions_all, targets_all, score_threshold=0.5)
    
    avg_time = np.mean(inference_times)
    metrics['avg_inference_time'] = avg_time
    metrics['fps'] = 1.0 / avg_time
    
    # Print comparison
    print(f"\nmAP@0.5: {metrics['avg_iou']:.4f} "
          f"(Δ {metrics['avg_iou'] - original_metrics['avg_iou']:.4f})")
    print(f"Precision: {metrics['precision']:.4f} "
          f"(Δ {metrics['precision'] - original_metrics['precision']:.4f})")
    print(f"Recall: {metrics['recall']:.4f} "
          f"(Δ {metrics['recall'] - original_metrics['recall']:.4f})")
    print(f"Inference time: {metrics['avg_inference_time']*1000:.2f} ms/batch "
          f"(Δ {(metrics['avg_inference_time'] - original_metrics['avg_inference_time'])*1000:.2f} ms)")
    
    speedup = original_metrics['avg_inference_time'] / metrics['avg_inference_time']
    print(f"Speedup: {speedup:.2f}x")
    print(f"FPS: {metrics['fps']:.2f}")
    
    return metrics