"""
INT8 IPEX Quantization Module
Intel Arc GPU optimization using Intel Extension for PyTorch
"""

import torch
import time
import numpy as np
from tqdm import tqdm

# Check IPEX availability
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False


def check_ipex():
    """Check if IPEX is available"""
    if not IPEX_AVAILABLE:
        raise ImportError(
            "Intel Extension for PyTorch (IPEX) not found!\n"
            "Install with: pip install intel-extension-for-pytorch\n"
            "This is required for Intel Arc GPU optimization."
        )
    return True


def quantize_ipex(model, calib_loader, device):
    """
    Apply INT8 quantization using IPEX
    Optimized for Intel Arc GPUs (XPU)
    
    Args:
        model: PyTorch model
        calib_loader: Calibration data loader
        device: torch.device (should be 'xpu' or 'cuda' or 'cpu')
    
    Returns:
        quantized_model: IPEX quantized model
    """
    check_ipex()
    
    print("\nApplying IPEX INT8 Quantization...")
    print(f"Target device: {device}")
    print("Mode: Static INT8 with IPEX optimization")
    
    model = model.to(device)
    model.eval()
    
    # Configure IPEX quantization
    qconfig = ipex.quantization.default_static_qconfig
    
    # Prepare example inputs for tracing
    print("\nPreparing calibration data...")
    example_inputs = None
    for images, _ in calib_loader:
        example_inputs = [img.to(device) for img in images]
        break
    
    if example_inputs is None:
        raise RuntimeError("Failed to get example inputs from calibration data")
    
    # Prepare model for quantization
    print("Preparing model with IPEX quantization config...")
    prepared_model = ipex.quantization.prepare(
        model,
        qconfig,
        example_inputs=example_inputs,
        inplace=False
    )
    
    # Calibration phase
    print("\nCalibrating model (running inference on calibration data)...")
    with torch.no_grad():
        for idx, (images, targets) in enumerate(tqdm(calib_loader, desc="Calibration")):
            if idx >= len(calib_loader):
                break
            
            images_dev = [img.to(device) for img in images]
            
            try:
                _ = prepared_model(images_dev)
            except Exception as e:
                # Some batches may fail, skip them
                print(f"\nWarning: Calibration batch {idx} failed: {e}")
                continue
    
    # Convert to quantized model
    print("\nConverting to quantized model...")
    quantized_model = ipex.quantization.convert(prepared_model)
    
    # Optimize with IPEX
    print("Applying IPEX optimizations...")
    quantized_model = ipex.optimize(
        quantized_model,
        dtype=torch.int8,
        inplace=True
    )
    
    print("✓ IPEX quantization complete")
    return quantized_model


def evaluate_ipex_model(model, val_loader, device, original_metrics):
    """
    Evaluate IPEX quantized model
    
    Args:
        model: IPEX quantized model
        val_loader: Validation data loader
        device: torch.device
        original_metrics: Original model metrics for comparison
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    from utils.evaluation import evaluate_detection
    
    print("\nEvaluating IPEX INT8 model...")
    model.eval()
    
    predictions_all = []
    targets_all = []
    inference_times = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating IPEX INT8"):
            images_dev = [img.to(device) for img in images]
            
            # Synchronize for accurate timing
            if device.type == 'xpu':
                torch.xpu.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            preds = model(images_dev)
            
            if device.type == 'xpu':
                torch.xpu.synchronize()
            elif device.type == 'cuda':
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


def get_ipex_version():
    """Get IPEX version if available"""
    if IPEX_AVAILABLE:
        return ipex.__version__
    return None
