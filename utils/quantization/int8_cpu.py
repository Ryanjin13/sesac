"""
OpenVINO INT8/FP16 Quantization Module
Converts PyTorch → ONNX → OpenVINO with INT8 quantization
Works on Intel CPU/GPU/Arc
"""

import torch
import numpy as np
import os
import time
from tqdm import tqdm
from pathlib import Path

# Check OpenVINO availability
try:
    import openvino as ov
    from openvino.tools import mo
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
    OV_VERSION = ov.__version__
except ImportError:
    OPENVINO_AVAILABLE = False
    OV_VERSION = None


def check_openvino():
    """Check if OpenVINO is available"""
    if not OPENVINO_AVAILABLE:
        raise ImportError(
            "OpenVINO not found!\n"
            "Install with: pip install openvino openvino-dev\n"
            "Required for Intel CPU/GPU/Arc optimization."
        )
    print(f"OpenVINO version: {OV_VERSION}")
    return True


def export_to_onnx(model, onnx_path, input_size=(640, 640)):
    """
    Export PyTorch Faster R-CNN to ONNX
    
    Args:
        model: PyTorch model
        onnx_path: Output ONNX file path
        input_size: Input image size
    
    Returns:
        success: True if export succeeded
    """
    print("\n" + "="*70)
    print("STEP 1: PyTorch → ONNX Export")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, *input_size).to(device)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Exporting to: {onnx_path}")
    print("Note: Faster R-CNN export may produce warnings (normal)")
    
    try:
        # Export with opset 11
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['boxes', 'labels', 'scores'],
            dynamic_axes={
                'input': {0: 'batch_size'},
            },
            verbose=False
        )
        
        print("✓ ONNX export successful")
        
        # Check file size
        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  ONNX file size: {onnx_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ONNX export failed: {e}")
        print("\nThis may happen with Faster R-CNN due to dynamic operations.")
        return False


def convert_to_openvino(onnx_path, output_dir, precision='FP16'):
    """
    Convert ONNX to OpenVINO IR format
    
    Args:
        onnx_path: Input ONNX file
        output_dir: Output directory for OpenVINO IR
        precision: 'FP32', 'FP16', or 'FP16' (INT8 done separately)
    
    Returns:
        ir_path: Path to .xml file
    """
    print("\n" + "="*70)
    print(f"STEP 2: ONNX → OpenVINO IR ({precision})")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Convert using Model Optimizer
        print("Converting with OpenVINO Model Optimizer...")
        
        # Use openvino.tools.mo (modern API)
        model = mo.convert_model(
            onnx_path,
            compress_to_fp16=(precision == 'FP16')
        )
        
        # Save the model
        ir_path = os.path.join(output_dir, 'model.xml')
        ov.save_model(model, ir_path)
        
        print(f"✓ OpenVINO IR saved to: {output_dir}")
        
        # Check file sizes
        xml_size = os.path.getsize(ir_path) / (1024 * 1024)
        bin_path = ir_path.replace('.xml', '.bin')
        bin_size = os.path.getsize(bin_path) / (1024 * 1024)
        
        print(f"  model.xml: {xml_size:.2f} MB")
        print(f"  model.bin: {bin_size:.2f} MB")
        print(f"  Total: {xml_size + bin_size:.2f} MB")
        
        return ir_path
        
    except Exception as e:
        print(f"\n✗ OpenVINO conversion failed: {e}")
        return None


def quantize_int8_openvino(ir_path, output_dir, calibration_loader):
    """
    Apply INT8 quantization using OpenVINO POT
    
    Args:
        ir_path: Path to FP16/FP32 OpenVINO IR
        output_dir: Output directory for INT8 model
        calibration_loader: DataLoader for calibration
    
    Returns:
        quantized_ir_path: Path to quantized model
    """
    print("\n" + "="*70)
    print("STEP 3: INT8 Quantization (Post-Training Optimization)")
    print("="*70)
    
    print("Note: Using NNCF (Neural Network Compression Framework)")
    
    try:
        from nncf import quantize, Dataset
        
        # Load model
        core = Core()
        model = core.read_model(ir_path)
        
        print("Preparing calibration dataset...")
        
        def transform_fn(data_item):
            """Transform calibration data"""
            images, _ = data_item
            # Convert to numpy and ensure correct shape
            if isinstance(images, (list, tuple)):
                images = images[0]
            if isinstance(images, torch.Tensor):
                images = images.cpu().numpy()
            return images
        
        # Create calibration dataset
        calibration_dataset = Dataset(calibration_loader, transform_fn)
        
        print(f"Calibrating with {len(calibration_loader)} batches...")
        
        # Quantize model
        quantized_model = quantize(
            model,
            calibration_dataset,
            preset='mixed',  # 'performance' or 'mixed' or 'accuracy'
            subset_size=len(calibration_loader)
        )
        
        # Save quantized model
        quantized_ir_path = os.path.join(output_dir, 'model.xml')
        ov.save_model(quantized_model, quantized_ir_path)
        
        print(f"✓ INT8 model saved to: {output_dir}")
        
        # Check file sizes
        xml_size = os.path.getsize(quantized_ir_path) / (1024 * 1024)
        bin_path = quantized_ir_path.replace('.xml', '.bin')
        bin_size = os.path.getsize(bin_path) / (1024 * 1024)
        
        print(f"  model.xml: {xml_size:.2f} MB")
        print(f"  model.bin: {bin_size:.2f} MB")
        print(f"  Total: {xml_size + bin_size:.2f} MB")
        
        return quantized_ir_path
        
    except ImportError:
        print("\n✗ NNCF not installed. Install with: pip install nncf")
        print("Falling back to FP16 model (no INT8 quantization)")
        return ir_path
    except Exception as e:
        print(f"\n✗ INT8 quantization failed: {e}")
        print("This may happen if the model structure is incompatible.")
        return ir_path


def evaluate_openvino_model(ir_path, val_loader, device_name='CPU', original_metrics=None):
    """
    Evaluate OpenVINO model
    
    Args:
        ir_path: Path to OpenVINO IR (.xml)
        val_loader: Validation data loader
        device_name: 'CPU', 'GPU', or 'AUTO'
        original_metrics: Original model metrics for comparison
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    from utils.evaluation import evaluate_detection
    
    print(f"\nEvaluating OpenVINO model on {device_name}...")
    
    # Load model
    core = Core()
    model = core.read_model(ir_path)
    compiled_model = core.compile_model(model, device_name)
    
    # Get input/output info
    input_layer = compiled_model.input(0)
    output_layers = compiled_model.outputs
    
    print(f"Input shape: {input_layer.shape}")
    print(f"Device: {device_name}")
    
    predictions_all = []
    targets_all = []
    inference_times = []
    
    for images, targets in tqdm(val_loader, desc=f"Evaluating on {device_name}"):
        # Prepare input
        if isinstance(images, (list, tuple)):
            # Take first image from batch
            image_tensor = images[0]
        else:
            image_tensor = images
        
        if isinstance(image_tensor, torch.Tensor):
            image_np = image_tensor.cpu().numpy()
        else:
            image_np = image_tensor
        
        # Ensure correct shape [1, 3, H, W]
        if image_np.ndim == 3:
            image_np = np.expand_dims(image_np, 0)
        
        start = time.time()
        
        # Run inference
        results = compiled_model([image_np])
        
        inference_times.append(time.time() - start)
        
        # Parse outputs
        # Note: Output format depends on model export
        # This is a simplified version, may need adjustment
        try:
            boxes = results[0] if len(results) > 0 else np.array([])
            labels = results[1] if len(results) > 1 else np.array([])
            scores = results[2] if len(results) > 2 else np.array([])
            
            pred = {
                'boxes': torch.from_numpy(boxes) if isinstance(boxes, np.ndarray) else boxes,
                'labels': torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels,
                'scores': torch.from_numpy(scores) if isinstance(scores, np.ndarray) else scores
            }
            
            predictions_all.append(pred)
            targets_all.extend(targets if isinstance(targets, list) else [targets])
            
        except Exception as e:
            print(f"\nWarning: Failed to parse prediction: {e}")
            continue
    
    # Calculate metrics
    try:
        metrics = evaluate_detection(predictions_all, targets_all, score_threshold=0.5)
    except Exception as e:
        print(f"\nWarning: Evaluation failed: {e}")
        print("This may happen if output format is incompatible.")
        metrics = {
            'avg_iou': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
    
    avg_time = np.mean(inference_times) if inference_times else 0.0
    metrics['avg_inference_time'] = avg_time
    metrics['fps'] = 1.0 / avg_time if avg_time > 0 else 0.0
    
    # Print comparison
    if original_metrics:
        print(f"\nmAP@0.5: {metrics['avg_iou']:.4f} "
              f"(Δ {metrics['avg_iou'] - original_metrics['avg_iou']:.4f})")
        print(f"Precision: {metrics['precision']:.4f} "
              f"(Δ {metrics['precision'] - original_metrics['precision']:.4f})")
        print(f"Recall: {metrics['recall']:.4f} "
              f"(Δ {metrics['recall'] - original_metrics['recall']:.4f})")
    else:
        print(f"\nmAP@0.5: {metrics['avg_iou']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
    
    print(f"Inference time: {metrics['avg_inference_time']*1000:.2f} ms/image")
    print(f"FPS: {metrics['fps']:.2f}")
    
    return metrics


def get_available_devices():
    """Get list of available OpenVINO devices"""
    if not OPENVINO_AVAILABLE:
        return []
    
    core = Core()
    devices = core.available_devices
    return devices