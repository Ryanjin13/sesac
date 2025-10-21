"""
Unified Visualization Module for Deep Learning Curriculum
Supports both Classification (ANN/DNN/CNN) and Detection (Faster R-CNN) tasks
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from typing import Dict, List, Optional
import os
from PIL import Image
import random


# ============================================================================
# CLASSIFICATION VISUALIZATION (ANN, DNN, CNN)
# ============================================================================

def plot_classification_history(history: Dict, 
                                title: str = "Training History",
                                save_path: Optional[str] = None):
    """
    Plot training history for classification tasks
    Automatically adapts to available data (train/val/test)
    
    Args:
        history: Dictionary with keys:
            - 'train_loss': list of train losses (required)
            - 'train_acc': list of train accuracies (optional)
            - 'val_loss': list of val losses (optional)
            - 'val_acc': list of val accuracies (optional)
            - 'test_loss': list of test losses (optional)
            - 'test_acc': list of test accuracies (optional)
        title: Plot title
        save_path: If provided, saves figure to this path
    
    Example:
        # Early ANN (no validation)
        history = {
            'train_loss': [0.5, 0.3, 0.2],
            'train_acc': [85, 90, 93],
            'test_acc': [83, 87, 89]
        }
        
        # Later CNN (with validation)
        history = {
            'train_loss': [0.5, 0.3, 0.2],
            'val_loss': [0.6, 0.4, 0.3],
            'train_acc': [85, 90, 93],
            'val_acc': [83, 88, 90],
            'test_acc': [82, 86, 89]
        }
    """
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Determine subplot layout
    has_accuracy = any(key in history for key in ['train_acc', 'val_acc', 'test_acc'])
    n_plots = 2 if has_accuracy else 1
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    # ========== Loss Plot ==========
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2, 
                marker='o', markersize=5, label='Train Loss')
    
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r--', linewidth=2,
                    marker='s', markersize=5, label='Val Loss')
    
    if 'test_loss' in history:
        axes[0].plot(epochs, history['test_loss'], 'g:', linewidth=2,
                    marker='^', markersize=5, label='Test Loss')
    
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Loss Curves', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Mark potential overfitting
    if 'val_loss' in history and len(history['val_loss']) > 1:
        for i in range(1, len(history['val_loss'])):
            if (history['val_loss'][i] > history['val_loss'][i-1] and 
                history['train_loss'][i] < history['train_loss'][i-1]):
                axes[0].axvline(x=i+1, color='orange', linestyle=':', alpha=0.5)
                axes[0].text(i+1, axes[0].get_ylim()[1]*0.9, 
                           'Potential\nOverfitting', 
                           ha='center', fontsize=7, color='orange')
                break
    
    # ========== Accuracy Plot ==========
    if has_accuracy:
        if 'train_acc' in history:
            axes[1].plot(epochs, history['train_acc'], 'b-', linewidth=2,
                        marker='o', markersize=5, label='Train Acc')
        
        if 'val_acc' in history:
            axes[1].plot(epochs, history['val_acc'], 'r--', linewidth=2,
                        marker='s', markersize=5, label='Val Acc')
        
        if 'test_acc' in history:
            axes[1].plot(epochs, history['test_acc'], 'g:', linewidth=2,
                        marker='^', markersize=5, label='Test Acc')
        
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Accuracy (%)', fontsize=11)
        axes[1].set_title('Accuracy Curves', fontsize=12, fontweight='bold')
        axes[1].legend(loc='best', fontsize=9)
        axes[1].grid(True, alpha=0.3)
        
        # Annotate best performance
        if 'val_acc' in history:
            best_idx = np.argmax(history['val_acc'])
            best_val = history['val_acc'][best_idx]
            axes[1].annotate(f'Best: {best_val:.2f}%',
                           xy=(best_idx+1, best_val),
                           xytext=(10, -15), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.4', fc='lightgreen', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                           fontsize=8)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    # Print summary
    _print_classification_summary(history)


def _print_classification_summary(history: Dict):
    """Print training summary statistics"""
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    
    # Check overfitting
    if 'val_loss' in history and len(history['train_loss']) > 0:
        train_improvement = history['train_loss'][0] - history['train_loss'][-1]
        val_change = history['val_loss'][-1] - history['val_loss'][0]
        
        if val_change > 0 and train_improvement > 0:
            print("⚠️  Potential overfitting detected:")
            print(f"   Train loss improved by {train_improvement:.4f}")
            print(f"   Val loss worsened by {val_change:.4f}")
        else:
            print("✓ No clear overfitting")
    
    # Final metrics
    print(f"\nFinal Epoch Metrics:")
    print(f"  Train Loss: {history['train_loss'][-1]:.4f}", end="")
    if 'train_acc' in history:
        print(f" | Train Acc: {history['train_acc'][-1]:.2f}%")
    else:
        print()
    
    if 'val_loss' in history:
        print(f"  Val Loss:   {history['val_loss'][-1]:.4f}", end="")
        if 'val_acc' in history:
            print(f" | Val Acc:   {history['val_acc'][-1]:.2f}%")
        else:
            print()
    
    if 'test_acc' in history:
        print(f"  Test Acc:   {history['test_acc'][-1]:.2f}%")
    
    # Best performance
    if 'val_acc' in history:
        best_epoch = np.argmax(history['val_acc']) + 1
        best_acc = max(history['val_acc'])
        print(f"\nBest Val Acc: {best_acc:.2f}% (Epoch {best_epoch})")
    
    print("="*60 + "\n")


# ============================================================================
# OBJECT DETECTION VISUALIZATION (Faster R-CNN)
# ============================================================================

def plot_detection_history(history: Dict,
                          title: str = "Detection Training History",
                          save_path: Optional[str] = None):
    """
    Plot training history for object detection tasks
    
    Args:
        history: Dictionary with keys:
            - 'train_loss': list of train losses (required)
            - 'val_loss': list of val losses (optional)
            - 'train_iou': list of train IoU values (required)
            - 'val_iou': list of val IoU values (required)
            - 'train_precision': list (optional)
            - 'val_precision': list (optional)
            - 'train_recall': list (optional)
            - 'val_recall': list (optional)
    
    Example:
        history = {
            'train_loss': [2.5, 1.8, 1.2],
            'val_loss': [2.7, 2.0, 1.5],
            'train_iou': [0.3, 0.45, 0.55],
            'val_iou': [0.28, 0.42, 0.52],
            'val_precision': [0.7, 0.75, 0.8],
            'val_recall': [0.65, 0.70, 0.75]
        }
    """
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Determine number of plots
    has_pr = ('val_precision' in history and 'val_recall' in history)
    n_plots = 3 if has_pr else 2
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    
    # ========== Loss Plot ==========
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2,
                marker='o', markersize=5, label='Train Loss')
    
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r--', linewidth=2,
                    marker='s', markersize=5, label='Val Loss')
    
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Loss Curves', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Mark overfitting
    if 'val_loss' in history and len(history['val_loss']) > 1:
        for i in range(1, len(history['val_loss'])):
            if (history['val_loss'][i] > history['val_loss'][i-1] and 
                history['train_loss'][i] < history['train_loss'][i-1]):
                axes[0].axvline(x=i+1, color='orange', linestyle=':', alpha=0.5)
                break
    
    # ========== IoU Plot ==========
    axes[1].plot(epochs, history['train_iou'], 'b-', linewidth=2,
                marker='o', markersize=5, label='Train IoU')
    axes[1].plot(epochs, history['val_iou'], 'g--', linewidth=2,
                marker='s', markersize=5, label='Val IoU')
    
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('IoU', fontsize=11)
    axes[1].set_title('IoU Progression', fontsize=12, fontweight='bold')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # Annotate best IoU
    best_iou_idx = np.argmax(history['val_iou'])
    best_iou = history['val_iou'][best_iou_idx]
    axes[1].annotate(f'Best: {best_iou:.4f}',
                    xy=(best_iou_idx+1, best_iou),
                    xytext=(10, -15), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.4', fc='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                    fontsize=8)
    
    # ========== Precision/Recall Plot ==========
    if has_pr:
        axes[2].plot(epochs, history['val_precision'], 'b-', linewidth=2,
                    marker='o', markersize=5, label='Precision')
        axes[2].plot(epochs, history['val_recall'], 'r-', linewidth=2,
                    marker='s', markersize=5, label='Recall')
        
        # F1 score if both available
        f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 
                     for p, r in zip(history['val_precision'], history['val_recall'])]
        axes[2].plot(epochs, f1_scores, 'g--', linewidth=2,
                    marker='^', markersize=5, label='F1 Score')
        
        axes[2].set_xlabel('Epoch', fontsize=11)
        axes[2].set_ylabel('Score', fontsize=11)
        axes[2].set_title('Precision / Recall / F1', fontsize=12, fontweight='bold')
        axes[2].legend(loc='best', fontsize=9)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1.05])
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    # Print summary
    _print_detection_summary(history)


def _print_detection_summary(history: Dict):
    """Print detection training summary"""
    print("\n" + "="*60)
    print("Detection Training Summary")
    print("="*60)
    
    # Overfitting check
    if 'val_loss' in history and len(history['val_loss']) > 1:
        train_improvement = history['train_loss'][0] - history['train_loss'][-1]
        val_change = history['val_loss'][-1] - history['val_loss'][0]
        
        if val_change > 0 and train_improvement > 0:
            print("⚠️  Potential overfitting:")
            print(f"   Train loss: {history['train_loss'][0]:.4f} → {history['train_loss'][-1]:.4f}")
            print(f"   Val loss: {history['val_loss'][0]:.4f} → {history['val_loss'][-1]:.4f}")
        else:
            print("✓ No clear overfitting detected")
    
    # Final metrics
    print(f"\nFinal Epoch:")
    print(f"  Train Loss: {history['train_loss'][-1]:.4f} | Train IoU: {history['train_iou'][-1]:.4f}")
    print(f"  Val Loss:   {history['val_loss'][-1]:.4f} | Val IoU:   {history['val_iou'][-1]:.4f}")
    
    if 'val_precision' in history:
        print(f"  Precision:  {history['val_precision'][-1]:.4f} | Recall:    {history['val_recall'][-1]:.4f}")
    
    # Best performance
    best_epoch = np.argmax(history['val_iou']) + 1
    best_iou = max(history['val_iou'])
    print(f"\nBest Val IoU: {best_iou:.4f} (Epoch {best_epoch})")
    
    print("="*60 + "\n")


# ============================================================================
# MODEL COMPARISON VISUALIZATION
# ============================================================================

def plot_metrics_comparison(models_data: Dict[str, Dict],
                           metric: str = 'accuracy',
                           title: Optional[str] = None,
                           save_path: Optional[str] = None):
    """
    Compare multiple models' performance
    
    Args:
        models_data: Dictionary of {model_name: {'train': [...], 'val': [...], 'test': value}}
        metric: 'accuracy', 'loss', or 'iou'
        title: Plot title
        save_path: Save path
    
    Example:
        models = {
            'ANN': {
                'train': [85, 88, 90],
                'test': 89
            },
            'DNN': {
                'train': [87, 91, 93],
                'val': [86, 89, 91],
                'test': 92
            },
            'CNN': {
                'train': [90, 94, 96],
                'val': [89, 92, 94],
                'test': 95
            }
        }
        plot_metrics_comparison(models, 'accuracy', 'Model Comparison')
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))
    
    # ========== Left: Training curves ==========
    for (model_name, data), color in zip(models_data.items(), colors):
        if 'train' in data:
            epochs = range(1, len(data['train']) + 1)
            ax1.plot(epochs, data['train'], linewidth=2, marker='o', 
                    markersize=5, label=f'{model_name} (train)', color=color)
            
            if 'val' in data:
                ax1.plot(epochs, data['val'], linewidth=2, marker='s', 
                        markersize=5, label=f'{model_name} (val)', 
                        color=color, linestyle='--')
    
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel(metric.capitalize(), fontsize=11)
    ax1.set_title(f'{metric.capitalize()} Progression', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # ========== Right: Final test comparison ==========
    model_names = list(models_data.keys())
    test_scores = [data.get('test', 0) for data in models_data.values()]
    
    bars = ax2.bar(model_names, test_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, score in zip(bars, test_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}%' if metric == 'accuracy' else f'{score:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel(f'Test {metric.capitalize()}', fontsize=11)
    ax2.set_title('Final Test Performance', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    if title:
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    # Print comparison table
    _print_comparison_table(models_data, metric)


def _print_comparison_table(models_data: Dict, metric: str):
    """Print comparison table"""
    print("\n" + "="*60)
    print(f"Model Comparison - {metric.capitalize()}")
    print("="*60)
    
    print(f"{'Model':<15} {'Test ' + metric.capitalize():<20} {'Improvement':<15}")
    print("-" * 60)
    
    test_scores = [(name, data.get('test', 0)) for name, data in models_data.items()]
    baseline = test_scores[0][1] if test_scores else 0
    
    for name, score in test_scores:
        improvement = ((score - baseline) / baseline * 100) if baseline > 0 else 0
        score_str = f"{score:.2f}%" if metric == 'accuracy' else f"{score:.4f}"
        improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
        
        print(f"{name:<15} {score_str:<20} {improvement_str:<15}")
    
    print("="*60 + "\n")


# ============================================================================
# OBJECT DETECTION VISUALIZATION (Faster R-CNN)
# ============================================================================

def visualize_predictions_original(images: List[torch.Tensor], 
                                   targets: List[Dict],
                                   predictions: List[Dict],
                                   img_dir: str,
                                   class_names: List[str] = None,
                                   score_threshold: float = 0.5,
                                   num_samples: int = 10,
                                   save_path: Optional[str] = None):
    """
    Visualize detection results at original resolution
    Shows ground truth (green dashed) and predictions (colored solid) on original images
    
    Args:
        images: list of image tensors (letterboxed)
        targets: list of target dicts with metadata
        predictions: list of prediction dicts
        img_dir: directory containing original images
        class_names: list of class names (default: ['__bg__', 'person', 'car'])
        score_threshold: confidence threshold for predictions
        num_samples: number of samples to visualize
        save_path: if provided, saves figure to this path
    
    Example:
        visualize_predictions_original(
            test_images, test_targets, test_predictions,
            './data/coco_val/val2017',
            num_samples=10,
            score_threshold=0.5
        )
    """
    
    if class_names is None:
        class_names = ['__bg__', 'person', 'car']
    
    n_samples = min(num_samples, len(images))
    
    # Create grid layout (2 rows x 5 cols for 10 images)
    n_cols = 5
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    
    if n_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx in range(n_samples):
        target = targets[idx]
        pred = predictions[idx] if predictions else None
        
        # Load original image
        filename = target['filename']
        img_orig = load_original_image(img_dir, filename)
        img_np = np.array(img_orig)
        
        # Extract metadata
        scale = target['scale']
        pad_left = target['pad_left']
        pad_top = target['pad_top']
        
        axes[idx].imshow(img_np)
        axes[idx].set_title(f'{filename[:20]}...', fontsize=10)
        axes[idx].axis('off')
        
        # Draw ground truth (green dashed)
        if 'boxes' in target and len(target['boxes']) > 0:
            gt_boxes_orig = restore_to_original(
                target['boxes'], scale, pad_left, pad_top
            )
            draw_boxes(axes[idx], gt_boxes_orig, target['labels'], 
                      class_names, color='green', style='--', linewidth=2)
        
        # Draw predictions (colored by class)
        if pred and 'boxes' in pred and len(pred['boxes']) > 0:
            scores = pred.get('scores', torch.ones(len(pred['boxes'])))
            mask = scores > score_threshold
            if mask.any():
                pred_boxes_orig = restore_to_original(
                    pred['boxes'][mask], scale, pad_left, pad_top
                )
                draw_boxes(axes[idx], pred_boxes_orig, pred['labels'][mask],
                          class_names, scores[mask], color='red', linewidth=2)
    
    # Hide empty subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def visualize_predictions(images: List[torch.Tensor], 
                         targets: List[Dict],
                         predictions: List[Dict],
                         class_names: List[str] = None,
                         score_threshold: float = 0.5,
                         num_samples: int = 4,
                         save_path: Optional[str] = None):
    """
    Visualize detection results (letterboxed view - for quick check)
    
    Args:
        images: list of image tensors
        targets: list of target dicts
        predictions: list of prediction dicts
        class_names: list of class names
        score_threshold: confidence threshold
        num_samples: number of samples to display
        save_path: if provided, saves figure to this path
    """
    
    if class_names is None:
        class_names = ['__bg__', 'person', 'car']
    
    n_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(1, n_samples, figsize=(5*n_samples, 5))
    
    if n_samples == 1:
        axes = [axes]
    
    for idx in range(n_samples):
        img = images[idx]
        target = targets[idx] if targets else None
        pred = predictions[idx] if predictions else None
        
        # Denormalize and convert to numpy
        img_np = denormalize_image(img)
        
        axes[idx].imshow(img_np)
        axes[idx].set_title(f'Sample {idx+1}')
        axes[idx].axis('off')
        
        # Draw ground truth (green)
        if target and 'boxes' in target and len(target['boxes']) > 0:
            draw_boxes(axes[idx], target['boxes'], target['labels'], 
                      class_names, color='green', style='--')
        
        # Draw predictions (red)
        if pred and 'boxes' in pred and len(pred['boxes']) > 0:
            scores = pred.get('scores', torch.ones(len(pred['boxes'])))
            mask = scores > score_threshold
            if mask.any():
                draw_boxes(axes[idx], pred['boxes'][mask], pred['labels'][mask],
                          class_names, scores[mask], color='red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def restore_to_original(boxes, scale, pad_left, pad_top):
    """
    Restore bounding boxes to original image coordinates
    
    Args:
        boxes: tensor of boxes in letterboxed coordinates [N, 4]
        scale: scaling factor
        pad_left, pad_top: padding offsets
    
    Returns:
        boxes in original coordinates
    """
    if len(boxes) == 0:
        return boxes
    
    boxes_orig = boxes.clone()
    
    # Extract values (handle both tensor and scalar)
    scale_val = scale.item() if torch.is_tensor(scale) else scale
    pad_left_val = pad_left.item() if torch.is_tensor(pad_left) else pad_left
    pad_top_val = pad_top.item() if torch.is_tensor(pad_top) else pad_top
    
    # Remove padding
    boxes_orig[:, [0, 2]] -= pad_left_val
    boxes_orig[:, [1, 3]] -= pad_top_val
    
    # Unscale
    boxes_orig /= scale_val
    
    return boxes_orig


def load_original_image(img_dir, filename):
    """Load original image from file"""
    img_path = os.path.join(img_dir, filename)
    return Image.open(img_path).convert('RGB')


def draw_boxes(ax, boxes, labels, class_names, scores=None, color='red', 
               style='-', linewidth=2):
    """
    Draw bounding boxes on axis
    
    Args:
        ax: matplotlib axis
        boxes: bounding boxes [N, 4]
        labels: class labels [N]
        class_names: list of class names
        scores: confidence scores [N] (optional)
        color: box color
        style: line style ('-' or '--')
        linewidth: line width
    """
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box[:4]
        
        # Convert to numpy if tensor
        x1 = x1.item() if torch.is_tensor(x1) else x1
        y1 = y1.item() if torch.is_tensor(y1) else y1
        x2 = x2.item() if torch.is_tensor(x2) else x2
        y2 = y2.item() if torch.is_tensor(y2) else y2
        
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                linewidth=linewidth, edgecolor=color, 
                                facecolor='none', linestyle=style)
        ax.add_patch(rect)
        
        # Add label
        label_idx = label.item() if hasattr(label, 'item') else label
        text = class_names[label_idx] if label_idx < len(class_names) else f'cls_{label_idx}'
        if scores is not None:
            score = scores[i].item() if hasattr(scores[i], 'item') else scores[i]
            text += f': {score:.2f}'
        
        ax.text(x1, y1-5, text, color=color, fontsize=10, weight='bold',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, boxstyle='round,pad=0.3'))


# ============================================================================
# SAMPLE PREDICTION VISUALIZATION (Classification Only)
# ============================================================================

def visualize_sample_predictions(model, 
                                test_loader, 
                                class_names: List[str],
                                num_samples: int = 10,
                                device: str = 'cuda',
                                save_path: Optional[str] = None):
    """
    Visualize random sample predictions from test set (Classification only)
    
    Args:
        model: Trained classification model
        test_loader: DataLoader for test set
        class_names: List of class names (e.g., ['0', '1', ..., '9'] for MNIST)
        num_samples: Number of samples to display (default: 10)
        device: Device to run inference ('cuda' or 'cpu')
        save_path: If provided, saves figure to this path
    
    Example:
        # MNIST
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        visualize_sample_predictions(model, test_loader, class_names, device='cuda')
        
        # FashionMNIST
        class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        visualize_sample_predictions(model, test_loader, class_names, device='cuda')
    """
    
    model.eval()
    
    # Collect all test data
    all_images = []
    all_labels = []
    all_predictions = []
    all_probs = []
    
    print(f"Collecting predictions from test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_images.extend(images.cpu())
            all_labels.extend(labels.cpu())
            all_predictions.extend(predicted.cpu())
            all_probs.extend(probs.cpu())
    
    # Convert to tensors
    all_images = torch.stack(all_images)
    all_labels = torch.tensor(all_labels)
    all_predictions = torch.tensor(all_predictions)
    all_probs = torch.stack(all_probs)
    
    # Random sampling
    n_total = len(all_images)
    n_samples = min(num_samples, n_total)
    
    random.seed(42)  # For reproducibility
    random_indices = random.sample(range(n_total), n_samples)
    
    # Create visualization grid (2 rows x 5 columns)
    n_cols = 5
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6))
    
    # Flatten axes for easier indexing
    if n_samples == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Visualize each sample
    for idx, sample_idx in enumerate(random_indices):
        img = all_images[sample_idx]
        true_label = all_labels[sample_idx].item()
        pred_label = all_predictions[sample_idx].item()
        confidence = all_probs[sample_idx][pred_label].item()
        
        # Denormalize image
        img_np = denormalize_image(img)
        
        # Display image
        if img_np.ndim == 2:  # Grayscale
            axes[idx].imshow(img_np, cmap='gray')
        else:  # RGB
            axes[idx].imshow(img_np)
        
        # Set title with prediction info
        correct = (true_label == pred_label)
        color = 'green' if correct else 'red'
        
        true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
        pred_name = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)
        
        title = f'True: {true_name}\n'
        title += f'Pred: {pred_name} ({confidence:.1%})'
        
        axes[idx].set_title(title, color=color, fontsize=10, fontweight='bold')
        axes[idx].axis('off')
        
        # Add colored border
        for spine in axes[idx].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)
    
    # Hide unused subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    # Add overall accuracy info
    correct_count = (all_predictions == all_labels).sum().item()
    total_count = len(all_labels)
    accuracy = 100. * correct_count / total_count
    
    # Count correct/incorrect in displayed samples
    displayed_correct = sum(1 for idx in random_indices 
                           if all_predictions[idx] == all_labels[idx])
    
    plt.suptitle(f'Sample Predictions (Showing {displayed_correct}/{n_samples} correct) - '
                f'Overall Test Accuracy: {accuracy:.2f}%',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    print(f"✓ Visualized {n_samples} random samples")
    print(f"  Correct predictions in samples: {displayed_correct}/{n_samples}")
    print(f"  Overall test accuracy: {accuracy:.2f}%")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def denormalize_image(img: torch.Tensor) -> np.ndarray:
    """
    Denormalize and convert tensor to numpy for visualization
    
    Args:
        img: Tensor image (C, H, W) or (B, C, H, W)
    
    Returns:
        Numpy array image (H, W, C) or (H, W)
    """
    if len(img.shape) == 4:
        img = img[0]
    
    img = img.cpu()
    if img.shape[0] == 3:  # RGB
        img = img.permute(1, 2, 0)
        # ImageNet normalization
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
    elif img.shape[0] == 1:  # Grayscale
        img = img.squeeze(0)
        # Common grayscale normalizations
        # Check if it's FashionMNIST/MNIST style normalization
        if img.min() < 0:  # Normalized data
            img = img * 0.3081 + 0.1307  # FashionMNIST/MNIST
    
    img = torch.clamp(img, 0, 1)
    return img.numpy()