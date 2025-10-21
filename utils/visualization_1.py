"""
Visualization utilities for object detection
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np
from typing import List, Dict, Optional
from PIL import Image
import os
import random


def plot_training_history(history, title="Training History"):
    """
    Plot training history with proper train/val comparison
    Left: Train Loss vs Val Loss (for overfitting detection)
    Right: Train IoU vs Val IoU
    
    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'train_iou', 'val_iou'
        title: Title for the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # ========== Left plot: Loss Comparison ==========
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, 
             marker='o', markersize=6, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r--', linewidth=2,
             marker='s', markersize=6, label='Val Loss')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Comparison (Overfitting Check)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Mark potential overfitting point
    for i in range(1, len(history['val_loss'])):
        if history['val_loss'][i] > history['val_loss'][i-1] and history['train_loss'][i] < history['train_loss'][i-1]:
            ax1.axvline(x=i+1, color='orange', linestyle=':', alpha=0.5)
            ax1.text(i+1, ax1.get_ylim()[1]*0.9, 'Potential\nOverfitting', 
                    ha='center', fontsize=8, color='orange')
            break
    
    # ========== Right plot: IoU Comparison ==========
    ax2.plot(epochs, history['train_iou'], 'b-', linewidth=2,
             marker='o', markersize=6, label='Train IoU')
    ax2.plot(epochs, history['val_iou'], 'g--', linewidth=2,
             marker='s', markersize=6, label='Val IoU')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('IoU', fontsize=12)
    ax2.set_title('IoU Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Annotate best validation IoU
    best_iou_idx = np.argmax(history['val_iou'])
    ax2.annotate(f'Best Val IoU: {history["val_iou"][best_iou_idx]:.4f}',
                xy=(best_iou_idx + 1, history['val_iou'][best_iou_idx]),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Print analysis
    print("\n" + "="*60)
    print("Training Analysis:")
    print("="*60)
    
    # Check for overfitting
    train_loss_decrease = history['train_loss'][0] - history['train_loss'][-1]
    val_loss_change = history['val_loss'][-1] - history['val_loss'][0]
    
    if val_loss_change > 0 and train_loss_decrease > 0:
        print("⚠️  Potential overfitting detected:")
        print(f"   Train loss decreased by {train_loss_decrease:.4f}")
        print(f"   Val loss increased by {val_loss_change:.4f}")
    else:
        print("✓ No clear overfitting detected")
    
    # Performance summary
    print(f"\nFinal Performance:")
    print(f"  Train Loss: {history['train_loss'][-1]:.4f} | Train IoU: {history['train_iou'][-1]:.4f}")
    print(f"  Val Loss:   {history['val_loss'][-1]:.4f} | Val IoU:   {history['val_iou'][-1]:.4f}")
    print(f"  Best Val IoU: {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou'])+1})")
    print("="*60)



# Alternative: Normalized plot (both metrics scaled to 0-1)
def plot_training_history_normalized(history, title="Training History"):
    """
    Plot training history with normalized values (0-1 scale)
    This makes it easier to compare trends
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Normalize values to 0-1 range
    def normalize(data):
        min_val = min(data)
        max_val = max(data)
        return [(x - min_val) / (max_val - min_val) if max_val > min_val else 0 for x in data]
    
    norm_loss = normalize(history['train_loss'])
    norm_iou = normalize(history['val_iou'])
    
    # Plot both metrics
    ax.plot(epochs, norm_loss, 'b-', linewidth=2, marker='o', 
            markersize=6, label='Train Loss (normalized)')
    ax.plot(epochs, norm_iou, 'g-', linewidth=2, marker='s', 
            markersize=6, label='Validation IoU (normalized)')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Normalized Value (0-1)', fontsize=12)
    ax.set_title(f'{title} - Normalized', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add actual value annotations
    for i, epoch in enumerate(epochs):
        if i == 0 or i == len(epochs) - 1:  # First and last epoch
            ax.annotate(f'{history["train_loss"][i]:.3f}', 
                       xy=(epoch, norm_loss[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, color='blue')
            ax.annotate(f'{history["val_iou"][i]:.3f}', 
                       xy=(epoch, norm_iou[i]), 
                       xytext=(5, -15), textcoords='offset points',
                       fontsize=8, color='green')
    
    plt.tight_layout()
    plt.show()


# Usage in your code:
# Just replace the existing plot_training_history call with one of these:
# plot_training_history(history, "Step 4: Faster R-CNN")  # Dual axes version
# OR
# plot_training_history_normalized(history, "Step 4: Faster R-CNN")  # Normalized version


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


def visualize_predictions_original(images: List[torch.Tensor], 
                                   targets: List[Dict],
                                   predictions: List[Dict],
                                   img_dir: str,
                                   class_names: List[str] = None,
                                   score_threshold: float = 0.5,
                                   num_samples: int = 10):
    """
    Visualize detection results at original resolution
    
    Args:
        images: list of image tensors (letterboxed)
        targets: list of target dicts with metadata
        predictions: list of prediction dicts
        img_dir: directory containing original images
        class_names: list of class names
        score_threshold: confidence threshold for predictions
        num_samples: number of samples to visualize
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
        
        # Draw ground truth (green)
        if 'boxes' in target and len(target['boxes']) > 0:
            gt_boxes_orig = restore_to_original(
                target['boxes'], scale, pad_left, pad_top
            )
            draw_boxes(axes[idx], gt_boxes_orig, target['labels'], 
                      class_names, color='green', style='--', linewidth=2)
        
        # Draw predictions (red)
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
    plt.show()


def visualize_predictions(images: List[torch.Tensor], 
                         targets: List[Dict],
                         predictions: List[Dict],
                         class_names: List[str] = None,
                         score_threshold: float = 0.5,
                         num_samples: int = 4):
    """
    Visualize detection results (letterboxed view - for quick check)
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
    plt.show()


def visualize_grid_output(image: torch.Tensor, 
                         grid_output: torch.Tensor,
                         grid_size: int = 7,
                         threshold: float = 0.5):
    """Grid 검출 출력 시각화 (YOLO 스타일)"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    img_np = denormalize_image(image)
    axes[0].imshow(img_np)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Grid predictions
    axes[1].imshow(img_np)
    h, w = img_np.shape[:2]
    cell_h, cell_w = h / grid_size, w / grid_size
    
    # Draw grid
    for i in range(grid_size + 1):
        axes[1].axhline(y=i*cell_h, color='gray', alpha=0.3, linewidth=0.5)
        axes[1].axvline(x=i*cell_w, color='gray', alpha=0.3, linewidth=0.5)
    
    # Draw predictions from grid
    for i in range(grid_size):
        for j in range(grid_size):
            if grid_output[i, j, 4] > threshold:  # confidence
                cx = (j + grid_output[i, j, 0]) * cell_w
                cy = (i + grid_output[i, j, 1]) * cell_h
                w_box = grid_output[i, j, 2] * w
                h_box = grid_output[i, j, 3] * h
                
                x1 = cx - w_box/2
                y1 = cy - h_box/2
                
                rect = patches.Rectangle((x1, y1), w_box, h_box,
                                        linewidth=2, edgecolor='red', 
                                        facecolor='none')
                axes[1].add_patch(rect)
    
    axes[1].set_title('Grid Detection Output')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def show_dataset_samples(dataset, num_samples: int = 6):
    """데이터셋 샘플 확인"""
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
            
        img, target = dataset[idx]
        img_np = denormalize_image(img)
        
        axes[i].imshow(img_np)
        
        # Detection mode
        if isinstance(target, dict):
            if 'boxes' in target and len(target['boxes']) > 0:
                for box, label in zip(target['boxes'], target['labels']):
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                            linewidth=2, edgecolor='green', 
                                            facecolor='none')
                    axes[i].add_patch(rect)
            
            n_persons = (target['labels'] == 1).sum() if 'labels' in target else 0
            n_cars = (target['labels'] == 2).sum() if 'labels' in target else 0
            axes[i].set_title(f'Persons: {n_persons}, Cars: {n_cars}')
        else:
            # Classification mode
            axes[i].set_title(f'Class: {"Car" if target == 1 else "Person"}')
        
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


# Helper functions
def denormalize_image(img: torch.Tensor) -> np.ndarray:
    """Denormalize and convert tensor to numpy"""
    if len(img.shape) == 4:
        img = img[0]
    
    img = img.cpu()
    if img.shape[0] == 3:  # RGB
        img = img.permute(1, 2, 0)
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
    else:  # Grayscale
        img = img.squeeze(0)
    
    img = torch.clamp(img, 0, 1)
    return img.numpy()


def draw_boxes(ax, boxes, labels, class_names, scores=None, color='red', 
               style='-', linewidth=2):
    """Draw bounding boxes on axis"""
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
        

def visualize_classification_predictions(model, test_loader, class_names, 
                                        num_samples=10, device='cuda'):
    """
    Visualize random classification predictions from test set
    
    Args:
        model: Trained classification model
        test_loader: DataLoader for test set
        class_names: List of class names
        num_samples: Number of random samples to display
        device: Device to run inference on
    """
    model.eval()
    
    # Collect all test data
    all_images = []
    all_labels = []
    all_predictions = []
    all_probs = []
    
    print("Collecting predictions...")
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
    random_indices = random.sample(range(n_total), n_samples)
    
    # Create visualization grid (2 rows x 5 columns for 10 samples)
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
        confidence = all_probs[sample_idx].max().item()
        
        # Denormalize image (assuming standard ImageNet normalization)
        if img.shape[0] == 3:  # RGB
            img = img.permute(1, 2, 0)
            img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        elif img.shape[0] == 1:  # Grayscale
            img = img.squeeze(0)
            img = img * 0.5 + 0.5  # For FashionMNIST normalization
        img = torch.clamp(img, 0, 1).numpy()
        
        # Display image
        axes[idx].imshow(img, cmap='gray' if img.ndim == 2 else None)
        
        # Set title with prediction info
        correct = true_label == pred_label
        color = 'green' if correct else 'red'
        
        # Format title
        title = f'True: {class_names[true_label]}\n'
        title += f'Pred: {class_names[pred_label]}'
        title += f' ({confidence:.2%})'
        
        axes[idx].set_title(title, color=color, fontsize=9, fontweight='bold')
        axes[idx].axis('off')
        
        # Add colored border for correct/incorrect
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
    accuracy = correct_count / total_count
    
    plt.suptitle(f'Classification Predictions - Test Accuracy: {accuracy:.2%} ({correct_count}/{total_count})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return accuracy