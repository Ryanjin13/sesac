"""
Evaluation metrics for object detection
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    # Convert to numpy if tensor
    if isinstance(box1, torch.Tensor):
        box1 = box1.cpu().numpy()
    if isinstance(box2, torch.Tensor):
        box2 = box2.cpu().numpy()
    
    x1_min, y1_min, x1_max, y1_max = box1[:4]
    x2_min, y2_min, x2_max, y2_max = box2[:4]
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return float(inter_area / union_area) if union_area > 0 else 0.0


def evaluate_detection(predictions: List[Dict], 
                      targets: List[Dict],
                      iou_threshold: float = 0.5,
                      score_threshold: float = 0.5) -> Dict:
    """
    Evaluate detection performance
    Returns: dict with metrics (precision, recall, f1, avg_iou)
    """
    
    all_ious = []
    class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for pred, target in zip(predictions, targets):
        if len(target['boxes']) == 0:
            # No ground truth
            if len(pred['boxes']) > 0 and 'scores' in pred:
                # All predictions are false positives
                high_conf_mask = pred['scores'] > score_threshold
                for label in pred['labels'][high_conf_mask]:
                    label_val = label.item() if hasattr(label, 'item') else label
                    class_metrics[label_val]['fp'] += 1
            continue
        
        # Match predictions to ground truth
        matched_gt = set()
        
        # Filter predictions by score
        if 'scores' in pred:
            score_mask = pred['scores'] > score_threshold
            pred_boxes = pred['boxes'][score_mask]
            pred_labels = pred['labels'][score_mask]
        else:
            pred_boxes = pred['boxes']
            pred_labels = pred['labels']
        
        # For each prediction, find best matching GT
        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            best_iou = 0
            best_gt_idx = -1
            pred_label_val = pred_label.item() if hasattr(pred_label, 'item') else pred_label
            
            for gt_idx, (gt_box, gt_label) in enumerate(zip(target['boxes'], target['labels'])):
                if gt_idx in matched_gt:
                    continue
                
                gt_label_val = gt_label.item() if hasattr(gt_label, 'item') else gt_label
                if pred_label_val != gt_label_val:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou > iou_threshold:
                # True positive
                matched_gt.add(best_gt_idx)
                class_metrics[pred_label_val]['tp'] += 1
                all_ious.append(best_iou)
            else:
                # False positive
                class_metrics[pred_label_val]['fp'] += 1
        
        # False negatives (unmatched ground truth)
        for gt_idx, gt_label in enumerate(target['labels']):
            if gt_idx not in matched_gt:
                gt_label_val = gt_label.item() if hasattr(gt_label, 'item') else gt_label
                class_metrics[gt_label_val]['fn'] += 1
    
    # Calculate final metrics
    results = calculate_metrics(class_metrics)
    results['avg_iou'] = np.mean(all_ious) if all_ious else 0.0
    
    return results


def calculate_metrics(class_metrics: Dict) -> Dict:
    """Calculate precision, recall, F1 from class metrics"""
    
    total_tp = sum(m['tp'] for m in class_metrics.values())
    total_fp = sum(m['fp'] for m in class_metrics.values())
    total_fn = sum(m['fn'] for m in class_metrics.values())
    
    # Overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn
    }


def print_results_table(results: Dict, step_name: str = None):
    """Print formatted results table"""
    
    print("\n" + "="*60)
    if step_name:
        print(f"{step_name} - Evaluation Results")
    else:
        print("Evaluation Results")
    print("="*60)
    
    if 'accuracy' in results:
        print(f"Accuracy:  {results['accuracy']:.4f}")
    
    if 'precision' in results:
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1-Score:  {results['f1']:.4f}")
    
    if 'avg_iou' in results:
        print(f"Avg IoU:   {results['avg_iou']:.4f}")
    
    if 'total_tp' in results:
        print(f"\nDetection counts:")
        print(f"  True Positives:  {results.get('total_tp', 0)}")
        print(f"  False Positives: {results.get('total_fp', 0)}")  
        print(f"  False Negatives: {results.get('total_fn', 0)}")
    
    print("="*60)