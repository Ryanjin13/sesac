"""
Universal COCO Dataset utilities for flexible class detection
Supports any COCO-format dataset (CVAT exports, custom annotations, etc.)
"""

import os
import json
import requests
import zipfile
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

try:
    from pycocotools.coco import COCO
except ImportError:
    raise SystemExit("pycocotools가 필요합니다. `pip install pycocotools` 를 먼저 실행하세요.")


def letterbox_resize(image, target_size=224, fill_color=(114, 114, 114)):
    """
    Resize image with aspect ratio preserved using letterboxing
    
    Args:
        image: PIL Image
        target_size: target dimension (square)
        fill_color: padding color
    
    Returns:
        resized_image: letterboxed image
        scale: scaling factor applied
        pad_left, pad_top: padding offsets
    """
    orig_width, orig_height = image.size
    
    # Calculate scale to fit longer side to target_size
    scale = target_size / max(orig_width, orig_height)
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    
    # Resize image
    resized = image.resize((new_width, new_height), Image.BILINEAR)
    
    # Create canvas with padding
    canvas = Image.new('RGB', (target_size, target_size), fill_color)
    
    # Calculate padding to center the image
    pad_left = (target_size - new_width) // 2
    pad_top = (target_size - new_height) // 2
    
    # Paste resized image onto canvas
    canvas.paste(resized, (pad_left, pad_top))
    
    return canvas, scale, pad_left, pad_top


def transform_bbox_letterbox(bbox, scale, pad_left, pad_top):
    """
    Transform bounding box coordinates for letterboxed image
    
    Args:
        bbox: [x, y, w, h] in original image coordinates
        scale: scaling factor
        pad_left, pad_top: padding offsets
    
    Returns:
        transformed bbox: [x1, y1, x2, y2] in letterboxed coordinates
    """
    x, y, w, h = bbox
    
    # Scale coordinates
    x1 = x * scale + pad_left
    y1 = y * scale + pad_top
    x2 = (x + w) * scale + pad_left
    y2 = (y + h) * scale + pad_top
    
    return [x1, y1, x2, y2]


def inverse_transform_bbox(bbox, scale, pad_left, pad_top):
    """
    Transform bounding box from letterboxed coordinates back to original
    
    Args:
        bbox: [x1, y1, x2, y2] in letterboxed coordinates
        scale: scaling factor used
        pad_left, pad_top: padding offsets used
    
    Returns:
        original bbox: [x1, y1, x2, y2] in original image coordinates
    """
    x1, y1, x2, y2 = bbox
    
    # Remove padding and unscale
    x1_orig = (x1 - pad_left) / scale
    y1_orig = (y1 - pad_top) / scale
    x2_orig = (x2 - pad_left) / scale
    y2_orig = (y2 - pad_top) / scale
    
    return [x1_orig, y1_orig, x2_orig, y2_orig]


def download_coco_val2017(data_root='./data'):
    """COCO val2017 다운로드 (1GB) - 기본 데이터셋용"""
    coco_dir = os.path.join(data_root, 'coco_val')
    val_dir = os.path.join(coco_dir, 'val2017')
    ann_file = os.path.join(coco_dir, 'annotations', 'instances_val2017.json')
    
    if os.path.exists(val_dir) and os.path.exists(ann_file):
        print("✓ COCO val2017 already exists!")
        return coco_dir
    
    os.makedirs(coco_dir, exist_ok=True)
    os.makedirs(os.path.join(coco_dir, 'annotations'), exist_ok=True)
    
    urls = {
        'val2017.zip': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations_trainval2017.zip': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }
    
    print("="*60)
    print("Downloading COCO val2017 (Total: ~1GB)")
    print("="*60)
    
    for filename, url in urls.items():
        filepath = os.path.join(coco_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"\nDownloading {filename}...")
            try:
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                
                with open(filepath, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            except Exception as e:
                print(f"Download failed: {e}")
                raise
        
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(coco_dir)
        os.remove(filepath)
    
    print("✓ COCO val2017 ready!")
    return coco_dir


class UniversalCOCODataset(Dataset):
    """
    Universal COCO Dataset for flexible class detection
    Supports any COCO-format annotations (CVAT exports, custom datasets, etc.)
    
    Example directory structure:
        ./data/my_dataset/
            ├── images/
            │   ├── img1.jpg
            │   ├── img2.jpg
            │   └── ...
            └── annotations/
                └── instances.json  (COCO format)
    """
    
    def __init__(self, 
                 root, 
                 annotation_file,
                 image_folder,
                 class_names,
                 split='train', 
                 transform=None, 
                 max_images=None, 
                 resize_size=224, 
                 detection_mode=True):
        """
        Args:
            root: Dataset root directory (e.g., './data/my_dataset')
            annotation_file: Annotation filename (e.g., 'instances.json')
            image_folder: Image folder name (e.g., 'images' or 'val2017')
            class_names: List of class names to detect (e.g., ['person', 'car', 'bicycle'])
                        Note: Should NOT include '__background__'
            split: 'train', 'val', or 'test'
            transform: Additional transforms
            max_images: Maximum number of images to use
            resize_size: Target image size (square)
            detection_mode: True for detection, False for classification
        
        Example:
            dataset = UniversalCOCODataset(
                root='./data/my_cvat_export',
                annotation_file='instances.json',
                image_folder='images',
                class_names=['cat', 'dog', 'bird'],
                split='train',
                resize_size=640
            )
        """
        self.root = root
        self.transform = transform
        self.resize_size = resize_size
        self.detection_mode = detection_mode
        self.class_names = class_names  # User-specified classes (without background)
        
        # Full class names with background
        self.class_names_with_bg = ['__background__'] + class_names
        
        # Image directory
        self.img_dir = os.path.join(root, image_folder)
        
        # Initialize COCO API
        ann_path = os.path.join(root, annotation_file)
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")
        
        print(f"Loading annotations from: {ann_path}")
        self.coco = COCO(ann_path)
        
        # Build category name to ID mapping
        self.cat_name_to_id = {}
        self.cat_id_to_name = {}
        for cat in self.coco.loadCats(self.coco.getCatIds()):
            self.cat_name_to_id[cat['name']] = cat['id']
            self.cat_id_to_name[cat['id']] = cat['name']
        
        # Get COCO category IDs for requested classes
        self.target_cat_ids = []
        self.coco_id_to_label = {}  # COCO cat_id -> our label (1, 2, 3, ...)
        
        for idx, class_name in enumerate(class_names, start=1):
            if class_name in self.cat_name_to_id:
                coco_cat_id = self.cat_name_to_id[class_name]
                self.target_cat_ids.append(coco_cat_id)
                self.coco_id_to_label[coco_cat_id] = idx
            else:
                print(f"⚠️  Warning: Class '{class_name}' not found in annotations!")
        
        if not self.target_cat_ids:
            raise ValueError(f"None of the specified classes {class_names} found in annotations!")
        
        print(f"✓ Found {len(self.target_cat_ids)} classes: {class_names}")
        print(f"  COCO category IDs: {self.target_cat_ids}")
        
        # Get all image IDs containing target classes
        self.image_ids = self.coco.getImgIds(catIds=self.target_cat_ids)
        
        # Limit images if specified
        if max_images:
            self.image_ids = self.image_ids[:max_images]
        
        # Train/val/test split (70/15/15)
        np.random.seed(42)
        np.random.shuffle(self.image_ids)
        
        n_total = len(self.image_ids)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        if split == 'train':
            self.image_ids = self.image_ids[:n_train]
        elif split == 'val':
            self.image_ids = self.image_ids[n_train:n_train+n_val]
        else:  # test
            self.image_ids = self.image_ids[n_train+n_val:]
        
        print(f"  {split} split: {len(self.image_ids)} images")
        
        # Normalization for tensor
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        # Load image info using COCO API
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Load image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        
        # Apply letterbox resize
        img_letterbox, scale, pad_left, pad_top = letterbox_resize(
            img, self.resize_size
        )
        
        # Get annotations using COCO API
        ann_ids = self.coco.getAnnIds(
            imgIds=img_id, 
            catIds=self.target_cat_ids,
            iscrowd=None
        )
        anns = self.coco.loadAnns(ann_ids)
        
        if self.detection_mode:
            # Detection mode: return boxes and labels
            boxes = []
            labels = []
            
            for ann in anns:
                x, y, w, h = ann['bbox']
                if w > 10 and h > 10:  # Filter tiny boxes
                    # Transform bbox for letterboxed image
                    transformed_bbox = transform_bbox_letterbox(
                        [x, y, w, h], scale, pad_left, pad_top
                    )
                    boxes.append(transformed_bbox)
                    
                    # Map COCO category ID to our label
                    coco_cat_id = ann['category_id']
                    label = self.coco_id_to_label.get(coco_cat_id, 0)
                    if label > 0:  # Valid class
                        labels.append(label)
            
            # Apply normalization
            img_tensor = self.normalize(img_letterbox)
            
            # Convert to tensors
            if len(boxes) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
            
            target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([idx]),
                # Store transform metadata for inverse transform
                'orig_size': torch.tensor([orig_w, orig_h]),
                'scale': torch.tensor(scale),
                'pad_left': torch.tensor(pad_left),
                'pad_top': torch.tensor(pad_top),
                'filename': img_info['file_name']
            }
            
            return img_tensor, target
        else:
            # Classification mode: return most frequent class
            # (or you can implement multi-label classification)
            img_tensor = self.normalize(img_letterbox)
            
            # Get most frequent class in image
            if anns:
                cat_ids = [ann['category_id'] for ann in anns]
                most_common_cat_id = max(set(cat_ids), key=cat_ids.count)
                label = self.coco_id_to_label.get(most_common_cat_id, 0)
            else:
                label = 0
            
            label = torch.tensor(label, dtype=torch.long)
            
            return img_tensor, label


def get_data_loaders(root, 
                     annotation_file,
                     image_folder,
                     class_names,
                     batch_size=4, 
                     max_images=None, 
                     resize_size=224, 
                     detection_mode=True):
    """
    Get train, val, test data loaders for any COCO-format dataset
    
    Args:
        root: Dataset root directory
        annotation_file: Annotation filename
        image_folder: Image folder name
        class_names: List of class names (without '__background__')
        batch_size: Batch size
        max_images: Maximum images per split
        resize_size: Target image size
        detection_mode: True for detection, False for classification
    
    Returns:
        train_loader, val_loader, test_loader
    
    Example:
        train_loader, val_loader, test_loader = get_data_loaders(
            root='./data/my_dataset',
            annotation_file='annotations.json',
            image_folder='images',
            class_names=['person', 'car', 'bicycle'],
            batch_size=16,
            resize_size=640
        )
    """
    
    train_dataset = UniversalCOCODataset(
        root, annotation_file, image_folder, class_names,
        'train', None, max_images, resize_size, detection_mode
    )
    val_dataset = UniversalCOCODataset(
        root, annotation_file, image_folder, class_names,
        'val', None, max_images, resize_size, detection_mode
    )
    test_dataset = UniversalCOCODataset(
        root, annotation_file, image_folder, class_names,
        'test', None, max_images, resize_size, detection_mode
    )
    
    collate = collate_fn if detection_mode else None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, collate_fn=collate, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, collate_fn=collate, num_workers=0)
    
    return train_loader, val_loader, test_loader


def collate_fn(batch):
    """Custom collate for varying number of objects"""
    return tuple(zip(*batch))
