#!/usr/bin/env python3
"""
Generate composite images with multiple shapes on grey background.
Reads from dataset/selected_shapes.json for balanced sampling.
Output: YOLO format dataset with images and labels.
"""
# IMPORTANT: Import SSL config FIRST before any network operations
from ssl_config import setup_ssl_certificates
setup_ssl_certificates()

import json
import random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple
import argparse

def load_selected_shapes(json_path: str) -> Dict[str, List[str]]:
    """Load balanced shape paths from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['shapes']

def load_shape_image(path: str) -> np.ndarray:
    """
    Load PNG shape and extract silhouette mask from alpha channel.
    Returns binary mask (0 or 255).
    """
    # Load image with alpha channel
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Could not load image: {path}")

    # Extract alpha channel as mask
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
    else:
        # If no alpha, assume white pixels are foreground
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        alpha = np.where(gray > 127, 255, 0).astype(np.uint8)

    # Threshold to binary
    _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

    return mask

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by angle (degrees)."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust rotation matrix for new dimensions
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Rotate with expanded canvas
    rotated = cv2.warpAffine(image, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)

    return rotated

def get_tight_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get tight bounding box around non-zero pixels.
    Returns (x, y, width, height).
    """
    coords = cv2.findNonZero(mask)
    if coords is None:
        return 0, 0, 0, 0

    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h

def check_overlap(bbox1: Tuple[int, int, int, int],
                  bbox2: Tuple[int, int, int, int],
                  min_separation: int = 10) -> bool:
    """
    Check if two bounding boxes overlap (with separation margin).
    bbox format: (x, y, width, height)
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Expand boxes by separation margin
    x1_min, y1_min = x1 - min_separation, y1 - min_separation
    x1_max, y1_max = x1 + w1 + min_separation, y1 + h1 + min_separation

    x2_min, y2_min = x2 - min_separation, y2 - min_separation
    x2_max, y2_max = x2 + w2 + min_separation, y2 + h2 + min_separation

    # Check overlap
    return not (x1_max < x2_min or x2_max < x1_min or
                y1_max < y2_min or y2_max < y1_min)

def paste_shape_augmented(canvas: np.ndarray,
                          shape_mask: np.ndarray,
                          class_id: int,
                          scale: float,
                          rotation: float,
                          existing_boxes: List[Tuple[int, int, int, int]],
                          max_attempts: int = 50) -> Tuple[bool, Tuple[int, int, int, int]]:
    """
    Paste shape onto canvas with random position (non-overlapping).

    Returns:
        (success, bbox): bbox is (x, y, width, height) if success, else None
    """
    canvas_h, canvas_w = canvas.shape[:2]

    # Resize shape
    shape_h, shape_w = shape_mask.shape
    new_w = int(shape_w * scale)
    new_h = int(shape_h * scale)

    if new_w < 10 or new_h < 10:  # Too small
        return False, None

    resized = cv2.resize(shape_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Rotate
    rotated = rotate_image(resized, rotation)

    # Get tight bounding box after rotation
    x, y, w, h = get_tight_bbox(rotated)
    if w == 0 or h == 0:
        return False, None

    # Crop to tight box
    cropped_mask = rotated[y:y+h, x:x+w]

    # Try to find non-overlapping position
    for _ in range(max_attempts):
        # Random position
        paste_x = random.randint(0, max(1, canvas_w - w))
        paste_y = random.randint(0, max(1, canvas_h - h))

        candidate_box = (paste_x, paste_y, w, h)

        # Check overlap with existing boxes
        overlaps = any(check_overlap(candidate_box, existing, min_separation=15)
                      for existing in existing_boxes)

        if not overlaps:
            # Paste shape (white on grey background)
            for i in range(h):
                for j in range(w):
                    if cropped_mask[i, j] > 127:  # Foreground pixel
                        canvas_y = paste_y + i
                        canvas_x = paste_x + j
                        if 0 <= canvas_y < canvas_h and 0 <= canvas_x < canvas_w:
                            canvas[canvas_y, canvas_x] = 255  # White

            return True, candidate_box

    return False, None

def bbox_to_yolo_format(bbox: Tuple[int, int, int, int],
                        img_width: int,
                        img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert bbox to YOLO format: (x_center, y_center, width, height) normalized.
    Input bbox: (x, y, width, height) in pixels
    """
    x, y, w, h = bbox

    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height

    return x_center, y_center, width, height

def generate_composite_image(shapes_dict: Dict[str, List[str]],
                             img_size: int = 640,
                             num_shapes_range: Tuple[int, int] = (2, 5)) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    """
    Generate one composite image with multiple shapes.

    Returns:
        (image, labels): labels is list of (class_id, x_center, y_center, width, height)
    """
    # Create grey background
    canvas = np.full((img_size, img_size), 128, dtype=np.uint8)

    # Random number of shapes
    num_shapes = random.randint(*num_shapes_range)

    labels = []
    existing_boxes = []
    category_counts = {f'cat{i}': 0 for i in range(1, 6)}

    for _ in range(num_shapes):
        # Randomly pick category
        category = random.choice(list(shapes_dict.keys()))
        category_counts[category] += 1

        # Get class ID (cat1 -> 0, cat2 -> 1, etc.)
        class_id = int(category.replace('cat', '')) - 1

        # Randomly pick shape from that category
        shape_path = random.choice(shapes_dict[category])

        try:
            # Load shape
            shape_mask = load_shape_image(shape_path)

            # Random augmentation parameters
            scale = random.uniform(0.1, 0.3)  # 10-30% of image size
            rotation = random.uniform(0, 360)

            # Paste shape
            success, bbox = paste_shape_augmented(
                canvas, shape_mask, class_id, scale, rotation, existing_boxes
            )

            if success:
                existing_boxes.append(bbox)

                # Convert to YOLO format
                x_c, y_c, w, h = bbox_to_yolo_format(bbox, img_size, img_size)
                labels.append((class_id, x_c, y_c, w, h))

        except Exception as e:
            print(f"Warning: Failed to process {shape_path}: {e}")
            continue

    return canvas, labels

def generate_dataset(shapes_json: str,
                     output_dir: str,
                     num_train: int = 500,
                     num_test: int = 125,
                     img_size: int = 640,
                     seed: int = 42):
    """
    Generate complete YOLO dataset.

    Args:
        shapes_json: Path to selected_shapes.json
        output_dir: Output directory for dataset
        num_train: Number of training images
        num_test: Number of test images
        img_size: Image size (square)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("Generating YOLO Composite Images Dataset")
    print("=" * 70)

    # Load shape paths
    print(f"\nLoading shape paths from: {shapes_json}")
    shapes_dict = load_selected_shapes(shapes_json)

    total_shapes = sum(len(paths) for paths in shapes_dict.values())
    print(f"Loaded {len(shapes_dict)} categories with {total_shapes} total shapes")

    # Create output directories
    output_path = Path(output_dir)
    for split in ['train', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Track category distribution
    category_distribution = {f'cat{i}': 0 for i in range(1, 6)}

    # Generate training set
    print(f"\nGenerating {num_train} training images...")
    for i in range(num_train):
        img, labels = generate_composite_image(shapes_dict, img_size)

        # Save image
        img_path = output_path / 'images' / 'train' / f'train_{i:04d}.jpg'
        cv2.imwrite(str(img_path), img)

        # Save labels
        label_path = output_path / 'labels' / 'train' / f'train_{i:04d}.txt'
        with open(label_path, 'w') as f:
            for class_id, x_c, y_c, w, h in labels:
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                category_distribution[f'cat{class_id + 1}'] += 1

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{num_train}")

    print(f"✓ Training set complete: {num_train} images")

    # Generate test set
    print(f"\nGenerating {num_test} test images...")
    for i in range(num_test):
        img, labels = generate_composite_image(shapes_dict, img_size)

        # Save image
        img_path = output_path / 'images' / 'test' / f'test_{i:04d}.jpg'
        cv2.imwrite(str(img_path), img)

        # Save labels
        label_path = output_path / 'labels' / 'test' / f'test_{i:04d}.txt'
        with open(label_path, 'w') as f:
            for class_id, x_c, y_c, w, h in labels:
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                category_distribution[f'cat{class_id + 1}'] += 1

        if (i + 1) % 25 == 0:
            print(f"  Progress: {i + 1}/{num_test}")

    print(f"✓ Test set complete: {num_test} images")

    # Summary
    print("\n" + "=" * 70)
    print("Dataset Generation Complete!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"  Training images: {num_train}")
    print(f"  Test images: {num_test}")
    print(f"  Image size: {img_size}x{img_size}")

    print(f"\nCategory distribution across all images:")
    total_instances = sum(category_distribution.values())
    for cat, count in sorted(category_distribution.items()):
        percentage = (count / total_instances * 100) if total_instances > 0 else 0
        print(f"  {cat}: {count} instances ({percentage:.1f}%)")

    print(f"\nTotal shape instances: {total_instances}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate YOLO composite images dataset')
    parser.add_argument('--shapes-json', type=str,
                       default='dataset/selected_shapes.json',
                       help='Path to selected shapes JSON')
    parser.add_argument('--output-dir', type=str,
                       default='dataset',
                       help='Output directory for dataset')
    parser.add_argument('--num-train', type=int, default=500,
                       help='Number of training images')
    parser.add_argument('--num-test', type=int, default=125,
                       help='Number of test images')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size (square)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Get project directory
    PROJECT_DIR = Path(__file__).parent.parent

    shapes_json = PROJECT_DIR / args.shapes_json
    output_dir = PROJECT_DIR / args.output_dir

    generate_dataset(
        shapes_json=str(shapes_json),
        output_dir=str(output_dir),
        num_train=args.num_train,
        num_test=args.num_test,
        img_size=args.img_size,
        seed=args.seed
    )
