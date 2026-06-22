#!/usr/bin/env python3
"""
Solid Background Experiment - Generate Composite Images

CRITICAL: Uses correct sequential class ID mapping to avoid cat4/cat5 bug!

Categories: cat1(0), cat2(1), cat4(2), cat5(3)
"""
import json
import random
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple

# CRITICAL: Correct class ID mapping (sequential, not formula-based!)
CLASS_MAPPING = {
    'cat1': 0,
    'cat2': 1,
    'cat4': 2,  # Class 2, NOT 3!
    'cat5': 3   # Class 3, NOT 4!
}

def load_selected_shapes(json_path: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Load balanced shape paths from JSON with train/test split.

    Returns:
        Tuple of (train_shapes, test_shapes)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    shapes = data['shapes']

    # Check if it's split format
    first_category = list(shapes.keys())[0]
    if isinstance(shapes[first_category], dict) and 'train' in shapes[first_category]:
        # Split format: {"cat1": {"train": [...], "test": [...]}}
        train_shapes = {cat: shapes[cat]['train'] for cat in shapes.keys()}
        test_shapes = {cat: shapes[cat]['test'] for cat in shapes.keys()}
        return train_shapes, test_shapes
    else:
        # Old format: {"cat1": [...]} - use same for both
        return shapes, shapes

def load_shape_image(path: str, project_root: Path) -> np.ndarray:
    """Load PNG and extract binary mask from alpha channel.

    Args:
        path: Relative path from project root (e.g., "Shapes/cat1/img001.png")
        project_root: Absolute path to project root
    """
    # Convert relative path to absolute
    abs_path = project_root / path

    img = cv2.imread(str(abs_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load: {path} (resolved to {abs_path})")

    # Extract alpha channel
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        alpha = np.where(gray > 127, 255, 0).astype(np.uint8)

    _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    return mask

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by angle (degrees)."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(image, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)
    return rotated

def get_tight_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Get tight bounding box around non-zero pixels."""
    coords = cv2.findNonZero(mask)
    if coords is None:
        return 0, 0, 0, 0
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h

def check_overlap(bbox1: Tuple[int, int, int, int],
                  bbox2: Tuple[int, int, int, int],
                  min_separation: int = 10) -> bool:
    """Check if two bounding boxes overlap."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x1_min, y1_min = x1 - min_separation, y1 - min_separation
    x1_max, y1_max = x1 + w1 + min_separation, y1 + h1 + min_separation

    x2_min, y2_min = x2 - min_separation, y2 - min_separation
    x2_max, y2_max = x2 + w2 + min_separation, y2 + h2 + min_separation

    return not (x1_max < x2_min or x2_max < x1_min or
                y1_max < y2_min or y2_max < y1_min)

def paste_shape_augmented(canvas: np.ndarray,
                          shape_mask: np.ndarray,
                          scale: float,
                          rotation: float,
                          existing_boxes: List,
                          max_attempts: int = 50) -> Tuple[bool, Tuple]:
    """Paste shape with random position (non-overlapping)."""
    canvas_h, canvas_w = canvas.shape[:2]

    # Resize
    shape_h, shape_w = shape_mask.shape
    new_w = int(shape_w * scale)
    new_h = int(shape_h * scale)

    if new_w < 10 or new_h < 10:
        return False, None

    resized = cv2.resize(shape_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Rotate
    rotated = rotate_image(resized, rotation)

    # Get tight bbox
    x, y, w, h = get_tight_bbox(rotated)
    if w == 0 or h == 0:
        return False, None

    cropped_mask = rotated[y:y+h, x:x+w]

    # Try random positions
    for _ in range(max_attempts):
        paste_x = random.randint(0, max(1, canvas_w - w))
        paste_y = random.randint(0, max(1, canvas_h - h))

        candidate_box = (paste_x, paste_y, w, h)

        overlaps = any(check_overlap(candidate_box, existing, min_separation=15)
                      for existing in existing_boxes)

        if not overlaps:
            # Paste shape
            for i in range(h):
                for j in range(w):
                    if cropped_mask[i, j] > 127:
                        canvas_y = paste_y + i
                        canvas_x = paste_x + j
                        if 0 <= canvas_y < canvas_h and 0 <= canvas_x < canvas_w:
                            canvas[canvas_y, canvas_x] = 255

            return True, candidate_box

    return False, None

def bbox_to_yolo_format(bbox: Tuple[int, int, int, int],
                        img_width: int,
                        img_height: int) -> Tuple[float, float, float, float]:
    """Convert bbox to YOLO format (normalized)."""
    x, y, w, h = bbox

    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height

    return x_center, y_center, width, height

def generate_composite_image(shapes_dict: Dict[str, List[str]],
                             project_root: Path,
                             img_size: int = 640,
                             shapes_range: Tuple[int, int] = (2, 7)) -> Tuple[np.ndarray, List]:
    """Generate one composite image with multiple shapes.

    Args:
        shapes_dict: Dict of category -> list of relative shape paths
        project_root: Absolute path to project root
        img_size: Image size in pixels
        shapes_range: (min, max) shapes per image
    """
    # Grey background
    canvas = np.full((img_size, img_size), 128, dtype=np.uint8)

    num_shapes = random.randint(*shapes_range)

    labels = []
    existing_boxes = []

    for _ in range(num_shapes):
        # Randomly pick category
        category = random.choice(list(shapes_dict.keys()))

        # CRITICAL: Use mapping, NOT formula!
        class_id = CLASS_MAPPING[category]

        # Pick shape (relative path)
        shape_path = random.choice(shapes_dict[category])

        try:
            shape_mask = load_shape_image(shape_path, project_root)

            scale = random.uniform(0.1, 0.3)
            rotation = random.uniform(0, 360)

            success, bbox = paste_shape_augmented(
                canvas, shape_mask, scale, rotation, existing_boxes
            )

            if success:
                existing_boxes.append(bbox)
                x_c, y_c, w, h = bbox_to_yolo_format(bbox, img_size, img_size)
                labels.append((class_id, x_c, y_c, w, h))

        except Exception as e:
            print(f"Warning: Failed to process {shape_path}: {e}")
            continue

    return canvas, labels

def generate_dataset(shapes_json: str,
                     output_dir: str,
                     num_train: int = 1000,
                     num_test: int = 250,
                     img_size: int = 640,
                     shapes_range: Tuple[int, int] = (2, 7),
                     seed: int = 42):
    """Generate complete YOLO dataset."""
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("Generating YOLO Composite Images Dataset (Splitted)")
    print("=" * 70)

    print(f"\nLoading shape paths from: {shapes_json}")
    train_shapes, test_shapes = load_selected_shapes(shapes_json)

    # Get project root (4 levels up from script: scripts -> solid_background_splitted -> experiments -> project_root)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent.parent

    total_train = sum(len(paths) for paths in train_shapes.values())
    total_test = sum(len(paths) for paths in test_shapes.values())
    print(f"Loaded {len(train_shapes)} categories")
    print(f"  Train shapes: {total_train}")
    print(f"  Test shapes: {total_test}")
    print(f"Project root: {project_root}")
    print(f"Class ID mapping: {CLASS_MAPPING}")
    print(f"\nSample train paths:")
    for cat in list(train_shapes.keys())[:2]:
        print(f"  {cat}: {train_shapes[cat][0]}")
    print(f"\nSample test paths:")
    for cat in list(test_shapes.keys())[:2]:
        print(f"  {cat}: {test_shapes[cat][0]}")

    # Create output directories
    output_path = Path(output_dir)
    for split in ['train', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Track class distribution
    class_distribution = {i: 0 for i in range(4)}  # Classes 0-3

    # Generate training set (using train_shapes)
    print(f"\nGenerating {num_train} training images (shapes per image: {shapes_range[0]}-{shapes_range[1]})...")
    for i in range(num_train):
        img, labels = generate_composite_image(train_shapes, project_root, img_size, shapes_range)

        img_path = output_path / 'images' / 'train' / f'train_{i:04d}.jpg'
        cv2.imwrite(str(img_path), img)

        label_path = output_path / 'labels' / 'train' / f'train_{i:04d}.txt'
        with open(label_path, 'w') as f:
            for class_id, x_c, y_c, w, h in labels:
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                class_distribution[class_id] += 1

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{num_train}")

    print(f"✓ Training set complete: {num_train} images")

    # Generate test set (using test_shapes - completely different shapes!)
    print(f"\nGenerating {num_test} test images (USING DIFFERENT SHAPES)...")
    for i in range(num_test):
        img, labels = generate_composite_image(test_shapes, project_root, img_size, shapes_range)

        img_path = output_path / 'images' / 'test' / f'test_{i:04d}.jpg'
        cv2.imwrite(str(img_path), img)

        label_path = output_path / 'labels' / 'test' / f'test_{i:04d}.txt'
        with open(label_path, 'w') as f:
            for class_id, x_c, y_c, w, h in labels:
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                class_distribution[class_id] += 1

        if (i + 1) % 50 == 0:
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

    print(f"\nClass ID distribution (labels across all images):")
    CATEGORIES = ['cat1', 'cat2', 'cat4', 'cat5']
    total_instances = sum(class_distribution.values())
    for class_id, count in sorted(class_distribution.items()):
        cat_name = CATEGORIES[class_id]
        percentage = (count / total_instances * 100) if total_instances > 0 else 0
        print(f"  Class {class_id} ({cat_name}): {count} instances ({percentage:.1f}%)")

    print(f"\nTotal shape instances: {total_instances}")
    print(f"\n✓ All labels use class IDs 0-3 only (no class 4!)")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate YOLO composite images')
    parser.add_argument('--shapes-json', type=str,
                       default='../dataset/selected_shapes.json',
                       help='Path to selected shapes JSON')
    parser.add_argument('--output-dir', type=str,
                       default='../dataset',
                       help='Output directory')
    parser.add_argument('--num-train', type=int, default=1000)
    parser.add_argument('--num-test', type=int, default=250)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--shapes-range', type=int, nargs=2, default=[2, 7])
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Convert relative paths to absolute
    script_dir = Path(__file__).parent
    shapes_json = (script_dir / args.shapes_json).resolve()
    output_dir = (script_dir / args.output_dir).resolve()

    generate_dataset(
        shapes_json=str(shapes_json),
        output_dir=str(output_dir),
        num_train=args.num_train,
        num_test=args.num_test,
        img_size=args.img_size,
        shapes_range=tuple(args.shapes_range),
        seed=args.seed
    )
