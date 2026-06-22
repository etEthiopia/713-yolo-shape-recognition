#!/usr/bin/env python3
"""
Generate composite images with controlled overlap for line_background_with_overlap_splitted experiment.

50% of images have chain overlaps with 25-75% visibility constraint.
Overlap metadata tracked in JSON files.
"""
import cv2
import numpy as np
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse

# Determine paths
script_dir = Path(__file__).resolve().parent
experiment_dir = script_dir.parent
project_root = experiment_dir.parent.parent

# Class mapping (sequential, not formula!)
CLASS_MAPPING = {
    'cat1': 0,
    'cat2': 1,
    'cat4': 2,  # NOT 3!
    'cat5': 3   # NOT 4!
}


def load_shape_image(path: str, project_root: Path) -> np.ndarray:
    """
    Load shape image from relative path and extract binary mask.

    Args:
        path: Relative path like "Shapes/cat1/img201.png"
        project_root: Project root directory

    Returns:
        Binary mask (255 = foreground, 0 = background)
    """
    abs_path = project_root / path

    if not abs_path.exists():
        raise FileNotFoundError(f"Shape image not found: {abs_path}")

    # Load image
    img = cv2.imread(str(abs_path), cv2.IMREAD_UNCHANGED)

    # Extract alpha channel
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
    """Rotate image by angle (degrees) with expanded canvas."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new canvas size
    cos_val = np.abs(M[0, 0])
    sin_val = np.abs(M[0, 1])
    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))

    # Adjust rotation matrix for new center
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Rotate
    rotated = cv2.warpAffine(image, M, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderValue=0)

    return rotated


def get_tight_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Get tight bounding box around non-zero pixels."""
    coords = cv2.findNonZero(mask)
    if coords is None:
        return 0, 0, 0, 0
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h


def calculate_visible_area(shape_mask: np.ndarray,
                           overlap_mask: np.ndarray,
                           paste_x: int,
                           paste_y: int) -> Tuple[int, int, float]:
    """
    Calculate visible area of a shape given current overlaps.

    Args:
        shape_mask: Binary mask of the shape to place
        overlap_mask: Accumulated mask of all placed shapes
        paste_x, paste_y: Position where shape will be pasted

    Returns:
        (total_area, visible_area, visibility_ratio)
    """
    h, w = shape_mask.shape
    total_area = np.sum(shape_mask > 0)

    if total_area == 0:
        return 0, 0, 0.0

    # Extract the region where this shape will be placed
    overlap_h, overlap_w = overlap_mask.shape
    end_y = min(paste_y + h, overlap_h)
    end_x = min(paste_x + w, overlap_w)

    # Calculate actual dimensions considering boundaries
    actual_h = end_y - paste_y
    actual_w = end_x - paste_x

    if actual_h <= 0 or actual_w <= 0:
        return total_area, 0, 0.0

    # Get the overlap region
    overlap_region = overlap_mask[paste_y:end_y, paste_x:end_x]
    shape_region = shape_mask[:actual_h, :actual_w]

    # Calculate occluded area
    occluded = np.sum((shape_region > 0) & (overlap_region > 0))
    visible_area = total_area - occluded
    visibility_ratio = visible_area / total_area if total_area > 0 else 0.0

    return total_area, visible_area, visibility_ratio


def bbox_to_yolo_format(bbox: Tuple[int, int, int, int],
                        img_width: int,
                        img_height: int) -> Tuple[float, float, float, float]:
    """Convert bbox to YOLO format (normalized center coords + size)."""
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return x_center, y_center, width, height


def paste_shape_with_overlap(canvas: np.ndarray,
                              shape_mask: np.ndarray,
                              scale: float,
                              rotation: float,
                              existing_boxes: List,
                              overlap_mask: np.ndarray,
                              target_visibility: Tuple[float, float] = (0.25, 0.75),
                              max_attempts: int = 100) -> Tuple[bool, Optional[Tuple], Dict]:
    """
    Paste shape with controlled overlap (25-75% visibility for occluded shapes).

    Returns:
        (success, bbox, overlap_metadata)
    """
    canvas_h, canvas_w = canvas.shape
    shape_h, shape_w = shape_mask.shape

    # Transform shape (matching baseline resize logic)
    # 1. Resize - shape-relative (not canvas-relative)
    new_w = int(shape_w * scale)
    new_h = int(shape_h * scale)

    if new_w < 10 or new_h < 10:
        return False, None, {}

    resized = cv2.resize(shape_mask, (new_w, new_h),
                         interpolation=cv2.INTER_LINEAR)

    # 2. Rotate
    rotated = rotate_image(resized, rotation)

    # 3. Get tight bbox
    x, y, w, h = get_tight_bbox(rotated)
    if w == 0 or h == 0:
        return False, None, {}

    cropped_mask = rotated[y:y+h, x:x+w]

    # Try random positions
    min_vis, max_vis = target_visibility

    for attempt in range(max_attempts):
        # Random position
        paste_x = random.randint(0, max(0, canvas_w - w))
        paste_y = random.randint(0, max(0, canvas_h - h))

        # Calculate visibility at this position
        total_area, visible_area, visibility_ratio = calculate_visible_area(
            cropped_mask, overlap_mask, paste_x, paste_y
        )

        # Check if visibility is in target range
        if min_vis <= visibility_ratio <= max_vis:
            # Valid placement!
            # Paste shape onto canvas
            for i in range(h):
                for j in range(w):
                    if cropped_mask[i, j] > 127:
                        cy = paste_y + i
                        cx = paste_x + j
                        if 0 <= cy < canvas_h and 0 <= cx < canvas_w:
                            canvas[cy, cx] = 255
                            overlap_mask[cy, cx] = 255  # Update overlap mask

            # Calculate bbox
            bbox = (paste_x, paste_y, w, h)

            # Determine which shapes this overlaps
            overlaps_with = []
            for idx, (ex, ey, ew, eh) in enumerate(existing_boxes):
                # Check if bboxes intersect
                if not (paste_x + w < ex or ex + ew < paste_x or
                        paste_y + h < ey or ey + eh < paste_y):
                    overlaps_with.append(idx)

            # Metadata
            metadata = {
                'total_area': int(total_area),
                'visible_area': int(visible_area),
                'occluded_area': int(total_area - visible_area),
                'visibility_ratio': float(visibility_ratio),
                'overlaps_with': overlaps_with,
                'overlap_type': 'partial' if overlaps_with else 'none'
            }

            return True, bbox, metadata

    # Failed to find valid placement
    return False, None, {}


def paste_shape_no_overlap(canvas: np.ndarray,
                            shape_mask: np.ndarray,
                            scale: float,
                            rotation: float,
                            existing_boxes: List,
                            overlap_mask: np.ndarray,
                            min_separation: int = 15,
                            max_attempts: int = 50) -> Tuple[bool, Optional[Tuple], Dict]:
    """Paste shape with NO overlap (original logic from solid_background)."""
    canvas_h, canvas_w = canvas.shape
    shape_h, shape_w = shape_mask.shape

    # Transform shape (matching baseline resize logic)
    new_w = int(shape_w * scale)
    new_h = int(shape_h * scale)

    if new_w < 10 or new_h < 10:
        return False, None, {}

    resized = cv2.resize(shape_mask, (new_w, new_h),
                         interpolation=cv2.INTER_LINEAR)
    rotated = rotate_image(resized, rotation)
    x, y, w, h = get_tight_bbox(rotated)

    if w == 0 or h == 0:
        return False, None, {}

    cropped_mask = rotated[y:y+h, x:x+w]

    # Try random positions (no overlap allowed)
    for attempt in range(max_attempts):
        paste_x = random.randint(0, max(0, canvas_w - w))
        paste_y = random.randint(0, max(0, canvas_h - h))

        candidate_box = (paste_x, paste_y, w, h)

        # Check overlap with existing boxes
        overlaps = False
        for (ex, ey, ew, eh) in existing_boxes:
            # Expand by separation margin
            ex_min = ex - min_separation
            ey_min = ey - min_separation
            ex_max = ex + ew + min_separation
            ey_max = ey + eh + min_separation

            px_min = paste_x
            py_min = paste_y
            px_max = paste_x + w
            py_max = paste_y + h

            if not (px_max < ex_min or ex_max < px_min or
                    py_max < ey_min or ey_max < py_min):
                overlaps = True
                break

        if not overlaps:
            # Paste shape
            for i in range(h):
                for j in range(w):
                    if cropped_mask[i, j] > 127:
                        cy = paste_y + i
                        cx = paste_x + j
                        if 0 <= cy < canvas_h and 0 <= cx < canvas_w:
                            canvas[cy, cx] = 255
                            overlap_mask[cy, cx] = 255

            metadata = {
                'total_area': int(np.sum(cropped_mask > 0)),
                'visible_area': int(np.sum(cropped_mask > 0)),
                'occluded_area': 0,
                'visibility_ratio': 1.0,
                'overlaps_with': [],
                'overlap_type': 'none'
            }

            return True, candidate_box, metadata

    return False, None, {}


def create_line_background(img_size: int = 640) -> np.ndarray:
    """
    Generate striped line background with random pattern.

    Args:
        img_size: Canvas size (square)

    Returns:
        Greyscale canvas with line pattern
    """
    COLORS = [255, 213, 170, 128]  # White to medium grey
    ORIENTATIONS = ['45deg', 'neg45deg', 'vertical', 'horizontal']

    # Randomize parameters per image
    orientation = random.choice(ORIENTATIONS)
    line_width = random.choice([2, 5, 10])
    spacing = random.choice([10, 20, 40])
    colors = COLORS if random.random() < 0.75 else COLORS[:2]  # 75% 4-color, 25% 2-color
    offset = random.randint(0, 20)

    # Initialize canvas with base grey
    canvas = np.full((img_size, img_size), 128, dtype=np.uint8)

    if orientation == 'vertical':
        # Vertical lines
        x = offset
        color_idx = 0
        while x < img_size:
            x_end = min(x + line_width, img_size)
            canvas[:, x:x_end] = colors[color_idx % len(colors)]
            x += line_width + spacing
            color_idx += 1

    elif orientation == 'horizontal':
        # Horizontal lines
        y = offset
        color_idx = 0
        while y < img_size:
            y_end = min(y + line_width, img_size)
            canvas[y:y_end, :] = colors[color_idx % len(colors)]
            y += line_width + spacing
            color_idx += 1

    elif orientation == '45deg':
        # 45-degree diagonal (top-left to bottom-right)
        c_start = -img_size + offset
        c = c_start
        color_idx = 0

        while c < img_size:
            color = colors[color_idx % len(colors)]

            for w in range(line_width):
                c_current = c + w

                # Find intersection points with boundaries
                if c_current >= 0:
                    x1, y1 = 0, c_current
                else:
                    x1, y1 = -c_current, 0

                if c_current <= 0:
                    x2, y2 = img_size, img_size + c_current
                else:
                    x2, y2 = img_size - c_current, img_size

                # Clip to bounds
                if x1 >= img_size or y1 >= img_size or x2 < 0 or y2 < 0:
                    continue

                x1 = max(0, min(img_size - 1, int(x1)))
                y1 = max(0, min(img_size - 1, int(y1)))
                x2 = max(0, min(img_size - 1, int(x2)))
                y2 = max(0, min(img_size - 1, int(y2)))

                cv2.line(canvas, (x1, y1), (x2, y2), int(color), 1)

            c += line_width + spacing
            color_idx += 1

    elif orientation == 'neg45deg':
        # -45-degree diagonal (top-right to bottom-left)
        c_start = offset
        c = c_start
        color_idx = 0

        while c < 2 * img_size:
            color = colors[color_idx % len(colors)]

            for w in range(line_width):
                c_current = c + w

                # Find intersection points
                if c_current <= img_size:
                    x1, y1 = 0, c_current
                else:
                    x1, y1 = c_current - img_size, img_size

                if c_current >= img_size:
                    x2, y2 = img_size, c_current - img_size
                else:
                    x2, y2 = c_current, 0

                # Clip to bounds
                if x1 >= img_size or y1 < 0 or x2 < 0 or y2 >= img_size:
                    continue

                x1 = max(0, min(img_size - 1, int(x1)))
                y1 = max(0, min(img_size - 1, int(y1)))
                x2 = max(0, min(img_size - 1, int(x2)))
                y2 = max(0, min(img_size - 1, int(y2)))

                cv2.line(canvas, (x1, y1), (x2, y2), int(color), 1)

            c += line_width + spacing
            color_idx += 1

    return canvas


def generate_image_with_overlap(shapes: List[Tuple[np.ndarray, str, int]],
                                 num_shapes: int,
                                 img_size: int,
                                 overlap_probability: float = 0.5) -> Tuple[np.ndarray, List, List[Dict]]:
    """
    Generate image with chain overlaps.

    Args:
        shapes: List of (mask, category, class_id)
        num_shapes: Number of shapes to place
        img_size: Canvas size
        overlap_probability: Probability each shape (after first) overlaps

    Returns:
        (canvas, labels, overlap_metadata)
    """
    canvas = create_line_background(img_size)
    overlap_mask = np.zeros((img_size, img_size), dtype=np.uint8)

    labels = []
    metadata_list = []
    existing_boxes = []

    for idx, (shape_mask, category, class_id) in enumerate(shapes[:num_shapes]):
        # Random augmentation (same as baseline for fair comparison)
        scale = np.random.uniform(0.1, 0.3)
        rotation = np.random.uniform(0, 360)

        if idx == 0:
            # First shape: no overlap possible
            success, bbox, metadata = paste_shape_no_overlap(
                canvas, shape_mask, scale, rotation, existing_boxes, overlap_mask
            )
        else:
            # Subsequent shapes: chance of overlap
            if np.random.random() < overlap_probability:
                success, bbox, metadata = paste_shape_with_overlap(
                    canvas, shape_mask, scale, rotation, existing_boxes, overlap_mask,
                    target_visibility=(0.25, 0.75)
                )

                # If overlap placement fails, try no-overlap
                if not success:
                    success, bbox, metadata = paste_shape_no_overlap(
                        canvas, shape_mask, scale, rotation, existing_boxes, overlap_mask
                    )
            else:
                success, bbox, metadata = paste_shape_no_overlap(
                    canvas, shape_mask, scale, rotation, existing_boxes, overlap_mask
                )

        if success:
            x_c, y_c, w, h = bbox_to_yolo_format(bbox, img_size, img_size)
            labels.append((class_id, x_c, y_c, w, h))

            metadata['shape_idx'] = len(existing_boxes)
            metadata['class_id'] = class_id
            metadata['category'] = category
            metadata_list.append(metadata)

            existing_boxes.append(bbox)

    return canvas, labels, metadata_list


def save_overlap_metadata(overlap_dir: Path,
                          image_name: str,
                          metadata_list: List[Dict],
                          labels: List[Tuple]) -> None:
    """Save overlap statistics to JSON file."""
    json_path = overlap_dir / f"{image_name.replace('.jpg', '.json')}"

    # Calculate statistics
    num_shapes = len(metadata_list)
    num_overlapping = sum(1 for m in metadata_list if m['overlap_type'] != 'none')
    total_overlap_area = sum(m['occluded_area'] for m in metadata_list)
    avg_visibility = np.mean([m['visibility_ratio'] for m in metadata_list]) if metadata_list else 0.0

    data = {
        "image": image_name,
        "num_shapes": num_shapes,
        "num_overlapping_shapes": num_overlapping,
        "shapes": metadata_list,
        "statistics": {
            "total_overlap_area": int(total_overlap_area),
            "avg_visibility_ratio": float(avg_visibility),
            "overlap_percentage": (num_overlapping / num_shapes * 100) if num_shapes > 0 else 0.0
        }
    }

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


def generate_dataset(shapes_json: str,
                     output_dir: str,
                     num_train: int = 1000,
                     num_test: int = 250,
                     img_size: int = 640,
                     overlap_percentage: float = 0.5,
                     seed: int = 42):
    """Generate complete dataset with 50% overlap images."""
    print("=" * 70)
    print("Generating Composite Images with Overlap")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Training images: {num_train}")
    print(f"  Test images: {num_test}")
    print(f"  Overlap percentage: {overlap_percentage * 100:.0f}%")
    print(f"  Visibility constraint: 25-75% for occluded shapes")
    print(f"  Random seed: {seed}")
    print()

    random.seed(seed)
    np.random.seed(seed)

    # Load selected shapes with train/test split
    print("Loading shapes...")
    with open(shapes_json) as f:
        selected_shapes = json.load(f)

    # Create separate train and test pools
    train_shapes = []
    test_shapes = []

    for category, split_dict in selected_shapes['shapes'].items():
        class_id = CLASS_MAPPING[category]

        # Load train shapes
        for path in split_dict['train']:
            mask = load_shape_image(path, project_root)
            train_shapes.append((mask, category, class_id))

        # Load test shapes
        for path in split_dict['test']:
            mask = load_shape_image(path, project_root)
            test_shapes.append((mask, category, class_id))

    print(f"  Train pool: {len(train_shapes)} shapes")
    print(f"  Test pool: {len(test_shapes)} shapes")
    print(f"  Shape separation: STRICT (zero overlap)")
    print()

    # Prepare output directories
    output_path = Path(output_dir)
    train_img_dir = output_path / 'images' / 'train'
    train_lbl_dir = output_path / 'labels' / 'train'
    train_overlap_dir = output_path / 'overlap' / 'train'
    test_img_dir = output_path / 'images' / 'test'
    test_lbl_dir = output_path / 'labels' / 'test'
    test_overlap_dir = output_path / 'overlap' / 'test'

    # Determine which images have overlap
    num_train_overlap = int(num_train * overlap_percentage)
    num_test_overlap = int(num_test * overlap_percentage)

    train_overlap_indices = set(random.sample(range(num_train), num_train_overlap))
    test_overlap_indices = set(random.sample(range(num_test), num_test_overlap))

    print(f"Overlap distribution:")
    print(f"  Train: {num_train_overlap}/{num_train} images with overlap")
    print(f"  Test: {num_test_overlap}/{num_test} images with overlap")
    print()

    # Generate training images
    print(f"Generating training images...")
    for i in range(num_train):
        # Sample shapes FROM TRAIN POOL ONLY
        num_shapes = random.randint(2, 7)
        sampled_shapes = random.sample(train_shapes, num_shapes)

        # Generate image
        has_overlap = i in train_overlap_indices
        canvas, labels, metadata = generate_image_with_overlap(
            sampled_shapes, num_shapes, img_size,
            overlap_probability=0.5 if has_overlap else 0.0
        )

        # Save image
        img_name = f"train_{i:04d}.jpg"
        cv2.imwrite(str(train_img_dir / img_name), canvas)

        # Save labels
        lbl_name = f"train_{i:04d}.txt"
        with open(train_lbl_dir / lbl_name, 'w') as f:
            for class_id, x_c, y_c, w, h in labels:
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

        # Save overlap metadata
        save_overlap_metadata(train_overlap_dir, img_name, metadata, labels)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{num_train} images generated")

    print(f"  ✓ Training set complete: {num_train} images")
    print()

    # Generate test images
    print(f"Generating test images...")
    for i in range(num_test):
        num_shapes = random.randint(2, 7)
        sampled_shapes = random.sample(test_shapes, num_shapes)  # TEST POOL ONLY

        has_overlap = i in test_overlap_indices
        canvas, labels, metadata = generate_image_with_overlap(
            sampled_shapes, num_shapes, img_size,
            overlap_probability=0.5 if has_overlap else 0.0
        )

        img_name = f"test_{i:04d}.jpg"
        cv2.imwrite(str(test_img_dir / img_name), canvas)

        lbl_name = f"test_{i:04d}.txt"
        with open(test_lbl_dir / lbl_name, 'w') as f:
            for class_id, x_c, y_c, w, h in labels:
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

        save_overlap_metadata(test_overlap_dir, img_name, metadata, labels)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{num_test} images generated")

    print(f"  ✓ Test set complete: {num_test} images")
    print()

    print("=" * 70)
    print("Dataset Generation Complete!")
    print("=" * 70)
    print(f"\nOutput directories:")
    print(f"  Images: {output_path / 'images'}")
    print(f"  Labels: {output_path / 'labels'}")
    print(f"  Overlap metadata: {output_path / 'overlap'}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Generate composite images with controlled overlap'
    )
    parser.add_argument('--shapes-json', type=str,
                       default=str(experiment_dir / 'dataset' / 'selected_shapes.json'),
                       help='Selected shapes JSON file')
    parser.add_argument('--output-dir', type=str,
                       default=str(experiment_dir / 'dataset'),
                       help='Output directory')
    parser.add_argument('--num-train', type=int, default=1000,
                       help='Number of training images')
    parser.add_argument('--num-test', type=int, default=250,
                       help='Number of test images')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size (square)')
    parser.add_argument('--overlap-percentage', type=float, default=0.5,
                       help='Percentage of images with overlap (0.0-1.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    generate_dataset(
        shapes_json=args.shapes_json,
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_test=args.num_test,
        img_size=args.img_size,
        overlap_percentage=args.overlap_percentage,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
