#!/usr/bin/env python3
"""
Test script to generate 20 line background patterns for exploration.

Generates striped backgrounds with 4 orientations (45°, -45°, vertical, horizontal)
and 5 variations each, plus shapes on top using overlap logic.
"""

import cv2
import numpy as np
import json
import random
from pathlib import Path

# Configuration
COLORS = [255, 213, 170, 128]  # White to medium grey
ORIENTATIONS = ['45deg', 'neg45deg', 'vertical', 'horizontal']
IMG_SIZE = 640
OUTPUT_DIR = Path(__file__).parent / "test_line_outputs"

# Test configurations: (orientation, line_width, spacing, colors, offset, description)
TEST_CONFIGS = [
    # 45-degree (5 variations)
    ('45deg', 2, 10, COLORS, 0, 'thin_dense'),
    ('45deg', 5, 20, COLORS, 0, 'medium'),
    ('45deg', 10, 40, COLORS, 0, 'thick_sparse'),
    ('45deg', 5, 20, COLORS[:2], 0, '2color_cycle'),
    ('45deg', 5, 20, COLORS, 15, 'random_offset'),

    # -45-degree (5 variations)
    ('neg45deg', 2, 10, COLORS, 0, 'thin_dense'),
    ('neg45deg', 5, 20, COLORS, 0, 'medium'),
    ('neg45deg', 10, 40, COLORS, 0, 'thick_sparse'),
    ('neg45deg', 5, 20, COLORS[:2], 0, '2color_cycle'),
    ('neg45deg', 5, 20, COLORS, 15, 'random_offset'),

    # Vertical (5 variations)
    ('vertical', 2, 10, COLORS, 0, 'thin_dense'),
    ('vertical', 5, 20, COLORS, 0, 'medium'),
    ('vertical', 10, 40, COLORS, 0, 'thick_sparse'),
    ('vertical', 5, 20, COLORS[:2], 0, '2color_cycle'),
    ('vertical', 5, 20, COLORS, 15, 'random_offset'),

    # Horizontal (5 variations)
    ('horizontal', 2, 10, COLORS, 0, 'thin_dense'),
    ('horizontal', 5, 20, COLORS, 0, 'medium'),
    ('horizontal', 10, 40, COLORS, 0, 'thick_sparse'),
    ('horizontal', 5, 20, COLORS[:2], 0, '2color_cycle'),
    ('horizontal', 5, 20, COLORS, 15, 'random_offset'),
]


def create_line_background(orientation, line_width, spacing, colors, offset=0):
    """
    Generate striped line background.

    Args:
        orientation: '45deg', 'neg45deg', 'vertical', or 'horizontal'
        line_width: Width of each line in pixels
        spacing: Gap between lines in pixels
        colors: List of greyscale values to cycle through
        offset: Starting offset for line placement

    Returns:
        Greyscale canvas (IMG_SIZE x IMG_SIZE)
    """
    canvas = np.full((IMG_SIZE, IMG_SIZE), 128, dtype=np.uint8)

    if orientation == 'vertical':
        # Vertical lines
        x = offset
        color_idx = 0
        while x < IMG_SIZE:
            x_end = min(x + line_width, IMG_SIZE)
            canvas[:, x:x_end] = colors[color_idx % len(colors)]
            x += line_width + spacing
            color_idx += 1

    elif orientation == 'horizontal':
        # Horizontal lines
        y = offset
        color_idx = 0
        while y < IMG_SIZE:
            y_end = min(y + line_width, IMG_SIZE)
            canvas[y:y_end, :] = colors[color_idx % len(colors)]
            y += line_width + spacing
            color_idx += 1

    elif orientation == '45deg':
        # 45-degree diagonal (top-left to bottom-right)
        # Draw lines parallel to y = x
        # Line equation: y = x + c, where c ranges from -IMG_SIZE to IMG_SIZE

        # Start from top-left corner, sweep to bottom-right
        c_start = -IMG_SIZE + offset
        c = c_start
        color_idx = 0

        while c < IMG_SIZE:
            color = colors[color_idx % len(colors)]

            # Draw a diagonal stripe from (x1, y1) to (x2, y2)
            # For each line, we need to draw a band of width `line_width`
            for w in range(line_width):
                c_current = c + w

                # Find intersection points with image boundaries
                # Line: y = x + c_current
                # Boundaries: x ∈ [0, IMG_SIZE], y ∈ [0, IMG_SIZE]

                # Start point
                if c_current >= 0:
                    x1, y1 = 0, c_current  # Intersects left edge
                else:
                    x1, y1 = -c_current, 0  # Intersects top edge

                # End point
                if c_current <= 0:
                    x2, y2 = IMG_SIZE, IMG_SIZE + c_current  # Intersects bottom edge
                else:
                    x2, y2 = IMG_SIZE - c_current, IMG_SIZE  # Intersects right edge

                # Clip to image bounds
                if x1 >= IMG_SIZE or y1 >= IMG_SIZE or x2 < 0 or y2 < 0:
                    continue

                x1 = max(0, min(IMG_SIZE - 1, int(x1)))
                y1 = max(0, min(IMG_SIZE - 1, int(y1)))
                x2 = max(0, min(IMG_SIZE - 1, int(x2)))
                y2 = max(0, min(IMG_SIZE - 1, int(y2)))

                # Draw line
                cv2.line(canvas, (x1, y1), (x2, y2), int(color), 1)

            c += line_width + spacing
            color_idx += 1

    elif orientation == 'neg45deg':
        # -45-degree diagonal (top-right to bottom-left)
        # Line equation: y = -x + c, where c ranges from 0 to 2*IMG_SIZE

        c_start = offset
        c = c_start
        color_idx = 0

        while c < 2 * IMG_SIZE:
            color = colors[color_idx % len(colors)]

            # Draw a diagonal stripe
            for w in range(line_width):
                c_current = c + w

                # Find intersection points
                # Line: y = -x + c_current

                # Start point
                if c_current <= IMG_SIZE:
                    x1, y1 = 0, c_current  # Intersects left edge
                else:
                    x1, y1 = c_current - IMG_SIZE, IMG_SIZE  # Intersects bottom edge

                # End point
                if c_current >= IMG_SIZE:
                    x2, y2 = IMG_SIZE, c_current - IMG_SIZE  # Intersects right edge
                else:
                    x2, y2 = c_current, 0  # Intersects top edge

                # Clip to bounds
                if x1 >= IMG_SIZE or y1 < 0 or x2 < 0 or y2 >= IMG_SIZE:
                    continue

                x1 = max(0, min(IMG_SIZE - 1, int(x1)))
                y1 = max(0, min(IMG_SIZE - 1, int(y1)))
                x2 = max(0, min(IMG_SIZE - 1, int(x2)))
                y2 = max(0, min(IMG_SIZE - 1, int(y2)))

                # Draw line
                cv2.line(canvas, (x1, y1), (x2, y2), int(color), 1)

            c += line_width + spacing
            color_idx += 1

    return canvas


def load_sample_shapes():
    """Load a few sample shapes from each category for overlay."""
    shapes_root = Path(__file__).parent.parent / "Shapes"

    sample_shapes = {}
    for cat_dir in shapes_root.iterdir():
        if cat_dir.is_dir() and cat_dir.name.startswith('cat') and cat_dir.name != 'cat3':
            cat_name = cat_dir.name
            shape_files = list(cat_dir.glob("*.png"))[:5]  # Take first 5 shapes
            sample_shapes[cat_name] = [str(f) for f in shape_files]

    return sample_shapes


def paste_shape_simple(canvas, shape_path, scale_range=(0.1, 0.3), max_attempts=50):
    """
    Paste a single shape onto canvas at random position.

    Simplified version without overlap tracking - just for visualization.
    """
    # Load shape
    shape_mask = cv2.imread(str(shape_path), cv2.IMREAD_GRAYSCALE)
    if shape_mask is None:
        return False, None

    # Apply scale
    scale = random.uniform(*scale_range)
    shape_h, shape_w = shape_mask.shape
    new_w = int(shape_w * scale)
    new_h = int(shape_h * scale)

    if new_w < 10 or new_h < 10:
        return False, None

    resized = cv2.resize(shape_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Random position
    for _ in range(max_attempts):
        paste_x = random.randint(0, IMG_SIZE - new_w)
        paste_y = random.randint(0, IMG_SIZE - new_h)

        # Paste shape
        shape_region = canvas[paste_y:paste_y + new_h, paste_x:paste_x + new_w]
        mask_region = (resized > 0).astype(np.uint8)

        # Blend shape onto background
        canvas[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = \
            np.where(mask_region, resized, shape_region)

        # Calculate YOLO bbox
        center_x = (paste_x + new_w / 2) / IMG_SIZE
        center_y = (paste_y + new_h / 2) / IMG_SIZE
        width = new_w / IMG_SIZE
        height = new_h / IMG_SIZE

        return True, (center_x, center_y, width, height)

    return False, None


def generate_test_images():
    """Generate all 20 test images with line backgrounds and shapes."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Load sample shapes
    sample_shapes = load_sample_shapes()
    all_shapes = []
    for shapes_list in sample_shapes.values():
        all_shapes.extend(shapes_list)

    if not all_shapes:
        print("Warning: No shapes found. Generating backgrounds only.")

    print(f"Generating 20 test images to {OUTPUT_DIR}/")
    print(f"Loaded {len(all_shapes)} sample shapes")
    print()

    metadata = []

    for i, (orient, width, spacing, colors, offset, desc) in enumerate(TEST_CONFIGS):
        # Create background
        bg = create_line_background(orient, width, spacing, colors, offset)

        # Add shapes (2-4 shapes per image)
        canvas = bg.copy()
        num_shapes = random.randint(2, 4)
        bboxes = []

        for _ in range(num_shapes):
            if not all_shapes:
                break
            shape_path = random.choice(all_shapes)
            success, bbox = paste_shape_simple(canvas, shape_path)
            if success:
                bboxes.append(bbox)

        # Save image
        img_name = f"test_{i:02d}_{orient}_{desc}.jpg"
        img_path = OUTPUT_DIR / img_name
        cv2.imwrite(str(img_path), canvas)

        # Save metadata
        metadata.append({
            'index': i,
            'filename': img_name,
            'orientation': orient,
            'line_width': width,
            'spacing': spacing,
            'colors': colors,
            'offset': offset,
            'description': desc,
            'num_shapes': len(bboxes)
        })

        print(f"✓ {img_name} ({orient}, {desc}, {len(bboxes)} shapes)")

    # Save metadata
    metadata_path = OUTPUT_DIR / "test_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print()
    print(f"✓ Generated {len(TEST_CONFIGS)} test images")
    print(f"✓ Saved to {OUTPUT_DIR}/")
    print(f"✓ Metadata: {metadata_path}")
    print()
    print("Next steps:")
    print("  1. Review images to select best patterns")
    print("  2. Decide on line width/spacing for full experiment")
    print("  3. Implement line_background_with_overlap experiment")


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    generate_test_images()
