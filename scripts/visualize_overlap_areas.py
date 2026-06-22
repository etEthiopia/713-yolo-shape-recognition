#!/usr/bin/env python3
"""
Visualize Intersection and Union Areas

Shows the geometric overlap between shapes using Shapely:
- Union area: Total area covered by all shapes combined
- Intersection area: Area where 2+ shapes overlap

Usage:
    python visualize_overlap_areas.py experiments/solid_background_with_overlap_splitted/detailed_evaluation test_0053
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from shapely.geometry import box
from shapely.ops import unary_union
from matplotlib.collections import PatchCollection

# Illustrator compatibility
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Category colors
CATEGORY_COLORS_RGB = {
    "cat1": (0/255, 170/255, 220/255),
    "cat2": (0/255, 83/255, 155/255),
    "cat4": (127/255, 47/255, 141/255),
    "cat5": (176/255, 31/255, 35/255)
}

CATEGORY_NAMES = {
    "cat1": "Unconstrained",
    "cat2": "Local: Variance",
    "cat4": "Local: Match All",
    "cat5": "Natural"
}


def calculate_shape_metrics(ground_truth_list):
    """
    Compute True Global Union, Shared Intersection (>=2 boxes), and Overlap Ratio.

    Args:
        ground_truth_list: List of dicts with 'bbox' key (YOLO normalized format)

    Returns:
        Dict with intersection_area, true_global_union_area, overlap_ratio, and polygons
    """
    polygons = []

    # Step 1: Convert all YOLO format bboxes to Shapely Polygons
    for item in ground_truth_list:
        bbox = item["bbox"]
        x_center = bbox["x_center"]
        y_center = bbox["y_center"]
        w = bbox["width"]
        h = bbox["height"]

        # Calculate standard corners
        x1 = x_center - (w / 2)
        x2 = x_center + (w / 2)
        y1 = y_center + (h / 2)  # Bottom (larger Y)
        y2 = y_center - (h / 2)  # Top (smaller Y)

        polygons.append(box(x1, y2, x2, y1))

    if not polygons:
        return {
            "intersection_area": 0.0,
            "true_global_union_area": 0.0,
            "overlap_ratio": 0.0,
            "polygons": [],
            "union_poly": None,
            "intersection_poly": None
        }

    # Step 2: Compute True Global Union Area
    global_union_poly = unary_union(polygons)
    true_global_union_area = global_union_poly.area

    # Step 3: Calculate sum of all individual areas
    sum_of_areas = sum(poly.area for poly in polygons)

    # Step 4: Compute Shared Intersection Area using pairwise intersections (CORRECT METHOD)
    # This finds the actual physical area where 2+ boxes overlap
    pairwise_intersections = []
    num_boxes = len(polygons)

    for i in range(num_boxes):
        for j in range(i + 1, num_boxes):
            inter = polygons[i].intersection(polygons[j])
            if not inter.is_empty:
                pairwise_intersections.append(inter)

    if pairwise_intersections:
        # Flatten overlapping regions into a single unique footprint
        intersection_poly = unary_union(pairwise_intersections)
        intersection_area = intersection_poly.area
        shared_area = intersection_area
    else:
        intersection_poly = None
        intersection_area = 0.0
        shared_area = 0.0

    # Step 5: Calculate Normalized Overlap Ratio
    # Overlap ratio = shared / union
    # "What fraction of the footprint has overlapping shapes?"
    if true_global_union_area > 0:
        overlap_ratio = shared_area / true_global_union_area
    else:
        overlap_ratio = 0.0

    # Report Stats
    return {
        "intersection_area": round(intersection_area, 6),
        "shared_area": round(shared_area, 6),  # Same as intersection_area
        "sum_of_areas": round(sum_of_areas, 6),
        "true_global_union_area": round(true_global_union_area, 6),
        "overlap_ratio": round(overlap_ratio, 4),
        "polygons": polygons,
        "union_poly": global_union_poly,
        "intersection_poly": intersection_poly
    }


def shapely_polygon_to_matplotlib_patch(poly, facecolor, edgecolor, alpha, linewidth=2, label=None):
    """
    Convert a Shapely polygon to matplotlib patches.
    Handles both Polygon and MultiPolygon.
    """
    from shapely.geometry import Polygon, MultiPolygon
    from matplotlib.patches import Polygon as MPLPolygon

    patches_list = []

    if isinstance(poly, Polygon):
        coords = list(poly.exterior.coords)
        patch = MPLPolygon(coords, closed=True, facecolor=facecolor,
                          edgecolor=edgecolor, alpha=alpha, linewidth=linewidth,
                          label=label)
        patches_list.append(patch)
    elif isinstance(poly, MultiPolygon):
        for p in poly.geoms:
            coords = list(p.exterior.coords)
            patch = MPLPolygon(coords, closed=True, facecolor=facecolor,
                             edgecolor=edgecolor, alpha=alpha, linewidth=linewidth,
                             label=label if not patches_list else None)
            patches_list.append(patch)

    return patches_list


def visualize_overlap_areas(image_path: Path, image_result: dict, output_path: Path):
    """
    Create visualization showing union and intersection areas.

    Args:
        image_path: Path to the test image
        image_result: Per-image result dict from evaluation
        output_path: Where to save the visualization
    """
    # Load image
    img = Image.open(image_path)
    if img.mode == 'L':
        img = img.convert('RGB')
    img_width, img_height = img.size

    # Get ground truth
    ground_truth = image_result['ground_truth']

    # Calculate metrics
    metrics = calculate_shape_metrics(ground_truth)

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

    # --- Subplot 1: Original Image with Bounding Boxes ---
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Ground Truth Bounding Boxes', fontsize=18, fontweight='bold', pad=15)
    ax1.axis('off')

    for i, gt in enumerate(ground_truth):
        bbox = gt['bbox']
        category = gt['category']

        # Convert to pixel coordinates
        x_center = bbox['x_center'] * img_width
        y_center = bbox['y_center'] * img_height
        width = bbox['width'] * img_width
        height = bbox['height'] * img_height
        x = x_center - width / 2
        y = y_center - height / 2

        color = CATEGORY_COLORS_RGB.get(category, (0.5, 0.5, 0.5))

        rect = patches.Rectangle(
            (x, y), width, height,
            linewidth=3,
            edgecolor=color,
            facecolor='none',
            linestyle='-'
        )
        ax1.add_patch(rect)

    # --- Subplot 2: Union Area ---
    ax2.imshow(img, cmap='gray')
    ax2.set_title(f'Total Stimuli Area\n(Total Coverage: {metrics["true_global_union_area"]:.4f})',
                 fontsize=18, fontweight='bold', pad=15)
    ax2.axis('off')

    if metrics['union_poly'] is not None:
        # Draw union polygon
        union_patches = shapely_polygon_to_matplotlib_patch(
            metrics['union_poly'],
            facecolor='blue',
            edgecolor='darkblue',
            alpha=0.4,
            linewidth=3,
            label='Union Area'
        )
        for patch in union_patches:
            # Scale to pixel coordinates
            xy = patch.get_xy()
            xy_scaled = xy * [img_width, img_height]
            patch.set_xy(xy_scaled)
            ax2.add_patch(patch)

    # --- Subplot 3: Intersection Area ---
    ax3.imshow(img, cmap='gray')
    overlap_pct = metrics["overlap_ratio"] * 100
    ax3.set_title(f'Intersection Area (Overlap)\n(Shared: {metrics["shared_area"]:.4f} = Sum({metrics["sum_of_areas"]:.4f}) - Union({metrics["true_global_union_area"]:.4f}), Ratio: {overlap_pct:.1f}%)',
                 fontsize=16, fontweight='bold', pad=15)
    ax3.axis('off')

    if metrics['intersection_poly'] is not None:
        # Draw intersection polygon
        intersection_patches = shapely_polygon_to_matplotlib_patch(
            metrics['intersection_poly'],
            facecolor='red',
            edgecolor='darkred',
            alpha=0.5,
            linewidth=3,
            label='Intersection Area'
        )
        for patch in intersection_patches:
            # Scale to pixel coordinates
            xy = patch.get_xy()
            xy_scaled = xy * [img_width, img_height]
            patch.set_xy(xy_scaled)
            ax3.add_patch(patch)
    else:
        # No intersection
        ax3.text(0.5, 0.5, 'No Overlap\n(Shapes do not intersect)',
                transform=ax3.transAxes,
                fontsize=20, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))

    # Add category legend
    from matplotlib.patches import Patch
    category_legend = [
        Patch(facecolor=CATEGORY_COLORS_RGB['cat1'], edgecolor='black', label='Unconstrained'),
        Patch(facecolor=CATEGORY_COLORS_RGB['cat2'], edgecolor='black', label='Local: Variance'),
        Patch(facecolor=CATEGORY_COLORS_RGB['cat4'], edgecolor='black', label='Local: Match All'),
        Patch(facecolor=CATEGORY_COLORS_RGB['cat5'], edgecolor='black', label='Natural')
    ]

    fig.legend(handles=category_legend, loc='lower center', ncol=4, fontsize=12,
              frameon=True, title='Categories', title_fontsize=13,
              bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize intersection and union areas',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('eval_dir', type=str,
                       help='Path to detailed evaluation directory')
    parser.add_argument('image_name', type=str,
                       help='Image name (e.g., test_0053, test_0001)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Custom output directory')

    args = parser.parse_args()

    # Paths
    eval_dir = Path(args.eval_dir)

    if not eval_dir.exists():
        print(f"Error: Evaluation directory does not exist: {eval_dir}")
        sys.exit(1)

    per_image_file = eval_dir / 'per_image_results.json'

    if not per_image_file.exists():
        print(f"Error: Per-image results not found: {per_image_file}")
        sys.exit(1)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = eval_dir.parent / 'visualizations'

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find image in dataset
    experiment_dir = eval_dir.parent

    # Determine if train or test
    if 'train_' in args.image_name:
        split = 'train'
    else:
        split = 'test'

    image_name_jpg = args.image_name if args.image_name.endswith('.jpg') else f"{args.image_name}.jpg"
    image_path = experiment_dir / 'dataset' / 'images' / split / image_name_jpg

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    print("=" * 70)
    print("Visualize Intersection and Union Areas")
    print("=" * 70)
    print(f"\nImage: {image_name_jpg}")
    print(f"Output: {output_dir}")
    print()

    # Load per-image results
    print("Loading evaluation results...")
    with open(per_image_file, 'r') as f:
        per_image_data = json.load(f)

    # Find the image result
    image_result = None
    for result in per_image_data:
        if image_name_jpg in result['image_path']:
            image_result = result
            break

    if image_result is None:
        print(f"Error: No evaluation results found for {image_name_jpg}")
        sys.exit(1)

    print(f"  ✓ Found evaluation results")
    print()

    # Calculate and display metrics
    ground_truth = image_result['ground_truth']
    metrics = calculate_shape_metrics(ground_truth)

    print("Overlap Metrics:")
    print(f"  Ground truth shapes: {len(ground_truth)}")
    print(f"  Sum of areas: {metrics['sum_of_areas']:.6f}")
    print(f"  Union area (footprint): {metrics['true_global_union_area']:.6f}")
    print(f"  Shared area (overlap): {metrics['shared_area']:.6f}")
    print(f"  Overlap ratio (shared/union): {metrics['overlap_ratio']:.1%}")
    print(f"    → {metrics['overlap_ratio']*100:.1f}% of footprint has stacked shapes")
    print()

    # Create visualization
    output_path = output_dir / f"{args.image_name}_overlap_areas.jpg"
    print("Creating visualization...")
    visualize_overlap_areas(image_path, image_result, output_path)
    print()

    print(f"Done! Visualization saved to: {output_path}")
    print()


if __name__ == '__main__':
    main()
