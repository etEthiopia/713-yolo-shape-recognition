#!/usr/bin/env python3
"""
Test script to verify intersection-based overlap calculation.
"""

# Simulate the calculation functions
def calculate_intersection_area(box1, box2):
    """Calculate intersection area between two boxes."""
    x1_min = box1['x_center'] - box1['width'] / 2
    y1_min = box1['y_center'] - box1['height'] / 2
    x1_max = box1['x_center'] + box1['width'] / 2
    y1_max = box1['y_center'] + box1['height'] / 2

    x2_min = box2['x_center'] - box2['width'] / 2
    y2_min = box2['y_center'] - box2['height'] / 2
    x2_max = box2['x_center'] + box2['width'] / 2
    y2_max = box2['y_center'] + box2['height'] / 2

    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    if x_inter_max <= x_inter_min or y_inter_max <= y_inter_min:
        return 0.0

    intersection = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    return intersection


def calculate_shape_overlap_ratios(shapes):
    """Calculate overlap ratio for each shape."""
    n = len(shapes)
    overlap_ratios = []

    for i in range(n):
        shape = shapes[i]
        shape_area = shape['bbox']['width'] * shape['bbox']['height']

        if shape_area == 0:
            overlap_ratios.append(0.0)
            continue

        total_intersection = 0.0

        for j in range(n):
            if i == j:
                continue

            other_shape = shapes[j]
            intersection = calculate_intersection_area(shape['bbox'], other_shape['bbox'])
            total_intersection += intersection

        overlap_ratio = total_intersection / shape_area
        overlap_ratios.append(overlap_ratio)

    return overlap_ratios


# Test case from test_0000.jpg
shapes = [
    {
        'bbox': {
            'x_center': 0.757031,
            'y_center': 0.615625,
            'width': 0.329688,
            'height': 0.346875
        }
    },
    {
        'bbox': {
            'x_center': 0.721094,
            'y_center': 0.628125,
            'width': 0.473438,
            'height': 0.35625
        }
    }
]

print("Test: Two overlapping shapes from test_0000.jpg")
print("=" * 60)

# Calculate areas
shape1_area = shapes[0]['bbox']['width'] * shapes[0]['bbox']['height']
shape2_area = shapes[1]['bbox']['width'] * shapes[1]['bbox']['height']

print(f"\nShape 1 area: {shape1_area:.6f}")
print(f"Shape 2 area: {shape2_area:.6f}")

# Calculate intersection
intersection = calculate_intersection_area(shapes[0]['bbox'], shapes[1]['bbox'])
print(f"\nIntersection area: {intersection:.6f}")

# Calculate overlap ratios
overlap_ratios = calculate_shape_overlap_ratios(shapes)

print(f"\nShape 1 overlap_ratio: {overlap_ratios[0]:.6f}")
print(f"  (intersection {intersection:.6f} / shape1_area {shape1_area:.6f})")

print(f"\nShape 2 overlap_ratio: {overlap_ratios[1]:.6f}")
print(f"  (intersection {intersection:.6f} / shape2_area {shape2_area:.6f})")

print("\n" + "=" * 60)
print("Expected behavior:")
print("  - Both shapes should have overlap_ratio > 0")
print("  - Smaller shape has higher overlap_ratio")
print("  - Larger shape has lower overlap_ratio")
print("  - Both ratios use the SAME intersection area")
print()

# Verify
if overlap_ratios[0] > 0 and overlap_ratios[1] > 0:
    print("✅ PASS: Both shapes show intersection")
else:
    print("❌ FAIL: One or both shapes show zero intersection")

if overlap_ratios[0] > overlap_ratios[1]:
    print("✅ PASS: Smaller shape has higher overlap ratio")
else:
    print("❌ FAIL: Overlap ratios are incorrect")
