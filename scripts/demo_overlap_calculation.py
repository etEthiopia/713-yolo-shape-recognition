#!/usr/bin/env python3
"""
Demonstration: Sum, Union, and Shared Area

Shows why "sum - union" is WRONG for calculating shared/overlapping area.

Usage:
    python demo_overlap_calculation.py
"""

from shapely.geometry import box
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

def demo_overlap_calculation():
    """Demonstrate the difference between correct and incorrect methods."""

    print("=" * 70)
    print("Overlap Calculation Demonstration")
    print("=" * 70)
    print()

    # Create 3 overlapping boxes
    # Box A: (0,0) to (2,2) = 4 sq units
    # Box B: (1,0) to (3,2) = 4 sq units
    # Box C: (1,0.5) to (3,2.5) = 4 sq units
    box_a = box(0, 0, 2, 2)
    box_b = box(1, 0, 3, 2)
    box_c = box(1, 0.5, 3, 2.5)

    boxes = [box_a, box_b, box_c]

    print("Three Overlapping Boxes:")
    print(f"  Box A: area = {box_a.area:.2f} sq units")
    print(f"  Box B: area = {box_b.area:.2f} sq units")
    print(f"  Box C: area = {box_c.area:.2f} sq units")
    print()

    # Calculate Sum
    sum_of_areas = sum(b.area for b in boxes)
    print(f"1. SUM of Areas:")
    print(f"   = {box_a.area} + {box_b.area} + {box_c.area}")
    print(f"   = {sum_of_areas:.2f} sq units")
    print(f"   (Counts overlaps multiple times!)")
    print()

    # Calculate Union
    union_poly = unary_union(boxes)
    union_area = union_poly.area
    print(f"2. UNION Area (Total Footprint):")
    print(f"   = {union_area:.2f} sq units")
    print(f"   (Each pixel counted once)")
    print()

    # Method 1: Sum - Union (WRONG)
    shared_wrong = sum_of_areas - union_area
    print(f"3a. SHARED Area - Method 1: Sum - Union (WRONG)")
    print(f"    = {sum_of_areas:.2f} - {union_area:.2f}")
    print(f"    = {shared_wrong:.2f} sq units")
    print(f"    ❌ Counts triple-overlap regions MULTIPLE times!")
    print()

    # Method 2: Pairwise Intersections (CORRECT)
    print(f"3b. SHARED Area - Method 2: Pairwise Intersections (CORRECT)")

    # Find all pairwise intersections
    ab_inter = box_a.intersection(box_b)
    ac_inter = box_a.intersection(box_c)
    bc_inter = box_b.intersection(box_c)

    print(f"    A ∩ B = {ab_inter.area:.2f} sq units")
    print(f"    A ∩ C = {ac_inter.area:.2f} sq units")
    print(f"    B ∩ C = {bc_inter.area:.2f} sq units")
    print()

    # CRITICAL: What if we just sum these?
    naive_sum = ab_inter.area + ac_inter.area + bc_inter.area
    print(f"    If we naively sum: {ab_inter.area:.2f} + {ac_inter.area:.2f} + {bc_inter.area:.2f}")
    print(f"                     = {naive_sum:.2f} sq units")
    print(f"    ❌ This TRIPLE-COUNTS the region where all 3 boxes overlap!")
    print()

    # Union the pairwise intersections (so triple-overlap counted once)
    pairwise = [ab_inter, ac_inter, bc_inter]
    shared_correct = unary_union(pairwise).area

    print(f"    Union(A∩B, A∩C, B∩C) = {shared_correct:.2f} sq units")
    print(f"    ✓ Each overlapping pixel counted ONCE")
    print(f"    ✓ The triple-overlap region appears in all 3 pairwise intersections,")
    print(f"      but unary_union removes the duplicates!")
    print()

    # Show the difference
    print("=" * 70)
    print("Comparison:")
    print("=" * 70)
    print(f"  Sum:           {sum_of_areas:.2f} sq units")
    print(f"  Union:         {union_area:.2f} sq units")
    print(f"  Shared (WRONG): {shared_wrong:.2f} sq units  ❌")
    print(f"  Shared (RIGHT): {shared_correct:.2f} sq units  ✓")
    print()
    print(f"  Difference:    {abs(shared_wrong - shared_correct):.2f} sq units")
    print(f"  Overestimate:  {(shared_wrong / shared_correct - 1) * 100:.1f}%")
    print()

    # Calculate overlap ratios (shared/union)
    overlap_wrong = shared_wrong / union_area * 100
    overlap_correct = shared_correct / union_area * 100

    print("Overlap Ratios (shared/union):")
    print(f"  Wrong method:  {overlap_wrong:.1f}%")
    print(f"  Correct method: {overlap_correct:.1f}%")
    print(f"  ")
    print(f"  Interpretation: {overlap_correct:.1f}% of the footprint has stacked shapes")
    print()

    print("=" * 70)
    print("Detailed Breakdown: Triple-Overlap Handling")
    print("=" * 70)
    print()

    # Find the triple overlap region
    triple_overlap = box_a.intersection(box_b).intersection(box_c)
    print(f"Triple-overlap region (A ∩ B ∩ C):")
    print(f"  Area = {triple_overlap.area:.2f} sq units")
    print()

    print("This triple-overlap region appears in ALL 3 pairwise intersections:")
    print(f"  - A ∩ B contains the triple-overlap: {triple_overlap.within(ab_inter)}")
    print(f"  - A ∩ C contains the triple-overlap: {triple_overlap.within(ac_inter)}")
    print(f"  - B ∩ C contains the triple-overlap: {triple_overlap.within(bc_inter)}")
    print()

    print("What happens when we process them:")
    print()
    print("❌ If we SUM the pairwise intersections:")
    print(f"   = {ab_inter.area:.2f} + {ac_inter.area:.2f} + {bc_inter.area:.2f}")
    print(f"   = {naive_sum:.2f} sq units")
    print(f"   Triple-overlap is counted 3 times!")
    print()

    print("✓ If we UNION the pairwise intersections:")
    print(f"   = Union(A∩B, A∩C, B∩C)")
    print(f"   = {shared_correct:.2f} sq units")
    print(f"   Triple-overlap is counted 1 time!")
    print()

    print("The magic:")
    print(f"  Naive sum - Correct union = {naive_sum:.2f} - {shared_correct:.2f} = {naive_sum - shared_correct:.2f}")
    print(f"  This difference = 2 × (triple-overlap area)")
    print(f"                  = 2 × {triple_overlap.area:.2f} = {2 * triple_overlap.area:.2f}")
    print(f"  Because the triple-overlap was in the naive sum 3 times,")
    print(f"  but should only be counted once!")
    print()
    print("=" * 70)


if __name__ == '__main__':
    demo_overlap_calculation()
