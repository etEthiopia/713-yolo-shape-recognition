#!/usr/bin/env python3
"""
Verify shape separation between train and test sets.
Ensures zero overlap - critical for valid evaluation.
"""
import json
from pathlib import Path

def verify_shape_split():
    """Verify no shapes appear in both train and test sets."""
    dataset_json = Path(__file__).parent.parent / 'dataset' / 'selected_shapes.json'

    with open(dataset_json) as f:
        data = json.load(f)

    print("Verifying shape separation...\n")

    all_passed = True
    for category in ['cat1', 'cat2', 'cat4', 'cat5']:
        train_set = set(data['shapes'][category]['train'])
        test_set = set(data['shapes'][category]['test'])

        overlap = train_set & test_set

        print(f"{category}:")
        print(f"  Train: {len(train_set)} shapes")
        print(f"  Test: {len(test_set)} shapes")
        print(f"  Overlap: {len(overlap)} shapes")

        if len(overlap) > 0:
            print(f"  ❌ FAILED: {len(overlap)} shapes in both train and test!")
            all_passed = False
        else:
            print(f"  ✅ PASSED: Zero overlap")
        print()

    if all_passed:
        print("✅ ALL CATEGORIES PASSED: Strict train/test separation verified")
    else:
        print("❌ VERIFICATION FAILED: Shape leakage detected!")
        exit(1)

if __name__ == '__main__':
    verify_shape_split()
