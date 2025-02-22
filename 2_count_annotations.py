import os
from pathlib import Path
from collections import defaultdict

# Class mapping (same as in conversion script)
CLASS_MAPPING = {
    "D00": 0,  # Longitudinal Crack
    "D10": 1,  # Transverse Crack
    "D20": 2,  # Alligator Crack
    "D40": 3,  # Pothole
}

# Reverse mapping for display
REV_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}


def count_annotations(processed_dir="data/processed"):
    """Count annotations for each class in the processed dataset"""
    labels_dir = Path(processed_dir) / "labels"

    # Initialize counters
    class_counts = defaultdict(int)
    total_files = 0

    # Count annotations in each file
    for label_file in labels_dir.glob("*.txt"):
        total_files += 1
        with open(label_file, "r") as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1

    # Print results
    print("\nAnnotation Count Summary:")
    print("-" * 50)
    print(f"Total files processed: {total_files}")
    print("\nClass-wise distribution:")
    print("-" * 50)

    total_annotations = 0
    for class_id in sorted(class_counts.keys()):
        class_name = REV_CLASS_MAPPING[class_id]
        count = class_counts[class_id]
        total_annotations += count
        print(f"Class {class_name} (ID: {class_id}): {count:,} annotations")

    print("-" * 50)
    print(f"Total annotations: {total_annotations:,}")

    # Calculate and print percentages
    print("\nClass distribution percentages:")
    print("-" * 50)
    for class_id in sorted(class_counts.keys()):
        class_name = REV_CLASS_MAPPING[class_id]
        percentage = (class_counts[class_id] / total_annotations) * 100
        print(f"Class {class_name}: {percentage:.2f}%")


if __name__ == "__main__":
    count_annotations()
