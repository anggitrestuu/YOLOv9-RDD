import xml.etree.ElementTree as ET
from collections import defaultdict
import random
import shutil
from pathlib import Path

DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"

prefix_name = "D00_D10_D20"

# Class mapping
CLASS_MAPPING = {
    "D00": 0,  # Longitudinal Crack
    "D10": 1,  # Transverse Crack
    "D20": 2,  # Alligator Crack
    # "D40": 3,  # Pothole
}

# Dataset balancing configuration
ENABLE_BALANCING = False  # Set to False to skip balancing
BALANCE_MODE = "min"  # Can be "min" or "max"   
MIN_ANNOTATIONS_PER_CLASS = 100
MAX_ANNOTATIONS_PER_CLASS = 400  # Add this new constant


def parse_xml(xml_path):
    """Parse XML file and return normalized bounding boxes with class names"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = float(root.find("size/width").text)
    height = float(root.find("size/height").text)

    boxes = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in CLASS_MAPPING:
            continue

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Convert to YOLO format (normalized coordinates)
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        boxes.append(
            (CLASS_MAPPING[class_name], x_center, y_center, box_width, box_height)
        )

    return boxes


def collect_annotations():
    """Collect all annotations and their distribution"""
    annotations = defaultdict(list)

    for dataset in ["China_Motorbike", "China_Drone"]:
    # for dataset in ["United_States/train",]:
        xml_dir = Path(DATA_RAW_DIR) / dataset / "annotations" / "xmls"
        img_dir = Path(DATA_RAW_DIR) / dataset / "images"

        for xml_file in xml_dir.glob("*.xml"):
            boxes = parse_xml(xml_file)
            if boxes:  # Only include if valid boxes are found
                img_file = img_dir / f"{xml_file.stem}.jpg"
                if img_file.exists():
                    # Store each box separately with its image info
                    for box in boxes:
                        class_idx = box[0]
                        annotations[class_idx].append((xml_file, img_file, [box]))

    return annotations


def print_class_distribution(annotations):
    """Print the distribution of annotations across classes"""
    print("\nInitial Class Distribution:")
    print("-" * 50)
    total = 0
    class_counts = {}
    
    for class_idx, anns in annotations.items():
        count = len(anns)
        class_counts[class_idx] = count
        total += count
        
    for class_idx, count in class_counts.items():
        class_name = [k for k, v in CLASS_MAPPING.items() if v == class_idx][0]
        percentage = (count / total) * 100
        print(f"Class {class_name} (ID: {class_idx}): {count:,} samples ({percentage:.2f}%)")
    
    print("-" * 50)
    return class_counts


def balance_dataset(annotations):
    """Balance dataset to ensure minimum or maximum annotations per class"""
    balanced_data = []
    
    # Print initial distribution
    class_counts = print_class_distribution(annotations)
    
    # Determine target count based on balance mode
    if BALANCE_MODE == "min":
        target_count = max(MIN_ANNOTATIONS_PER_CLASS, min(class_counts.values()))
        comparison_str = "only"
    else:  # max mode
        target_count = min(MAX_ANNOTATIONS_PER_CLASS, max(class_counts.values()))
        comparison_str = "already"

    print(f"\nBalancing dataset with {BALANCE_MODE.upper()} mode (target: {target_count} samples)")
    print("-" * 50)

    # Ensure target annotations per class
    final_counts = {}
    for class_idx in CLASS_MAPPING.values():
        class_annotations = annotations[class_idx]
        initial_count = len(class_annotations)
        
        # Randomly sample or duplicate to reach target
        if BALANCE_MODE == "min":
            if initial_count < target_count:
                print(f"Class {class_idx}: Duplicating {initial_count} → {target_count} samples")
            selected = random.choices(class_annotations, k=target_count)
        else:
            if initial_count > target_count:
                print(f"Class {class_idx}: Reducing {initial_count} → {target_count} samples")
            elif initial_count < target_count:
                print(f"Warning: Class {class_idx} has {comparison_str} {initial_count} samples")
            selected = random.sample(class_annotations, k=min(initial_count, target_count))
        
        final_counts[class_idx] = len(selected)
        balanced_data.extend(selected)

    # Print final distribution
    print("\nFinal Class Distribution:")
    print("-" * 50)
    total = sum(final_counts.values())
    for class_idx, count in final_counts.items():
        class_name = [k for k, v in CLASS_MAPPING.items() if v == class_idx][0]
        percentage = (count / total) * 100
        print(f"Class {class_name} (ID: {class_idx}): {count:,} samples ({percentage:.2f}%)")
    print("-" * 50)

    return balanced_data


def save_yolo_format(output_dir, balanced_data):
    """Save balanced dataset in YOLO format"""
    output_dir = Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Keep track of images we've already processed
    processed_images = set()

    # Group annotations by image
    image_annotations = defaultdict(list)
    for xml_path, img_path, boxes in balanced_data:
        key = (xml_path, img_path)
        image_annotations[key].extend(boxes)

    # Save grouped annotations
    for (xml_path, img_path), boxes in image_annotations.items():
        # Copy image (only once)
        dest_img_name = f"{prefix_name}_{img_path.name}"
        if dest_img_name not in processed_images:
            shutil.copy2(img_path, output_dir / "images" / dest_img_name)
            processed_images.add(dest_img_name)

        # Save YOLO format labels
        label_path = output_dir / "labels" / f"{prefix_name}_{xml_path.stem}.txt"
        with open(label_path, "w") as f:
            for box in boxes:
                f.write(
                    f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n"
                )


def main():
    # remove data/processed
    shutil.rmtree(DATA_PROCESSED_DIR, ignore_errors=True)

    # Collect all annotations
    print("Collecting annotations...")
    annotations = collect_annotations()

    # Process the dataset with or without balancing
    data_to_save = None
    
    if ENABLE_BALANCING:
        print("\nBalancing dataset...")
        data_to_save = balance_dataset(annotations)
    else:
        print("\nSkipping dataset balancing...")
        # Just combine all annotations without balancing
        data_to_save = []
        for class_anns in annotations.values():
            data_to_save.extend(class_anns)
        # Still print the distribution for information
        print_class_distribution(annotations)

    # Save in YOLO format
    print("\nSaving in YOLO format...")
    save_yolo_format(DATA_PROCESSED_DIR, data_to_save)

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
