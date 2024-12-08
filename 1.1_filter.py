import xml.etree.ElementTree as ET
from collections import defaultdict
import random
import shutil
from pathlib import Path

DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed2"

prefix_name = "D00_D10_D20"

# Class mapping
CLASS_MAPPING = {
    "D00": 0,  # Longitudinal Crack
    "D10": 1,  # Transverse Crack
    "D20": 2,  # Alligator Crack
    # "D40": 3,  # Pothole
}

BALANCE_MODE = "max"  # Can be "min" or "max"
MIN_ANNOTATIONS_PER_CLASS = 1000
MAX_ANNOTATIONS_PER_CLASS = 1500 


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
        xml_dir = Path(DATA_RAW_DIR) / dataset / "annotations" / "xmls"
        img_dir = Path(DATA_RAW_DIR) / dataset / "images"

        for xml_file in xml_dir.glob("*.xml"):
            boxes = parse_xml(xml_file)
            if boxes:  # Only include if valid boxes are found
                img_file = img_dir / f"{xml_file.stem}.jpg"
                if img_file.exists():
                    for box in boxes:
                        class_idx = box[0]
                        annotations[class_idx].append((xml_file, img_file, boxes))

    return annotations


def balance_dataset(annotations):
    """Balance dataset to ensure minimum or maximum annotations per class"""
    balanced_data = []
    
    # Determine target count based on balance mode
    if BALANCE_MODE == "min":
        target_count = MIN_ANNOTATIONS_PER_CLASS
        comparison_str = "only"
    else:  # max mode
        target_count = MAX_ANNOTATIONS_PER_CLASS
        comparison_str = "already"

    # Ensure target annotations per class
    for class_idx in CLASS_MAPPING.values():
        class_annotations = annotations[class_idx]
        if len(class_annotations) < target_count:
            print(
                f"Warning: Class {class_idx} has {comparison_str} {len(class_annotations)} annotations"
            )

        # Randomly sample or duplicate to reach target
        if BALANCE_MODE == "min":
            # For minimum mode, duplicate if needed
            selected = random.choices(class_annotations, k=target_count)
        else:
            # For maximum mode, cap at maximum
            selected = random.sample(class_annotations, k=min(len(class_annotations), target_count))
            
        balanced_data.extend(selected)

    return balanced_data



def save_yolo_format(output_dir, balanced_data):
    """Save balanced dataset in YOLO format"""
    output_dir = Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    for xml_path, img_path, boxes in balanced_data:
        # Copy image
        shutil.copy2(img_path, output_dir / "images" / f"{prefix_name}_{img_path.name}")

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

    # Balance dataset
    print("Balancing dataset...")
    balanced_data = balance_dataset(annotations)

    # Save in YOLO format
    print("Saving in YOLO format...")
    save_yolo_format(DATA_PROCESSED_DIR, balanced_data)

    print("Conversion complete!")


if __name__ == "__main__":
    main()