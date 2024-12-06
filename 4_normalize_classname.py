import os

DIR_DATASET = "YOLOv9"

DIR_DATASET_D20 = f"{DIR_DATASET}/china_d20/train/labels"  # TODO : convert .txt from 0 0.315625 0.4703125 0.5640625 0.17578125 to 2 0.315625 0.4703125 0.5640625 0.17578125
DIR_DATASET_D40 = f"{DIR_DATASET}/china_d40/train/labels"  # TODO : convert .txt from 0 0.4890625 0.35625 0.134375 0.38125 to 3 0.4890625 0.35625 0.134375 0.38125


def convert_classname_in_txt(directory, old_class, new_class):
    """Convert class index in YOLO format .txt files"""
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as file:
                lines = file.readlines()

            with open(file_path, "w") as file:
                for line in lines:
                    parts = line.strip().split()
                    if parts[0] == str(old_class):
                        parts[0] = str(new_class)
                    file.write(" ".join(parts) + "\n")


def main():
    # Convert class index for D20
    convert_classname_in_txt(DIR_DATASET_D20, 0, 2)
    # Convert class index for D40
    convert_classname_in_txt(DIR_DATASET_D40, 0, 3)


if __name__ == "__main__":
    main()
