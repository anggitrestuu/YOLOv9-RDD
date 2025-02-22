import os

DIR_DATASET = "D40"

DIR_DATASET_D40 = f"{DIR_DATASET}/train/labels"


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
    # convert_classname_in_txt(DIR_DATASET_D20, 0, 2)
    # Convert class index for D40
    convert_classname_in_txt(DIR_DATASET_D40, 0, 3)


if __name__ == "__main__":
    main()
