import shutil

DATA_PROCESSED_DIR = "data/augmented"

CHINA_D20 = "https://app.roboflow.com/ds/H0jUKXLPrn?key=Br1har67T7"
CHINA_D40 = "https://app.roboflow.com/ds/Xqt6jIWGVT?key=0P0QVGgDZo"

# TODO : download dataset with curl and show progress bar

import subprocess


def download_dataset(url, output_path):
    """Download dataset using curl with a progress bar"""
    try:
        subprocess.run(
            ["curl", "-L", url, "-o", output_path, "--progress-bar"], check=True
        )
        print(f"Downloaded dataset to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download dataset: {e}")


def main():
    # Example usage
    download_dataset(CHINA_D20, "china_d20.zip")
    download_dataset(CHINA_D40, "china_d40.zip")


if __name__ == "__main__":
    main()
