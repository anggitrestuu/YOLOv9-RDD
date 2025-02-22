import shutil

DATA_PROCESSED_DIR = "data/augmented"

D40 = "https://app.roboflow.com/ds/NfQAnyZJiC?key=HENqBzciJu"

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
    download_dataset(D40, "D40.zip")


if __name__ == "__main__":
    main()
