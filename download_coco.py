# download_coco.py
# Use this script to download the COCO dataset reliably with progress bars.

import wget
import os
import time

# --- Configuration ---
DATA_DIR = 'coco_dataset'
os.makedirs(DATA_DIR, exist_ok=True)
os.chdir(DATA_DIR)

# List of URLs to download (Train Images is the large one)
urls_to_download = [
    # 1. Training Images (Approx 18 GB)
    ("http://images.cocodataset.org/zips/train2017.zip", "train2017.zip"),
    # 2. Validation Images (Approx 1 GB)
    ("http://images.cocodataset.org/zips/val2017.zip", "val2017.zip"),
    # 3. Annotations/Captions (Approx 241 MB)
    ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "annotations_trainval2017.zip")
]

# --- Download Logic ---

def download_file(url, filename):
    """Downloads a file using wget and handles existing files."""
    if os.path.exists(filename):
        print(f"File '{filename}' already exists. Skipping download.")
        return
    
    print(f"\nStarting download for {filename}...")
    
    # wget.download handles progress bar and large files better
    wget.download(url, filename)
    print(f"\n{filename} downloaded successfully!")

print("--- Starting COCO Dataset Download ---")

for url, filename in urls_to_download:
    download_file(url, filename)

print("\n--- All COCO files checked/downloaded. ---")
print(f"Next step: Extract the files inside the '{DATA_DIR}' folder.")