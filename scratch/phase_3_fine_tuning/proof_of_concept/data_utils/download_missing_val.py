import os
import json
import shutil
from huggingface_hub import hf_hub_download
from tqdm import tqdm

DATA_JSON = 'data/dataset.json'
IMG_DIR = 'data/raw/images'
REPO_ID = "ibrahimhamamci/CT-RATE"

def download_missing():
    with open(DATA_JSON, 'r') as f:
        dataset = json.load(f)

    # Find the missing validation images
    missing_val_files = []
    for item in dataset.get('val', []):
        name = item['name']
        if not os.path.exists(os.path.join(IMG_DIR, name)):
            missing_val_files.append(name)

    print(f"Found {len(missing_val_files)} missing validation images.")
    if len(missing_val_files) == 0:
        return

    # Download them directly from the dataset repository
    for name in tqdm(missing_val_files, desc="Downloading missing CT images"):
        try:
            # Reconstruct the CT-RATE nested path:
            # Example name: train_2550_a_2.nii.gz
            # folder1 = train_2550
            # folder2 = train_2550_a
            name_no_ext = name.split('.')[0]
            parts = name_no_ext.split('_')
            
            if parts[0] == 'train':
                folder1 = f"{parts[0]}_{parts[1]}"
                folder2 = f"{parts[0]}_{parts[1]}_{parts[2]}"
                remote_path = f"dataset/train/{folder1}/{folder2}/{name}"
            elif parts[0] == 'valid':
                folder1 = f"{parts[0]}_{parts[1]}"
                folder2 = f"{parts[0]}_{parts[1]}_{parts[2]}"
                remote_path = f"dataset/valid/{folder1}/{folder2}/{name}"
            else:
                raise ValueError(f"Unknown split prefix in filename: {name}")

            downloaded_path = hf_hub_download(
                repo_id=REPO_ID, 
                filename=remote_path, 
                repo_type="dataset"
            )
            
            target_path = os.path.join(IMG_DIR, name)
            shutil.copy(downloaded_path, target_path)
            
        except Exception as e:
            print(f"Failed to download {name}: {e}")

if __name__ == "__main__":
    download_missing()
