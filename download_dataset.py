"""
Dataset Downloader for PlantHealth-AI Pro
Fetches balanced high-resolution leaf pathology images directly from the
official PlantVillage repository (spMohanty/PlantVillage-Dataset).
"""

import os
import urllib.request
import json
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Classes to download from PlantVillage color dataset
TARGET_CLASSES = {
    "Tomato___healthy": "Tomato_Healthy",
    "Tomato___Early_blight": "Tomato_Early_Blight",
    "Tomato___Late_blight": "Tomato_Late_Blight",
    "Tomato___Bacterial_spot": "Tomato_Bacterial_Spot",
    "Tomato___Septoria_leaf_spot": "Tomato_Septoria_Leaf_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato_Yellow_Leaf_Curl"
}

IMAGES_PER_CLASS = 120  # Balanced 120 images per class = 720 high-res leaf images

def fetch_class_image_urls(repo_folder_name, limit=120):
    """Fetch download URLs for images from GitHub API"""
    api_url = f"https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color/{repo_folder_name}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            items = json.loads(resp.read().decode("utf-8"))
            image_urls = [
                item["download_url"] for item in items 
                if item["name"].lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            return image_urls[:limit]
    except Exception as e:
        print(f"Error fetching directory {repo_folder_name}: {e}")
        return []

def download_single_image(url, dest_path):
    """Download a single image file"""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return True
    try:
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def setup_dataset():
    """Download and prepare the dataset folder structure"""
    os.makedirs(DATASET_DIR, exist_ok=True)
    print("=" * 60)
    print("PLANT HEALTH DATASET ACQUISITION (PlantVillage)")
    print("=" * 60)
    
    total_downloaded = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        for pv_name, local_name in TARGET_CLASSES.items():
            class_dir = os.path.join(DATASET_DIR, local_name)
            os.makedirs(class_dir, exist_ok=True)
            print(f"\nFetching URLs for [{local_name}]...")
            
            urls = fetch_class_image_urls(pv_name, limit=IMAGES_PER_CLASS)
            print(f"Found {len(urls)} images for {local_name}. Downloading...")
            
            futures = []
            for i, url in enumerate(urls):
                ext = os.path.splitext(url)[1] or ".jpg"
                dest_path = os.path.join(class_dir, f"{local_name.lower()}_{i+1:03d}{ext}")
                futures.append(executor.submit(download_single_image, url, dest_path))
            
            success_count = sum(1 for f in futures if f.result())
            print(f"Successfully saved {success_count} images to dataset/{local_name}")
            total_downloaded += success_count
            
    print("\n" + "=" * 60)
    print(f"Dataset download complete! Total images: {total_downloaded}")
    print(f"Saved to: {DATASET_DIR}")
    print("=" * 60)
    return total_downloaded

if __name__ == "__main__":
    setup_dataset()
