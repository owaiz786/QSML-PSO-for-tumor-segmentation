# src/data_loader.py
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

IMG_SIZE = 128
DATA_DIR = 'data/archive/lgg-mri-segmentation/kaggle_3m' 

def load_data(sample_size=1000):
    """Loads a sample of images and masks and prepares them for training."""

    # ✅ --- CRITICAL FIX: Corrected the search pattern to be one level shallower ---
    mask_files_pattern = os.path.join(DATA_DIR, '*/*_mask.tif')
    # ✅ --- END OF FIX ---
    
    all_files = glob.glob(mask_files_pattern)
    
    if not all_files:
        raise FileNotFoundError(f"No mask files found using the pattern: {mask_files_pattern}. "
                              f"Please check that the DATA_DIR is correct and that mask files exist one level inside it.")

    actual_sample_size = min(sample_size, len(all_files))
    sample_files = np.random.choice(all_files, actual_sample_size, replace=False)
    
    images = []
    masks = []

    for mask_path in tqdm(sample_files, desc="Loading Data"):
        try:
            image_path = mask_path.replace('_mask.tif', '.tif')
            
            img = Image.open(image_path).convert('L')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            images.append(np.array(img))
            
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize((IMG_SIZE, IMG_SIZE))
            binary_mask = (np.array(mask) > 0).astype(np.uint8) 
            masks.append(binary_mask)
        except FileNotFoundError:
            print(f"Warning: Corresponding image not found for mask: {mask_path}")
            continue

    if len(images) == 0:
        raise ValueError("Failed to load any image-mask pairs. Please check file paths and naming conventions.")

    images = np.array(images, dtype=np.float32) / 255.0
    masks = np.array(masks, dtype=np.float32)
    images = np.expand_dims(images, axis=-1)
    masks = np.expand_dims(masks, axis=-1)

    X_train, X_temp, y_train, y_temp = train_test_split(images, masks, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Data loaded successfully: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples.")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)