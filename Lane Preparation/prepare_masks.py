import os
import cv2
import numpy as np
from tqdm import tqdm

def create_multiclass_masks():
    """
    Merges drivable area and lane masks into a single multi-class mask.
    Fixes Windows path issues (\b interpreted as backspace).
    """
    
    # --- 1. Configuration: Set this to 'val' or 'train' ---
    DATA_SPLIT = "val"   # change to "train" when needed

    BASE_DATA_DIR = "data"
    
    # Input directories (safe join, no mixed slashes)
    DRIVABLE_DIR = os.path.join(BASE_DATA_DIR, "drivable_masks", DATA_SPLIT)
    LANE_DIR = os.path.join(BASE_DATA_DIR, "lane_annotations", "labels", DATA_SPLIT)
    
    # Output directory
    OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "multi_class_masks", DATA_SPLIT)
    
    print(f"Starting mask preparation for the '{DATA_SPLIT}' set...")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
        
    drivable_mask_files = os.listdir(DRIVABLE_DIR)
    
    for filename in tqdm(drivable_mask_files, desc=f"Processing {DATA_SPLIT} masks"):
        # --- Get base name without extension ---
        base_name_with_suffix = os.path.splitext(filename)[0]
        # Handle BDD's extra suffix
        base_name = base_name_with_suffix.replace("_val_id", "").replace("_train_id", "")

        # --- Construct correct input paths ---
        drivable_mask_path = os.path.normpath(os.path.join(DRIVABLE_DIR, filename))
        lane_mask_path = os.path.normpath(os.path.join(LANE_DIR, f"{base_name}.png"))
        
        # Debug: check file existence
        if not os.path.exists(drivable_mask_path):
            print(f"⚠️ Drivable mask not found: {drivable_mask_path}")
            continue
        if not os.path.exists(lane_mask_path):
            print(f"⚠️ Lane mask missing: {lane_mask_path}")
        
        # Load masks
        drivable_mask = cv2.imread(drivable_mask_path, cv2.IMREAD_GRAYSCALE)
        lane_mask = cv2.imread(lane_mask_path, cv2.IMREAD_GRAYSCALE)
        
        if drivable_mask is None:
            print(f"❌ Failed to load drivable mask: {drivable_mask_path}")
            continue
            
        new_mask = np.zeros_like(drivable_mask, dtype=np.uint8)
        new_mask[drivable_mask == 1] = 1  # class 1 = drivable
        
        if lane_mask is not None:
            new_mask[lane_mask == 1] = 2   # class 2 = lane type 1
            new_mask[lane_mask == 2] = 3   # class 3 = lane type 2
            
        # --- Save the new mask with simple base name ---
        output_path = os.path.normpath(os.path.join(OUTPUT_DIR, f"{base_name}.png"))
        cv2.imwrite(output_path, new_mask)
        
    print(f"\n✅ Done! Your '{DATA_SPLIT}' multi-class masks are saved in:\n{OUTPUT_DIR}")

if __name__ == '__main__':
    create_multiclass_masks()
