import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Custom imports
from dataloader import LaneDataset
from model import UNET
from albumentations.pytorch import ToTensorV2
import albumentations as A

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
MODEL_PATH = "unet_lanes.pth"
DATA_SPLIT = "val"

BASE_DATA_DIR = "data"
IMAGE_DIR = os.path.join(BASE_DATA_DIR, "images", DATA_SPLIT)
DRIVABLE_DIR = os.path.join(BASE_DATA_DIR, "drivable_masks", DATA_SPLIT)
LANE_DIR = os.path.join(BASE_DATA_DIR, "lane_annotations", "labels", DATA_SPLIT)
OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "multi_class_masks", DATA_SPLIT)

# --- Step 1: Generate Multi-class Masks ---
def create_multiclass_masks():
    print(f"\n[Step 1] Preparing masks for '{DATA_SPLIT}' set...")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # Use the drivable masks as the source of truth, as there are more of them
    drivable_files = os.listdir(DRIVABLE_DIR)
    
    for drivable_file in tqdm(drivable_files, desc=f"Generating masks for {DATA_SPLIT}"):
        # Get the simple base name, handling all known BDD suffixes
        base_name = os.path.splitext(drivable_file)[0].replace("_drivable_id", "").replace("_val_id", "")

        # IMPORTANT: Only proceed if a matching .jpg image actually exists
        image_path = os.path.join(IMAGE_DIR, f"{base_name}.jpg")
        if not os.path.exists(image_path):
            continue

        drivable_mask_path = os.path.join(DRIVABLE_DIR, drivable_file)
        lane_mask_path = os.path.join(LANE_DIR, f"{base_name}.png")

        drivable_mask = cv2.imread(drivable_mask_path, cv2.IMREAD_GRAYSCALE)
        if drivable_mask is None:
            continue

        new_mask = np.zeros_like(drivable_mask, dtype=np.uint8)
        new_mask[drivable_mask == 1] = 1

        lane_mask = cv2.imread(lane_mask_path, cv2.IMREAD_GRAYSCALE)
        if lane_mask is not None:
            new_mask[lane_mask == 1] = 2
            new_mask[lane_mask == 2] = 3

        # Save the mask with the simple base name that is guaranteed to match an image
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}.png")
        cv2.imwrite(output_path, new_mask)

    print(f"\n✅ Done! Multi-class masks saved in:\n{OUTPUT_DIR}")

# --- Step 2: Validation Function ---
val_transform = A.Compose(
    [
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    model.eval()

    print("\n[Step 2] Starting validation...")
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating"):
            x = x.to(device)
            y = y.to(device).squeeze(1).long()

            preds = model(x)
            preds = torch.argmax(preds, dim=1)

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

    if num_pixels == 0:
        print("⚠️ No pixels found! The dataloader found 0 matching pairs.")
        return 0.0

    accuracy = (num_correct / num_pixels) * 100
    print(f"\n🎯 Pixel Accuracy on Validation Set: {accuracy:.2f}%")
    model.train()
    return accuracy

# --- Step 3: Main ---
def main():
    create_multiclass_masks()

    model = UNET(in_channels=3, out_channels=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))

    val_dataset = LaneDataset(
        image_dir=IMAGE_DIR,
        mask_dir=OUTPUT_DIR,
        transform=val_transform
    )

    if len(val_dataset) == 0:
        print("❌ No valid dataset found after mask creation! Check the directories and file names.")
        return

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    check_accuracy(val_loader, model, device=DEVICE)

if __name__ == "__main__":
    main()