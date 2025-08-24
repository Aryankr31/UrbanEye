import os

# --- Configuration ---
IMAGE_DIR = "data/images/val"
MASK_DIR = "data/drivable_masks/val"

def debug_the_names():
    print("--- Debugging File Naming Conventions ---")
    try:
        image_files = sorted(os.listdir(IMAGE_DIR))
        mask_files = sorted(os.listdir(MASK_DIR))

        print(f"\n--- First 5 Image Filenames ---")
        for f in image_files[:5]:
            print(f)

        print(f"\n--- First 5 Mask Filenames ---")
        for f in mask_files[:5]:
            print(f)

    except FileNotFoundError as e:
        print(f"❌ Error: A directory was not found. Please check your paths. Details: {e}")

if __name__ == "__main__":
    debug_the_names()