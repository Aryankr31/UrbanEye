import os
import cv2
import numpy as np

# --- 1. Correct Configuration ---
# These paths are based on the ones you provided.
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images/train")
LANE_DIR = os.path.join(DATA_DIR, "lane_annotations/labels/train") 

# --- 2. Automatically Find a Valid Filename ---
def find_valid_sample(image_dir, lane_dir):
    """
    Finds a valid base filename that exists in both the image and lane mask folders,
    even if the mask has a different suffix (like _train_id.png).
    """
    print("Searching for a valid file that exists in both folders...")
    try:
        image_basenames = {os.path.splitext(f)[0] for f in os.listdir(image_dir)}
        lane_mask_filenames = os.listdir(lane_dir)
    except FileNotFoundError as e:
        print(f"❌ FATAL ERROR: A directory was not found. Please check your paths in the script. Details: {e}")
        return None, None

    # Handle BDD's complex naming (e.g., '...-...._train_id.png') vs simple names ('...-....png')
    for image_base in image_basenames:
        # Check for complex name first
        expected_mask_name = f"{image_base}_train_id.png"
        if expected_mask_name in lane_mask_filenames:
            print(f"✅ Found a valid sample: {image_base}")
            return image_base, expected_mask_name
        
        # Fallback to check for simple name
        expected_mask_name = f"{image_base}.png"
        if expected_mask_name in lane_mask_filenames:
            print(f"✅ Found a valid sample: {image_base}")
            return image_base, expected_mask_name

    print("❌ Error: Could not find any matching files between the image and lane mask folders.")
    return None, None

# Find a file to test
test_filename_base, lane_mask_fullname = find_valid_sample(IMAGE_DIR, LANE_DIR)

# Exit if no valid file was found
if not test_filename_base:
    exit()

# --- 3. Construct Final Paths Using Found Filenames ---
image_path = os.path.join(IMAGE_DIR, f"{test_filename_base}.jpg")
lane_mask_path = os.path.join(LANE_DIR, lane_mask_fullname)

# --- 4. Load and Visualize ---
print(f"Loading image: {image_path}")
print(f"Loading lane mask: {lane_mask_path}")

image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load image at {image_path}")
    exit()

lane_mask = cv2.imread(lane_mask_path, cv2.IMREAD_GRAYSCALE)
if lane_mask is None:
    print(f"Error: Lane mask not found at {lane_mask_path}")
    exit()

visualization = image.copy()

# Create a RED overlay for pixel value 1 (e.g., solid lines)
red_overlay = np.zeros_like(visualization)
red_overlay[lane_mask == 1] = [0, 0, 255] # Red

# Create a BLUE overlay for pixel value 2 (e.g., dashed lines)
blue_overlay = np.zeros_like(visualization)
blue_overlay[lane_mask == 2] = [255, 0, 0] # Blue

# Blend overlays
visualization = cv2.addWeighted(visualization, 1.0, red_overlay, 0.5, 0)
visualization = cv2.addWeighted(visualization, 1.0, blue_overlay, 0.5, 0)

print("Displaying image with lane overlays. Press any key to close.")

cv2.imshow("Lane Mask Verification", visualization)
cv2.waitKey(0)
cv2.destroyAllWindows()