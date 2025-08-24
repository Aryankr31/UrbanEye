import cv2
import os
import numpy as np

# --- Configuration ---
# Path to the mask you want to view
MASK_PATH = r"C:\Users\ARYAN\Dropbox\PC\Desktop\lane_detection\data\multi_class_masks\val\b1c9c847-3bda4659.png" # <-- CHANGE THIS

# Define the same color map from our other scripts
# 0=background, 1=drivable, 2=lane_type_1, 3=lane_type_2
CLASS_COLOR_MAP = {
    0: (0, 0, 0),       # Black for background
    1: (0, 255, 0),     # Green for drivable area
    2: (0, 0, 255),     # Red for lanes
    3: (255, 0, 0),     # Blue for other lanes
}

# --- Main Script ---
def visualize_mask(mask_path):
    # Load the grayscale mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not read the mask at {mask_path}")
        return

    # Create an empty color image
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Apply the colors based on the pixel values
    for class_id, color in CLASS_COLOR_MAP.items():
        color_mask[mask == class_id] = color
    
    # Show the colorized mask
    print("Displaying colorized mask. Press any key to close.")
    cv2.imshow("Mask Visualization", color_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # --- IMPORTANT ---
    # Find a filename in your 'multi_class_masks/train' folder
    # and replace "some_filename.png" in the MASK_PATH variable above.
    if "some_filename.png" in MASK_PATH:
        print("Please update the MASK_PATH variable in the script first!")
    else:
        visualize_mask(MASK_PATH)