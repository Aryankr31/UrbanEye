import os

# --- Configuration ---
IMAGE_DIR = "data/images/val"
MASK_DIR = "data/drivable_masks/val"
OUTPUT_FILE = "matching_files.txt"
MAX_FILES = 1000

def find_and_save_matches():
    """
    Finds matching basenames between an image and a mask directory
    and saves them to a text file.
    """
    print("--- Finding Matching Files ---")
    
    try:
        # Get basenames from images (e.g., 'abc-123' from 'abc-123.jpg')
        image_basenames = {os.path.splitext(f)[0] for f in os.listdir(IMAGE_DIR)}
        print(f"Found {len(image_basenames)} total images.")

        # Get basenames from masks, handling complex names like '_drivable_id.png'
        mask_basenames = {
            os.path.splitext(f)[0].replace("_drivable_id", "").replace("_val_id", "") 
            for f in os.listdir(MASK_DIR)
        }
        print(f"Found {len(mask_basenames)} total masks.")

        # Find the intersection (names that are in both sets)
        common_basenames = list(image_basenames.intersection(mask_basenames))
        
        if not common_basenames:
            print("\n❌ No matching files found between the two directories.")
            return

        print(f"\n✅ Found {len(common_basenames)} matching files.")
        
        # Limit to the first 1000 matches
        matches_to_save = common_basenames[:MAX_FILES]
        
        # Save the list to a text file
        with open(OUTPUT_FILE, 'w') as f:
            for name in matches_to_save:
                f.write(f"{name}\n")
                
        print(f"\nSuccessfully saved {len(matches_to_save)} matching filenames to '{OUTPUT_FILE}'.")

    except FileNotFoundError as e:
        print(f"❌ Error: A directory was not found. Please check your paths. Details: {e}")

if __name__ == "__main__":
    find_and_save_matches()