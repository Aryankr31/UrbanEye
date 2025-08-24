import json
import os

# --- CONFIGURATION ---
IMAGE_DIR = "data/images/train"
MASK_DIR = "data/drivable_masks/train"
ANNOTATION_DIR = "data/lane_annotations/train"

def find_and_inspect_first_valid_file():
    """
    Finds the first common file that has a non-empty 'labels' list
    and prints the unique categories found inside it.
    """
    print("Searching for the first annotation file with actual content...")
    
    try:
        # Find all files that exist in all three directories
        image_files = {os.path.splitext(f)[0] for f in os.listdir(IMAGE_DIR)}
        mask_files = {os.path.splitext(f)[0] for f in os.listdir(MASK_DIR)}
        anno_files = {os.path.splitext(f)[0] for f in os.listdir(ANNOTATION_DIR)}
        
        common_files = image_files.intersection(mask_files, anno_files)
        
        if not common_files:
            print("❌ Could not find any common files to inspect.")
            return

        # Iterate through common files to find one with content
        for filename in common_files:
            json_path = os.path.join(ANNOTATION_DIR, f"{filename}.json")
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                if 'frames' in data and data['frames']:
                    labels = data['frames'][0].get('labels', [])
                    
                    # If the labels list is NOT empty, we've found what we need
                    if labels:
                        print(f"\n--- Found a non-empty file: {filename}.json ---")
                        unique_categories = {label.get('category', 'N/A') for label in labels}
                        
                        print("\nResult: Found the following unique categories in this file:")
                        for category in sorted(list(unique_categories)):
                            print(f"- {category}")
                        return # Exit after finding the first valid file

            except (json.JSONDecodeError, IndexError):
                continue # Skip corrupted or malformed files
    
        print("\n❌ Searched all {len(common_files)} common files and none contained any labels.")

    except FileNotFoundError as e:
        print(f"❌ Error: A directory was not found. Please check your paths. Details: {e}")

if __name__ == "__main__":
    find_and_inspect_first_valid_file()