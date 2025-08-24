import os

# --- Configuration ---
# The two folders we need to find a matching filename in
IMAGE_DIR = "data/images/train"
LANE_DIR = "data/images/train"

def find_common_sample():
    """Compares two directories and finds a base filename that exists in both."""
    print("Searching for a filename that exists in both image and lane mask folders...")
    
    try:
        # Get a set of base names from each directory
        image_files = {os.path.splitext(f)[0] for f in os.listdir(IMAGE_DIR)}
        lane_files = {os.path.splitext(f)[0] for f in os.listdir(LANE_DIR)}
    except FileNotFoundError as e:
        print(f"❌ Error: A directory was not found. Please check your paths. Details: {e}")
        return

    # Find the names that are in both sets
    common_files = image_files.intersection(lane_files)
    
    if common_files:
        example_name = common_files.pop()
        print(f"\n✅ Success! Found {len(common_files) + 1} matching file pairs.")
        print("------------------------------------------------------------")
        print("You can use this base filename in 'check_data.py':")
        print(f'test_filename_base = "{example_name}"')
        print("------------------------------------------------------------")
    else:
        print("\n❌ No common filenames found between the two directories.")

if __name__ == "__main__":
    find_common_sample()