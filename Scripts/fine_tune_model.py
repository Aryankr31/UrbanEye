from ultralytics import YOLO

# --- CONFIGURATION ---
# This script will fine-tune your existing model on a new dataset.

# 1. Path to YOUR existing, custom-trained model from the first training
EXISTING_MODEL_PATH = r"C:\Users\ARYAN\runs\detect\train2\weights\best.pt"

# 2. Path to your NEW multi-class dataset's configuration file
NEW_DATA_YAML_PATH = r"C:\Users\ARYAN\set2\data.yaml"

# 3. Stable training parameters for your hardware
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 8  # Use 8 for your GTX 1650. If you get a memory error, change to 4.
WORKERS = 0     # Use 0 for stability on Windows
# --------------------


# --- SCRIPT ---
# You don't need to change anything below this line.

# Load your existing model
# It already has expert knowledge from your first training session.
print(f"Loading existing model from: {EXISTING_MODEL_PATH}")
model = YOLO(EXISTING_MODEL_PATH) 

# Continue training this model on the new dataset to teach it the new classes
print(f"Starting fine-tuning on new dataset: {NEW_DATA_YAML_PATH}")
model.train(
   data=NEW_DATA_YAML_PATH,
   epochs=EPOCHS,
   imgsz=IMAGE_SIZE,
   batch=BATCH_SIZE,    
   workers=WORKERS
)

print("-----------------------------------------------------")
print("Fine-tuning session has finished successfully!")
print("A new 'best.pt' file has been saved in a new 'train' folder.")
print("-----------------------------------------------------")