import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


# Import our custom classes from other files
from dataloader import LaneDataset
from model import UNET

# --- 1. Configuration and Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

LEARNING_RATE = 2e-4
BATCH_SIZE = 8
NUM_EPOCHS = 5 # An epoch is one full pass over the training data
NUM_WORKERS = 2

# --- 2. Data Loading ---
# IMPORTANT: Update the MASK_DIR to point to our new multi-class masks
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images/train")
MASK_DIR = os.path.join(DATA_DIR, "multi_class_masks/train") # <-- FINAL MASK DIRECTORY

# We need to update our dataloader to handle the new multi-class masks
# Specifically, the mask needs to be a LongTensor for CrossEntropyLoss
# Let's import the transformations from dataloader.py and adjust them
from dataloader import A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose(
    [
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

# Create the dataset and dataloader
train_dataset = LaneDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    transform=train_transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True # Speeds up data transfer to GPU
)

# --- 3. The Training Loop ---
def train_model(loader, model, optimizer, loss_fn):
    """Runs one epoch of training."""
    loop = tqdm(loader) # For a nice progress bar

    for batch_idx, (data, targets) in enumerate(loop):
        # Move data to the GPU (if available)
        data = data.to(device=DEVICE)
        targets = targets.squeeze(1).long() # Squeeze and convert to LongTensor
        targets = targets.to(device=DEVICE)

        # Forward pass
        predictions = model(data)
        
        # Calculate loss
        loss = loss_fn(predictions, targets)

        # Backward pass (backpropagation)
        optimizer.zero_grad() # Reset gradients
        loss.backward() # Calculate new gradients
        
        # Gradient descent
        optimizer.step() # Update model weights

        # Update the progress bar description
        loop.set_postfix(loss=loss.item())

# --- 4. Main Function to Run Everything ---
def main():
    # Initialize the model, loss, and optimizer
    model = UNET(in_channels=3, out_channels=4).to(DEVICE)
    
    # CrossEntropyLoss is standard for multi-class segmentation
    loss_fn = nn.CrossEntropyLoss()
    
    # Adam is a popular and effective optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Run the training for the specified number of epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_model(train_loader, model, optimizer, loss_fn)

    # --- 5. Save the Model ---
    torch.save(model.state_dict(), "unet_lanes.pth")
    print("\n✅ Training complete! Model saved to unet_lanes.pth")

if __name__ == "__main__":
    main()