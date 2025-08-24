import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LaneDataset(Dataset):
    """
    Final, robust version of the custom PyTorch Dataset.
    It finds the intersection of available images and masks to prevent errors.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.samples = self._find_matching_samples()
        
        print(f"Found {len(self.samples)} matching image-mask pairs in the provided directories.")


    def _find_matching_samples(self):
        """
        Finds the intersection of image and mask files to ensure all pairs exist.
        """
        image_basenames = {os.path.splitext(f)[0] for f in os.listdir(self.image_dir)}
        mask_basenames = {os.path.splitext(f)[0] for f in os.listdir(self.mask_dir)}
        
        common_basenames = list(image_basenames.intersection(mask_basenames))
        
        # Create a list of (image_filename, mask_filename) tuples
        samples = [(f"{name}.jpg", f"{name}.png") for name in common_basenames]
        
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, mask_name = self.samples[idx]
        
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            mask = mask.unsqueeze(0)

        return image, mask

# (The test block below remains the same for testing the train set)
if __name__ == '__main__':
    DATA_DIR = "data"
    IMAGE_DIR = os.path.join(DATA_DIR, "images/train")
    MASK_DIR = os.path.join(DATA_DIR, "multi_class_masks/train") 
    
    train_transform = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    
    train_dataset = LaneDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        transform=train_transform
    )
    
    if len(train_dataset) > 0:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=2
        )
        
        images, masks = next(iter(train_loader))
        
        print(f"\nShape of one batch of images: {images.shape}")
        print(f"Shape of one batch of masks: {masks.shape}")
        print("\nDataLoader test is working correctly! 👍")