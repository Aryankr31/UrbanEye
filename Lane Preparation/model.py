import torch
import torch.nn as nn

# This is a standard building block for U-Net
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# The main U-Net model
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNET, self).__init__()
        
        # Encoder (Downsampling path)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = DoubleConv(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(128, 256)
        self.down3 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(256, 512)
        self.down4 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(512, 1024)
        
        # Decoder (Upsampling path)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv5 = DoubleConv(1024, 512) # Skip connection from conv3 is added
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(512, 256) # Skip connection from conv2
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(256, 128) # Skip connection from conv1
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(128, 64) # Skip connection from inc
        
        # Final output layer
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.conv1(x2)
        x3 = self.down2(x2)
        x3 = self.conv2(x3)
        x4 = self.down3(x3)
        x4 = self.conv3(x4)
        x5 = self.down4(x4)
        x5 = self.conv4(x5)
        
        # Decoder with skip connections
        x = self.up1(x5)
        # Concatenate the upsampled feature map with the corresponding feature map from the encoder
        x = torch.cat([x, x4], dim=1)
        x = self.conv5(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv6(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv7(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv8(x)
        
        # Get the final prediction mask
        logits = self.outc(x)
        return logits

# This block is for testing the model architecture
if __name__ == '__main__':
    # Create a dummy input tensor with the same shape as a batch of images
    # [batch_size, channels, height, width]
    dummy_image_batch = torch.randn(4, 3, 256, 256)
    
    # Create an instance of the U-Net model
    # Input channels = 3 (for RGB images)
    # Output channels = 1 (for a single-class mask)
    model = UNET(in_channels=3, out_channels=4)
    
    # Pass the dummy input through the model
    preds = model(dummy_image_batch)
    
    # Check if the output shape is correct
    print("Testing the U-Net model...")
    print(f"Shape of the input batch: {dummy_image_batch.shape}")
    print(f"Shape of the output predictions: {preds.shape}")
    
    # The output shape should match the mask shape from our data loader
    assert preds.shape == (4, 4, 256, 256)
    print("\nU-Net model is working correctly! 👍")