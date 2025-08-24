import torch
import cv2
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO

# Import our U-Net model definition
from model import UNET

# --- 1. Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UNET_MODEL_PATH = r"C:\Users\ARYAN\Dropbox\PC\Desktop\lane_detection\unet_lanes.pth"
YOLO_MODEL_PATH = r"C:\Users\ARYAN\runs\detect\train5\weights\best.pt"
VIDEO_PATH = r"C:\Users\ARYAN\Dropbox\PC\Desktop\TrafficVideoTEST.mp4"  # <-- CHANGE THIS

# Define the image dimensions for the U-Net
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Define the color mapping for the U-Net's output classes
CLASS_COLOR_MAP = {
    0: (0, 0, 0),       # Black for background
    1: (0, 255, 0),     # Green for drivable area
    2: (0, 0, 255),     # Red for one lane type
    3: (255, 0, 0),     # Blue for another lane type
}


# --- 2. Main Processing Function ---
def process_video(unet_model, yolo_model, video_path, transform, device):
    """
    Opens a video file and processes each frame with both U-Net and YOLO.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        original_height, original_width, _ = frame.shape

        # --- U-Net Prediction ---
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image_rgb)
        image_tensor = transformed['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            segmentation_pred = unet_model(image_tensor)

        predicted_mask = torch.argmax(segmentation_pred, dim=1).squeeze(0).cpu().numpy()

        color_mask = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)
        for class_id, color in CLASS_COLOR_MAP.items():
            color_mask[predicted_mask == class_id] = color

        color_mask_resized = cv2.resize(color_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # Create the initial overlay with the segmentation mask
        final_frame = cv2.addWeighted(frame, 0.6, color_mask_resized, 0.4, 0)

        # --- YOLO Prediction ---
        yolo_results = yolo_model(frame)
        result = yolo_results[0]  # first (and only) image

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            class_name = result.names[int(box.cls[0].cpu())]
            confidence = float(box.conf[0].cpu())

            # Draw bounding box
            cv2.rectangle(final_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # Draw label
            cv2.putText(
                final_frame,
                f"{class_name} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )

        # Display the final frame
        cv2.imshow("Real-Time Lane and Vehicle Detection", final_frame)

        # Press 'q' to exit the video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# --- 3. Script Execution ---
if __name__ == '__main__':
    # Define U-Net transformations
    unet_transform = A.Compose(
        [
            A.Resize(IMG_HEIGHT, IMG_WIDTH),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # Load the trained U-Net model
    print(f"Loading U-Net model from {UNET_MODEL_PATH}...")
    unet = UNET(in_channels=3, out_channels=4).to(DEVICE)
    unet.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=torch.device(DEVICE)))
    unet.eval()
    print("U-Net model loaded successfully.")

    # Load the trained YOLO model
    print(f"Loading YOLOv8 model from {YOLO_MODEL_PATH}...")
    yolo = YOLO(YOLO_MODEL_PATH)
    print("YOLOv8 model loaded successfully.")

    # Process the video
    process_video(unet, yolo, VIDEO_PATH, unet_transform, DEVICE)
