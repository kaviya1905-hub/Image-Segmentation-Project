# Cityscapes Image Segmentation with U-Net
-This project implements semantic image segmentation to detect objects like vehicles and pedestrians in urban scenes using the Cityscapes dataset and a U-Net architecture. The goal is to generate pixel-wise masks for images from real-world streetscapes.

# Dataset

- **Source**: [Cityscapes on Kaggle](https://www.kaggle.com/datasets/shuvoalok/cityscapes)
- **Description:** High-resolution urban scene images with pixel-level labels (e.g., road, building, vehicle, pedestrian).
- **Folder Structure After Extraction:**
 Image Segmentation Model/
    ├── train/
    │   ├── images/
    │   └── masks/
    └── val/
        ├── images/
        └── masks/
  
## Model

- **Architecture:** U-Net (encoder-decoder with skip connections)
- **Loss:** Weighted Binary Cross-Entropy to address class imbalance (foreground vs background)
- **Augmentation:** Random horizontal flip, rotation, color jitter, resizing
- **Metrics:** Dice coefficient and Intersection over Union (IoU) for boundary and overlap accuracy

## Setup

1. Download and extract the Cityscapes dataset as described above.
2. Install the required packages:
    ```
    pip install torch torchvision pillow numpy matplotlib
    ```
3. Update `DATA_PATH` in `segmentation_unet.py` if needed.
4. Run the training script:
    ```
    python segmentation_unet.py
    ```
# Training and Validation Progress

- The model trains for 15 epochs, with clear tracking of loss and metrics.
Epoch 01/15 | Train Loss: 1.0681 | Val Loss: 1.0481 | Dice: 0.0106 | IoU: 0.0053
...
Epoch 15/15 | Train Loss: 0.6160 | Val Loss: 1.2396 | Dice: 0.6672 | IoU: 0.5006

# Training Screenshot
<img width="949" height="872" alt="Screenshot 2025-10-30 120144" src="https://github.com/user-attachments/assets/fc486362-9c4e-49f9-a52f-f5af350284d1" />

## Prediction Visualizations
-The plots of these are:
<img width="1539" height="500" alt="output" src="https://github.com/user-attachments/assets/8a263587-1098-409c-8995-116a2d946464" />

- Columns: **Input | Ground Truth | Prediction**
- Example result (outputs from the last validation batch):
<img width="1912" height="958" alt="Screenshot 2025-10-30 121608" src="https://github.com/user-attachments/assets/a69eb6d5-46ca-474d-b749-ed1be6bd0e01" />

**Interpretation:**  
- The model quickly learns to map from input images to masks (even with small/dummy datasets).
- Dice and IoU scores significantly improve after the first epoch, and the model maintains decent boundary overlap metrics across epochs.
- Output masks visually resemble ground truth, showing the U-Net's ability to localize object boundaries.

## Methodology & Key Choices

- **Model Architecture:**  
  U-Net is chosen for its ability to preserve spatial detail with skip connections, which is critical for precise segmentation tasks such as edge and boundary detection in traffic scenes.

- **Class Imbalance Handling:**  
  The dataset is dominated by background pixels. A weighted binary cross-entropy loss (foreground-weighted) is used to ensure that rare object pixels (cars, pedestrians) are not ignored during training.

- **Augmentation:**  
  Random flips, rotation, color jitter, and resizing are used to make the model robust to diverse weather, lighting, and camera angles (which are common in street scenes).

- **Boundary Accuracy:**  
  Dice and IoU metrics are monitored every epoch to evaluate overlap and penalize poor boundary predictions, which are important for real-world segmentation use cases.

## Author

Kaviya M

