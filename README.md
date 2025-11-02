# U-Net Binary Image Segmentation:

A complete pipeline for binary (foreground/background) segmentation in urban and traffic scenes, using U-Net. The model learns to generate pixel-level masks for vehicles, pedestrians, and other objects. Results and outputs include training curves, visualized predictions, boundary metrics, and checkpoint saving.

---

## Project Structure:
-unet_segmenatation.py<br>
-data/
  *image/
  *masks/
-results/
  *training_curves.png 
-README.md

---

## Dataset

- **Default:** Synthetic images and masks (rectangles/ellipses) are auto-generated for demo and quick testing.
- **Real Data:** Replace `data/images/` and `data/masks/` with real urban images and corresponding pixel-wise masks (e.g., crops from Cityscapes). Must be 256×256 (or will be auto-resized).


---

## Model

- **Architecture:** U-Net (PyTorch), encoder-decoder with skip connections
- **Loss Function:** Dice + weighted Binary Cross-Entropy (foreground upweighted)
- **Augmentation:** Random flip, rotation, color jitter, Cutout
- **Metrics:** Loss, Dice, IoU, Boundary F1, Hausdorff Distance

---

## Setup & Usage

**1. Install Dependencies**
pip install torch torchvision pillow numpy matplotlib scipy tqdm
**2. Run the Script**
python unet_segmentation.py
- If images and masks don't exist, generates synthetic data.
- Trains U-Net for 15 epochs with per-epoch loss/metric logging.
- Automatically saves training curves and several prediction outputs.

## Training Output (Sample)
Epoch 15/15 Results:
Train Loss: 0.3638 | Val Loss: 0.3540
Val IoU: 0.926 | Val Dice: 0.961
Val Boundary F1: 0.541 | Val Hausdorff: 24.45
<img width="1207" height="773" alt="Screenshot 2025-11-02 214354" src="https://github.com/user-attachments/assets/29d6f2c3-ab91-430f-841c-62584b1a83e5" />
<img width="1204" height="765" alt="Screenshot 2025-11-02 214403" src="https://github.com/user-attachments/assets/726ac703-2d5b-44d9-9974-814aefc995d9" />
<img width="1216" height="775" alt="Screenshot 2025-11-02 214412" src="https://github.com/user-attachments/assets/13ac968a-3120-4af4-9469-a0ec6345dd1d" />
<img width="1220" height="771" alt="Screenshot 2025-11-02 214421" src="https://github.com/user-attachments/assets/def04fdd-cff9-4dec-804b-a9a301738fcf" />
<img width="1224" height="749" alt="Screenshot 2025-11-02 214429" src="https://github.com/user-attachments/assets/fbeed09a-23f3-4042-8ffa-065f6d22db0c" />
<img width="1167" height="740" alt="Screenshot 2025-11-02 214440" src="https://github.com/user-attachments/assets/eb5b0745-d90e-4f68-b04a-20c9f01744ae" />


## Prediction Visualizations
<img width="1915" height="998" alt="Screenshot 2025-11-02 214316" src="https://github.com/user-attachments/assets/5dc2dc58-d289-4d8a-a09a-15e47f17fba2" />


## Methodology & Key Choices

**Model Architecture:**  
- U-Net enables sharp object boundaries thanks to skip connections, essential for traffic/urban scenes
  
**Class Imbalance:**  
- Synthetic and real data usually have many more background pixels than objects. Combined Dice + weighted BCE loss ensures foreground pixels (vehicles, pedestrians) are not ignored.
  
**Augmentations:**  
- Applied: random flip, rotation, brightness/contrast jitter, Cutout. They help generalize to diverse weather, time, and sensor conditions in urban imagery.

**Boundary Accuracy:**  
- Validation tracks Dice/IoU for mask overlap and Boundary F1/Hausdorff metrics for edge precision.


## Preprocessing Steps

1. **Resize:** All images and masks are scaled to 256×256.
2. **Normalize images:** Min-max scaling to [0,1] range.
3. **Mask binarization:** Convert masks to single channel, binary (0/1).
4. **Augmentations:** Random flips, rotation, color jitter, Cutout for generalization.
5. **Class imbalance:** Compute foreground/background ratio for dynamic pos_weight in BCE.
6. **PyTorch batching:** Efficient batch sampling and DataLoader handling.

---

## Strengths

- **High localization accuracy:** Skip connections allow for precise boundaries[web:22][web:30].
- **Efficient on limited data:** Fast convergence and good performance, even when data is scarce.
- **Interpretable outputs:** Probability maps and overlay masks make results easy to review.
- **Class imbalance robustness:** Dynamic weighted losses handle rare object pixels.
- **Flexible:** Works for both synthetic and real input just by switching folders.

---

## Weaknesses

- **Small object performance:** May struggle with very tiny or highly occluded items due to output resolution[web:20][web:21].
- **Context limitation:** Classic U-Net can miss long-range context without additional modules (see transformer-U-Net hybrids)[web:22].
- **Memory use:** Deep U-Net can be heavy on CPU and RAM (GPU highly recommended).
- **Boundary crispness:** May need post-processing (e.g., CRF) for perfectly sharp edges.

---

## Limitations

- **Binary segmentation only:** This code only handles foreground/background (for multi-class, see U-Net+Categorical Cross-Entropy).
- **Annotation dependence:** Requires high-quality masks; noisy masks reduce performance.
- **Generalization:** Synthetic training data will not transfer perfectly to real-world scenes without further domain adaptation.

## Author

Kaviya M

