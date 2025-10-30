import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
DATA_PATH = r"D:\OneDrive\Documents\Image Segmentation Model"
BATCH_SIZE = 4
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
IMG_HEIGHT, IMG_WIDTH = 256, 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")
print(f"Data path: {DATA_PATH}\n")

folders = [
    os.path.join(DATA_PATH, "train", "images"),
    os.path.join(DATA_PATH, "train", "masks"),
    os.path.join(DATA_PATH, "val", "images"),
    os.path.join(DATA_PATH, "val", "masks")
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✓ Created/verified: {folder}")

print("\nGenerating dummy dataset...")
for split in ['train', 'val']:
    img_dir = os.path.join(DATA_PATH, split, "images")
    mask_dir = os.path.join(DATA_PATH, split, "masks")
    
    num_samples = 20 if split == 'train' else 5
    
    if len(os.listdir(img_dir)) == 0:
        for i in range(num_samples):
            img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            img.save(os.path.join(img_dir, f"img_{i:03d}.png"))
            mask = Image.fromarray(np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255)
            mask.save(os.path.join(mask_dir, f"img_{i:03d}.png"))
        
        print(f"  ✓ Created {num_samples} samples in {split}/")
    else:
        existing = len(os.listdir(img_dir))
        print(f"  ✓ {split}/ already has {existing} images")

print("\n" + "="*60)
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        print(f"Dataset: Found {len(self.images)} images in {os.path.basename(os.path.dirname(image_dir))}/")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        mask = (mask > 0).float()
        
        return image, mask
train_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.bottleneck = DoubleConv(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        self.out = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return torch.sigmoid(self.out(dec1))

class WeightedBCELoss(nn.Module):
    def __init__(self, weight_positive=2.0):
        super(WeightedBCELoss, self).__init__()
        self.weight_positive = weight_positive
    
    def forward(self, pred, target):
        bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        weight = torch.where(target > 0.5, self.weight_positive, 1.0)
        return (bce * weight).mean()

def dice_coefficient(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses, val_losses, dice_scores, iou_scores = [], [], [], []
    
    print("\n" + "="*60)
    print("TRAINING STARTED")
    print("="*60)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = dice_sum = iou_sum = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
                dice_sum += dice_coefficient(outputs, masks).item()
                iou_sum += iou_score(outputs, masks).item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_dice = dice_sum / len(val_loader)
        avg_iou = iou_sum / len(val_loader)
        
        val_losses.append(avg_val_loss)
        dice_scores.append(avg_dice)
        iou_scores.append(avg_iou)
        
        print(f"Epoch {epoch+1:02d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f}")
    
    print("="*60)
    print("TRAINING COMPLETED")
    print("="*60 + "\n")
    
    return train_losses, val_losses, dice_scores, iou_scores
def plot_results(train_losses, val_losses, dice_scores, iou_scores):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(train_losses, label='Train', marker='o')
    axes[0].plot(val_losses, label='Val', marker='s')
    axes[0].set_title('Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(dice_scores, marker='o', color='green')
    axes[1].set_title('Dice Score', fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    axes[2].plot(iou_scores, marker='o', color='red')
    axes[2].set_title('IoU Score', fontweight='bold')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, val_loader, num_samples=4):
    model.eval()
    images, masks = next(iter(val_loader))
    images, masks = images.to(DEVICE), masks.to(DEVICE)
    
    with torch.no_grad():
        preds = model(images)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples*3))
    
    for i in range(min(num_samples, len(images))):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')
        
        mask = masks[i].cpu().squeeze().numpy()
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        pred = preds[i].cpu().squeeze().numpy()
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    train_dataset = SegmentationDataset(
        os.path.join(DATA_PATH, "train/images"),
        os.path.join(DATA_PATH, "train/masks"),
        train_transform
    )
    
    val_dataset = SegmentationDataset(
        os.path.join(DATA_PATH, "val/images"),
        os.path.join(DATA_PATH, "val/masks"),
        val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    model = UNet(3, 1).to(DEVICE)
    criterion = WeightedBCELoss(2.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    train_losses, val_losses, dice_scores, iou_scores = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS
    )
    
    model_path = os.path.join(DATA_PATH, "unet_segmentation.pth")
    torch.save(model.state_dict(), model_path)
    print(f" Model saved: {model_path}\n")
    
    plot_results(train_losses, val_losses, dice_scores, iou_scores)
    visualize_predictions(model, val_loader, 4)
    
    print(" Complete!")

