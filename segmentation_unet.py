import os, random, glob, numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import directed_hausdorff
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList([DoubleConv(in_ch, features[0])] +
            [DoubleConv(features[i], features[i+1]) for i in range(len(features)-1)])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.upconv = nn.ModuleList([nn.ConvTranspose2d(features[-1]*2 // (2**i), features[-1]//(2**i), 2, 2) 
            for i in range(len(features))])
        self.decoder = nn.ModuleList([DoubleConv(features[-1]//(2**i)*2, features[-1]//(2**i)) 
            for i in range(len(features))])
        self.final_conv = nn.Conv2d(features[0], out_ch, 1)
    def forward(self, x):
        skips = []
        for down in self.encoder:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for idx in range(0, len(self.upconv)):
            x = self.upconv[idx](x)
            skip = skips[idx]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.decoder[idx](x)
        return self.final_conv(x)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0): super().__init__(); self.smooth = smooth
    def forward(self, pred, target):
        pred = torch.sigmoid(pred).view(-1); target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2.*intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, pos_weight=None, device='cpu'):
        super().__init__()
        self.alpha = alpha; self.dice = DiceLoss()
        if pos_weight:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        else:
            self.bce = nn.BCEWithLogitsLoss()
    def forward(self, pred, target): return self.alpha*self.bce(pred, target)+(1-self.alpha)*self.dice(pred, target)

def generate_sample(size=256, n_obj=5):
    img = np.ones((size, size, 3), np.uint8)*240; msk = np.zeros((size, size), np.uint8)
    I,M = Image.fromarray(img), Image.fromarray(msk); di,dm = ImageDraw.Draw(I), ImageDraw.Draw(M)
    for _ in range(n_obj):
        typ = random.choice(['rect','ellipse'])
        x1, y1, d1, d2 = random.randint(10,200), random.randint(10,200), random.randint(30,70), random.randint(30,70)
        x2, y2 = min(x1+d1,246), min(y1+d2,246)
        col = tuple(random.randint(50,200) for _ in range(3))
        if typ=='rect': di.rectangle([x1,y1,x2,y2],fill=col); dm.rectangle([x1,y1,x2,y2],fill=255)
        else: di.ellipse([x1,y1,x2,y2],fill=col); dm.ellipse([x1,y1,x2,y2],fill=255)
    return np.array(I), np.array(M)

def save_synth_data(root='data', n=100):
    os.makedirs(f'{root}/images',exist_ok=True); os.makedirs(f'{root}/masks',exist_ok=True)
    print(f"Generating {n} synthetic samples...")
    for i in range(n):
        im,msk=generate_sample()
        Image.fromarray(im).save(f'{root}/images/img_{i:03d}.png')
        Image.fromarray(msk).save(f'{root}/masks/msk_{i:03d}.png')
    print(f"✓ Synthetic data saved to {root}/")

class SegmDataset(Dataset):
    def __init__(self,img_dir,msk_dir,augment=True):
        self.img_p = sorted(glob.glob(f'{img_dir}/*.png')); self.msk_p = sorted(glob.glob(f'{msk_dir}/*.png'))
        assert len(self.img_p) == len(self.msk_p), f"Mismatch: {len(self.img_p)} images vs {len(self.msk_p)} masks"
        self.augment=augment; self.totensor = transforms.ToTensor()
        self.jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)
    def __len__(self): return len(self.img_p)
    def __getitem__(self,idx):
        image = Image.open(self.img_p[idx]).convert('RGB').resize((256,256))
        mask = Image.open(self.msk_p[idx]).convert('L').resize((256,256))
        image = self.totensor(image); mask = self.totensor(mask)
        mask = (mask>0.5).float()
        if self.augment:
            if random.random()<0.5: image = transforms.functional.hflip(image); mask = transforms.functional.hflip(mask)
            if random.random()<0.5: image = transforms.functional.vflip(image); mask = transforms.functional.vflip(mask)
            if random.random()<0.5: 
                angle=random.uniform(-30,30)
                image=transforms.functional.rotate(image, angle); mask=transforms.functional.rotate(mask, angle)
            image = self.jitter(image)
           
            if random.random()<0.3:
                x = random.randint(0,224); y = random.randint(0,224)
                image[:,y:y+32,x:x+32] = image.mean()
        return image, mask

def iou(pred, targ, thr=0.5):
    pred = (pred>thr).float(); targ = targ.float()
    inter = (pred*targ).sum(); union = pred.sum()+targ.sum()-inter
    return (inter/(union+1e-8)).item() if union!=0 else 1.0
def dice(pred, targ, thr=0.5):
    pred = (pred>thr).float(); targ = targ.float()
    inter = (pred*targ).sum(); d = (2*inter)/(pred.sum()+targ.sum()+1e-8)
    return d.item()
def pixel_acc(pred, targ, thr=0.5):
    pred = (pred>thr).float(); t = targ.float()
    return (pred==t).float().mean().item()
def extract_boundary(mask, dilate=2):
    er = binary_erosion(mask, iterations=dilate)
    return mask.astype(np.uint8)-er.astype(np.uint8)
def boundary_f1(pred, targ, thr=0.5, dilate=2):
    p = (pred>thr).cpu().numpy().squeeze()
    t = targ.cpu().numpy().squeeze()
    pb, tb = extract_boundary(p,dilate), extract_boundary(t,dilate)
    tp = np.logical_and(pb, tb).sum()
    fp = np.logical_and(pb, ~tb.astype(bool)).sum()
    fn = np.logical_and(~pb.astype(bool), tb).sum()
    precision = tp/(tp+fp+1e-8); recall = tp/(tp+fn+1e-8)
    return (2*precision*recall)/(precision+recall+1e-8)
def hausdorff(pred, targ, thr=0.5):
    try:
        p = (pred>thr).cpu().numpy().squeeze()
        t = targ.cpu().numpy().squeeze()
        pb, tb = extract_boundary(p), extract_boundary(t)
        pc, tc = np.argwhere(pb), np.argwhere(tb)
        if len(pc)==0 or len(tc)==0: return 0.0
        return max(directed_hausdorff(pc,tc)[0], directed_hausdorff(tc,pc)[0])
    except:
        return 0.0

def calc_class_weight(loader):
    pos, neg = 0, 0
    print("Calculating class weights...")
    for _,msk in loader: pos+=msk.sum().item(); neg+=(1-msk).sum().item()
    weight = neg/(pos+1e-8)
    print(f"Class imbalance ratio (neg/pos): {weight:.2f}")
    return weight

def train(model, tr_loader, val_loader, n_epoch, device, criterion, optimizer):
    tr_hist, val_hist = [], []
    best_val_loss = float('inf')
    for ep in range(n_epoch):
        model.train()
        tr_loss=0; tr_iou=[]; tr_dice=[]; tr_f1=[]; tr_hd=[]
        for img, msk in tqdm(tr_loader,desc=f"Epoch {ep+1}/{n_epoch} [Train]"):
            img, msk = img.to(device), msk.to(device)
            out = model(img); loss = criterion(out, msk)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); tr_loss+=loss.item()
            with torch.no_grad():
                pr = torch.sigmoid(out)
                for b in range(pr.shape[0]):
                    tr_iou.append(iou(pr[b],msk[b]))
                    tr_dice.append(dice(pr[b],msk[b]))
                    tr_f1.append(boundary_f1(pr[b],msk[b]))
                    tr_hd.append(hausdorff(pr[b],msk[b]))
        tr_hist.append([tr_loss/len(tr_loader),np.mean(tr_iou),np.mean(tr_dice),np.mean(tr_f1),np.mean([h for h in tr_hd if h<100])])
        
        model.eval()
        val_loss=0; val_iou=[]; val_dice=[]; val_f1=[]; val_hd=[]
        with torch.no_grad():
            for img, msk in tqdm(val_loader,desc=f"Epoch {ep+1}/{n_epoch} [Val]"):
                img, msk = img.to(device), msk.to(device)
                out = model(img); loss = criterion(out, msk); val_loss+=loss.item()
                pr = torch.sigmoid(out)
                for b in range(pr.shape[0]):
                    val_iou.append(iou(pr[b],msk[b]))
                    val_dice.append(dice(pr[b],msk[b]))
                    val_f1.append(boundary_f1(pr[b],msk[b]))
                    val_hd.append(hausdorff(pr[b],msk[b]))
        val_hist.append([val_loss/len(val_loader),np.mean(val_iou),np.mean(val_dice),np.mean(val_f1),np.mean([h for h in val_hd if h<100])])
        
        print(f"\n{'='*60}")
        print(f"Epoch {ep+1}/{n_epoch} Results:")
        print(f"Train Loss: {tr_hist[-1][0]:.4f} | Val Loss: {val_hist[-1][0]:.4f}")
        print(f"Val IoU: {val_hist[-1][1]:.3f} | Val Dice: {val_hist[-1][2]:.3f}")
        print(f"Val Boundary F1: {val_hist[-1][3]:.3f} | Val Hausdorff: {val_hist[-1][4]:.2f}")
        print(f"{'='*60}\n")
        
        if val_hist[-1][0] < best_val_loss:
            best_val_loss = val_hist[-1][0]
            torch.save(model.state_dict(), "best_unet_model.pth")
            print("✓ Saved best model checkpoint\n")
            
    return tr_hist, val_hist

def plot_curves(tr_hist, val_hist):
    tr_hist, val_hist = np.array(tr_hist), np.array(val_hist)
    mets = ['Loss','IoU','Dice','Boundary F1','Hausdorff']
    plt.figure(figsize=(15,5))
    for i in range(tr_hist.shape[1]):
        plt.subplot(1,5,i+1)
        plt.plot(tr_hist[:,i],label='Train',marker='o')
        plt.plot(val_hist[:,i],label='Val',marker='s')
        plt.title(mets[i],fontsize=12,fontweight='bold')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True,alpha=0.3)
    plt.tight_layout()
    os.makedirs('results',exist_ok=True)
    plt.savefig('results/training_curves.png',dpi=150)
    print("✓ Training curves saved to results/training_curves.png")
    plt.show()

def infer_visual(model, img_path, device, out_dir='results'):
    img = Image.open(img_path).convert('RGB').resize((256,256))
    img_t = transforms.ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.sigmoid(model(img_t)[0,0]).cpu().numpy()
    binary = (pred>0.5).astype(np.uint8)
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1); plt.imshow(img); plt.axis('off'); plt.title('Original Image',fontsize=14,fontweight='bold')
    plt.subplot(1,3,2); plt.imshow(pred,cmap='jet',vmin=0,vmax=1); plt.colorbar(); plt.axis('off'); plt.title('Probability Map',fontsize=14,fontweight='bold')
    plt.subplot(1,3,3); plt.imshow(img); plt.imshow(binary,alpha=0.5,cmap='Reds'); plt.axis('off'); plt.title('Segmentation Overlay',fontsize=14,fontweight='bold')
    plt.tight_layout(); os.makedirs(out_dir,exist_ok=True)
    out_path = f"{out_dir}/prediction_{os.path.basename(img_path)}"
    plt.savefig(out_path,dpi=150,bbox_inches='tight')
    print(f"✓ Saved prediction: {out_path}")
    plt.close()
    
if __name__=="__main__":
    print("\n" + "="*60)
    print("U-NET BINARY SEGMENTATION PIPELINE")
    print("="*60 + "\n")
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device.upper()}")
    if device == 'cpu':
        print("⚠ Running on CPU (training will be slower)\n")
    
    if not (os.path.exists('data/images') and os.path.exists('data/masks')):
        save_synth_data('data', n=80)
    else:
        print(f"✓ Found existing data in data/images and data/masks\n")
    
    full_ds = SegmDataset('data/images','data/masks',augment=True)
    print(f"Total samples: {len(full_ds)}")
    
    idxs = list(range(len(full_ds))); random.shuffle(idxs)
    split=int(len(full_ds)*0.8); tr_idxs=idxs[:split]; val_idxs=idxs[split:]
    train_ds = torch.utils.data.Subset(full_ds, tr_idxs)
    val_ds = torch.utils.data.Subset(full_ds, val_idxs)
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}\n")
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=4, num_workers=0)
 
    model = UNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: U-Net")
    print(f"Total parameters: {total_params:,}\n")
    
    pos_weight = calc_class_weight(train_loader)
    criterion = CombinedLoss(alpha=0.5, pos_weight=pos_weight, device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print(f"\nLoss: Combined (BCE + Dice) with pos_weight={pos_weight:.2f}")
    print(f"Optimizer: Adam (lr=1e-4)")
    print(f"Epochs: 15\n")
    
    print("="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    tr_hist, val_hist = train(model, train_loader, val_loader, n_epoch=15, device=device, criterion=criterion, optimizer=optimizer)
    

    plot_curves(tr_hist, val_hist)
    
    torch.save(model.state_dict(), "final_unet_model.pth")
    print("\n✓ Final model saved as final_unet_model.pth")

    print("\n" + "="*60)
    print("RUNNING INFERENCE ON SAMPLE IMAGES")
    print("="*60 + "\n")
    sample_imgs = glob.glob("data/images/*.png")[:5]
    for img in sample_imgs:
        infer_visual(model, img, device)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    
