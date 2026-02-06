"""
Video Deblurring Training
==========================
Train UNet model for video deblurring.

Dataset Structure:
    data/
    ├── blurred/
    │   ├── video001_frame_001.png
    │   ├── video001_frame_002.png
    │   └── ...
    └── sharp/
        ├── video001_frame_001.png
        ├── video001_frame_002.png
        └── ...

Usage:
    python train.py --data_dir data/ --epochs 50 --batch_size 16
    python train.py --data_dir data/ --epochs 50 --batch_size 16 --light
"""

import os
import argparse
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import model
try:
    from model import UNetDeblur, UNetDeblurLight
except ImportError:
    from deblurring.model import UNetDeblur, UNetDeblurLight


class PairedImageDataset(Dataset):
    """
    Dataset for paired blurred/sharp images
    
    Expects directory structure:
        data_dir/
            blurred/
            sharp/
    """
    
    def __init__(self, data_dir, transform=None):
        self.blurred_dir = os.path.join(data_dir, 'blurred')
        self.sharp_dir = os.path.join(data_dir, 'sharp')
        
        # Get list of files
        self.blurred_files = sorted(os.listdir(self.blurred_dir))
        self.sharp_files = sorted(os.listdir(self.sharp_dir))
        
        # Validate
        assert len(self.blurred_files) == len(self.sharp_files), \
            f"Mismatch: {len(self.blurred_files)} blurred vs {len(self.sharp_files)} sharp"
        
        self.transform = transform
        
        print(f"Dataset loaded: {len(self.blurred_files)} image pairs")
    
    def __len__(self):
        return len(self.blurred_files)
    
    def __getitem__(self, idx):
        # Load images
        blurred_path = os.path.join(self.blurred_dir, self.blurred_files[idx])
        sharp_path = os.path.join(self.sharp_dir, self.sharp_files[idx])
        
        blurred = Image.open(blurred_path).convert('RGB')
        sharp = Image.open(sharp_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            blurred = self.transform(blurred)
            sharp = self.transform(sharp)
        
        return blurred, sharp


class CombinedLoss(nn.Module):
    """
    Combined loss function:
    - L1 Loss (MAE) for pixel-wise reconstruction
    - Perceptual Loss (optional, using MSE as proxy)
    """
    
    def __init__(self, l1_weight=1.0, mse_weight=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
    
    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        mse_loss = self.mse(pred, target)
        return self.l1_weight * l1_loss + self.mse_weight * mse_loss


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for blurred, sharp in pbar:
        blurred = blurred.to(device)
        sharp = sharp.to(device)
        
        # Forward pass
        output = model(blurred)
        loss = criterion(output, sharp)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for blurred, sharp in dataloader:
            blurred = blurred.to(device)
            sharp = sharp.to(device)
            
            output = model(blurred)
            loss = criterion(output, sharp)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def save_sample_images(model, dataloader, device, save_path, num_samples=4):
    """Save sample predictions for visualization"""
    model.eval()
    
    blurred_batch, sharp_batch = next(iter(dataloader))
    blurred_batch = blurred_batch[:num_samples].to(device)
    sharp_batch = sharp_batch[:num_samples]
    
    with torch.no_grad():
        output_batch = model(blurred_batch).cpu()
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))
    
    for i in range(num_samples):
        # Blurred input
        axes[i, 0].imshow(blurred_batch[i].cpu().permute(1, 2, 0))
        axes[i, 0].set_title('Blurred Input')
        axes[i, 0].axis('off')
        
        # Deblurred output
        axes[i, 1].imshow(torch.clamp(output_batch[i], 0, 1).permute(1, 2, 0))
        axes[i, 1].set_title('Deblurred Output')
        axes[i, 1].axis('off')
        
        # Sharp ground truth
        axes[i, 2].imshow(sharp_batch[i].permute(1, 2, 0))
        axes[i, 2].set_title('Sharp Ground Truth')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train(args):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = PairedImageDataset(args.data_dir, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize model
    print("Initializing model...")
    if args.light:
        model = UNetDeblurLight()
        print("Using Lightweight UNet")
    else:
        model = UNetDeblur()
        print("Using Standard UNet")
    
    model.to(device)
    
    # Loss and optimizer
    criterion = CombinedLoss(l1_weight=1.0, mse_weight=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    best_loss = float('inf')
    train_losses = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.6f}")
        
        # Learning rate scheduling
        scheduler.step(train_loss)
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            best_path = output_dir / "best_model.pth"
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Best model saved: {best_path} (loss: {best_loss:.6f})")
        
        # Save sample images
        if epoch % args.sample_every == 0:
            sample_path = output_dir / f"samples_epoch_{epoch}.png"
            save_sample_images(model, train_loader, device, sample_path)
            print(f"  ✓ Sample images saved: {sample_path}")
    
    # Save final model
    final_path = output_dir / "final_model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\n✓ Final model saved: {final_path}")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'training_curve.png')
    plt.close()
    print(f"✓ Training curve saved: {output_dir / 'training_curve.png'}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Models saved in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Video Deblurring Model')
    
    # Data
    parser.add_argument('--data_dir', required=True, help='Path to dataset directory')
    parser.add_argument('--img_size', type=int, default=256, help='Image size (default: 256)')
    
    # Model
    parser.add_argument('--light', action='store_true', help='Use lightweight model')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers (default: 4)')
    
    # Output
    parser.add_argument('--output_dir', default='./checkpoints', help='Output directory')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--sample_every', type=int, default=5, help='Save samples every N epochs')
    
    args = parser.parse_args()
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    if not os.path.exists(os.path.join(args.data_dir, 'blurred')):
        raise FileNotFoundError(f"'blurred' folder not found in {args.data_dir}")
    
    if not os.path.exists(os.path.join(args.data_dir, 'sharp')):
        raise FileNotFoundError(f"'sharp' folder not found in {args.data_dir}")
    
    train(args)


if __name__ == '__main__':
    main()