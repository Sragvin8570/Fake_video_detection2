"""
UNet Deblurring Model
=====================
A U-Net architecture for image-to-image deblurring.

Architecture:
- Encoder: Downsampling with skip connections
- Bottleneck: Deep feature extraction
- Decoder: Upsampling with skip connections
- Output: Reconstructed sharp image
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive convolution layers with ReLU activation"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downsampling block with max pooling"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block with skip connections"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.upconv(x)
        
        # Handle size mismatch
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        # Concatenate skip connection
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNetDeblur(nn.Module):
    """
    U-Net for video deblurring
    
    Args:
        in_channels: Input image channels (default: 3 for RGB)
        out_channels: Output image channels (default: 3 for RGB)
        features: List of feature dimensions for each level
    """
    
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Initial convolution
        self.initial = DoubleConv(in_channels, features[0])
        
        # Encoder (downsampling path)
        for i in range(len(features) - 1):
            self.encoder.append(DownBlock(features[i], features[i + 1]))
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder (upsampling path)
        for i in range(len(features) - 1, 0, -1):
            self.decoder.append(UpBlock(features[i] * 2, features[i]))
        
        # Final upsampling
        self.decoder.append(UpBlock(features[0] * 2, features[0]))
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        # Store skip connections
        skip_connections = []
        
        # Initial conv
        x = self.initial(x)
        skip_connections.append(x)
        
        # Encoder
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]
        
        # Decoder with skip connections
        for idx, up in enumerate(self.decoder):
            x = up(x, skip_connections[idx])
        
        # Final output
        x = self.final_conv(x)
        
        return x


class UNetDeblurLight(nn.Module):
    """
    Lightweight U-Net for faster inference
    Fewer features, suitable for real-time or resource-constrained environments
    """
    
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        
        # Final output
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Decoder
        dec3 = self.up3(bottleneck)
        dec3 = torch.cat([enc3, dec3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([enc2, dec2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([enc1, dec1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final(dec1)


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the models
    print("Testing UNet Models...")
    print("=" * 60)
    
    # Test standard UNet
    model = UNetDeblur()
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    
    print(f"Standard UNet:")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters:   {count_parameters(model):,}")
    print()
    
    # Test lightweight UNet
    model_light = UNetDeblurLight()
    output_light = model_light(x)
    
    print(f"Lightweight UNet:")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output_light.shape}")
    print(f"  Parameters:   {count_parameters(model_light):,}")
    print()
    
    print("✓ All tests passed!")