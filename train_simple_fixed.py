#!/usr/bin/env python3
"""
Simple AV-sync model training on HAV-DF dataset
FIXED: Works with video_metadata.csv column names
"""

import os
import sys
import argparse
import numpy as np
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import cv2
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import MP_AViT
from backbone.select_backbone import select_backbone
from audio_process import AudioEncoder

class SimpleHAVDFDataset(Dataset):
    """Simple dataset that handles missing files gracefully"""
    
    def __init__(self, videos_dir, metadata_csv, sample_rate=16000, resize=224, num_frames=8):
        self.videos_dir = Path(videos_dir)
        self.sample_rate = sample_rate
        self.resize = resize
        self.num_frames = num_frames
        
        # Load metadata
        self.videos = []
        self.labels = []
        
        print(f"[INFO] Loading metadata from {metadata_csv}...")
        with open(metadata_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try both 'video_name' and 'filename' columns
                video_name = row.get('video_name') or row.get('filename')
                if not video_name:
                    continue
                    
                video_path = self.videos_dir / video_name
                
                # Only add if file exists
                if video_path.exists():
                    self.videos.append(str(video_path))
                    label = 1 if row['label'].upper() == 'FAKE' else 0
                    self.labels.append(label)
        
        print(f"[OK] Loaded {len(self.videos)} videos")
        fake_count = sum(self.labels)
        real_count = len(self.labels) - fake_count
        print(f"  FAKE: {fake_count}, REAL: {real_count}")
    
    def __len__(self):
        return len(self.videos)
    
    def load_video(self, video_path):
        """Load video frames"""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            frames = []
            for _ in range(self.num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (self.resize, self.resize))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            
            cap.release()
            
            # Pad if needed
            while len(frames) < self.num_frames:
                frames.append(frames[-1] if frames else np.zeros((self.resize, self.resize, 3)))
            
            frames = np.array(frames[:self.num_frames])
            
            # Convert to tensor: (num_frames, H, W, 3) -> (3, num_frames, H, W)
            # This is the format expected by 3D ResNet: (C, T, H, W)
            video = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
            
            return video
        except Exception as e:
            print(f"  Error loading video {video_path}: {e}")
            return None
    
    def load_audio(self, video_path):
        """Extract and load audio as mel-spectrogram"""
        try:
            import librosa
            import subprocess
            
            # Extract audio to wav
            wav_path = video_path.replace('.mp4', '_temp.wav')
            cmd = f'ffmpeg -threads 1 -loglevel error -y -i "{video_path}" -ac 1 -acodec pcm_s16le -ar {self.sample_rate} "{wav_path}"'
            subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if os.path.exists(wav_path):
                audio, _ = librosa.load(wav_path, sr=self.sample_rate)
                os.remove(wav_path)
                
                # Pad to match video length  
                target_len = self.num_frames * 2000
                if len(audio) < target_len:
                    audio = np.pad(audio, (0, target_len - len(audio)))
                else:
                    audio = audio[:target_len]
                
                # Convert to mel-spectrogram
                # This is what AudioEncoder expects
                mel_spec = librosa.feature.melspectrogram(
                    y=audio,
                    sr=self.sample_rate,
                    n_fft=512,
                    hop_length=160,
                    n_mels=80
                )
                
                # Convert to log scale
                mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Normalize to [0, 1]
                mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
                
                # Convert to tensor: (n_mels, time) -> add batch dim later
                return torch.from_numpy(mel_spec).float()
            else:
                # Return silence spectrogram if extraction fails
                mel_spec = np.zeros((80, 100))  # 80 mel bins, 100 time steps
                return torch.from_numpy(mel_spec).float()
        except Exception as e:
            print(f"  Error loading audio {video_path}: {e}")
            mel_spec = np.zeros((80, 100))
            return torch.from_numpy(mel_spec).float()
    
    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]
        
        video = self.load_video(video_path)
        audio = self.load_audio(video_path)
        
        # If loading failed, return dummy data
        # Shape: (3, num_frames, H, W) for 3D ResNet
        if video is None:
            video = torch.zeros(3, self.num_frames, self.resize, self.resize)
        
        return {
            'video': video,
            'audio': audio,
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_data in pbar:
        try:
            # Handle dictionary batch format
            if isinstance(batch_data, dict):
                video = batch_data['video'].to(device)
                audio = batch_data['audio'].to(device)
                labels = batch_data['label'].to(device).float()
                if audio.dim() == 2:
                    audio = audio.unsqueeze(0).unsqueeze(1)
                elif audio.dim() == 3:
                    audio = audio.unsqueeze(1)
                elif audio.dim() == 4 and audio.shape[1] != 1:
                    audio = audio.mean(dim=1, keepdim=True)
            else:
                video, audio, labels = batch_data
                video = video.to(device)
                audio = audio.to(device)
                labels = labels.to(device).float()
                if audio.dim() == 2:
                    audio = audio.unsqueeze(0).unsqueeze(1)
                elif audio.dim() == 3:
                    audio = audio.unsqueeze(1)
                elif audio.dim() == 4 and audio.shape[1] != 1:
                    audio = audio.mean(dim=1, keepdim=True)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(video, audio)
            
            # Handle output shape
            if outputs.dim() > 1:
                outputs = outputs.squeeze()
            
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).long()
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.0*correct/total:.2f}%'})
        except Exception as e:
            print(f"  Error in batch: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    avg_loss = total_loss / max(len(train_loader), 1)
    accuracy = 100.0 * correct / max(total, 1)
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train/Eval AV-sync model on HAV-DF')
    parser.add_argument('--train_videos', default='HAV-DF/train_videos', help='Path to training videos')
    parser.add_argument('--metadata_csv', default='HAV-DF/video_metadata.csv', help='Path to metadata CSV')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--save_dir', default='checkpoints/havdf', help='Directory to save models')
    parser.add_argument('--pretrained', default='sync_model.pth', help='Pretrained model path')
    parser.add_argument('--eval_only', action='store_true', help='Skip training and run evaluation')
    parser.add_argument('--eval_videos', default=None, help='Directory of videos to evaluate (filters metadata to those files)')
    parser.add_argument('--make_report', action='store_true', help='Generate benchmark report with and without deblur')
    parser.add_argument('--deblur_amount', type=float, default=1.0, help='Unsharp mask strength for deblur')
    
    args = parser.parse_args()
    
    print("="*70)
    print("SIMPLE HAV-DF TRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Videos: {args.train_videos}")
    print(f"  Metadata: {args.metadata_csv}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Device: {args.device}")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n[OK] Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset
    print("\n[1/4] Loading training dataset...")
    dataset = SimpleHAVDFDataset(args.train_videos, args.metadata_csv)
    
    if len(dataset) == 0:
        print("[FAIL] No training data found!")
        print(f"  Check that videos exist in: {args.train_videos}")
        print(f"  Check that CSV exists at: {args.metadata_csv}")
        return
    
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"  Batches per epoch: {len(train_loader)}")
    
    # Initialize model
    print("\n[2/4] Initializing model...")
    
    # Use your existing model architecture
    from backbone.select_backbone import select_backbone
    from audio_process import AudioEncoder
    
    # Visual encoder (ResNet-18)
    vis_enc, _ = select_backbone(network='r18')
    
    # Audio encoder
    aud_enc = AudioEncoder()
    
    # Transformer
    transformer = MP_AViT(
        image_size=14, 
        patch_size=0, 
        num_classes=1,
        dim=512, 
        depth=3, 
        heads=4, 
        mlp_dim=512,
        dim_head=128, 
        dropout=0.1, 
        emb_dropout=0.1,
        max_visual_len=dataset.num_frames, 
        max_audio_len=64
    )
    
    # Create network combining all components
    class AVSyncNetwork(nn.Module):
        def __init__(self, vis_enc, aud_enc, transformer):
            super().__init__()
            self.vis_enc = vis_enc
            self.aud_enc = aud_enc
            self.transformer = transformer
        
        def forward(self, video, audio):
            vid_emb = self.vis_enc(video)
            aud_emb = self.aud_enc(audio)
            output = self.transformer(vid_emb, aud_emb)
            return output
    
    model = AVSyncNetwork(vis_enc, aud_enc, transformer)
    
    if os.path.exists(args.pretrained):
        print(f"  Loading pretrained weights from {args.pretrained}...")
        checkpoint = torch.load(args.pretrained, map_location='cpu', weights_only=False)
        current_sd = model.state_dict()
        filtered = {}
        for k, v in checkpoint.items():
            if k in current_sd and hasattr(v, 'shape') and hasattr(current_sd[k], 'shape'):
                if v.shape == current_sd[k].shape:
                    filtered[k] = v
            elif k in current_sd and not hasattr(v, 'shape'):
                filtered[k] = v
        current_sd.update(filtered)
        model.load_state_dict(current_sd, strict=False)
        print(f"  [OK] Loaded pretrained weights (shape-mismatched params skipped)")
    else:
        print(f"  [WARNING] No pretrained weights found at {args.pretrained}")
        print(f"  Training from scratch...")
    
    model = model.to(device)
    
    # Loss and optimizer
    print("\n[3/4] Setting up training...")
    criterion = nn.BCEWithLogitsLoss()  # Binary classification
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Training loop
    print("\n[4/4] Starting training...")
    print("="*70)
    
    best_loss = float('inf')
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-"*70)
        
        avg_loss, accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        
        print(f"\nResults:")
        print(f"  Loss:     {avg_loss:.6f}")
        print(f"  Accuracy: {accuracy:.2f}%")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  Saved: {checkpoint_path}")
        
        # Save best model
        if accuracy > best_acc:
            best_acc = accuracy
            best_loss = avg_loss
            patience_counter = 0
            best_path = os.path.join(args.save_dir, 'sync_model_havdf_best.pth')
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ New best accuracy: {best_acc:.2f}%")
            print(f"  ✓ Saved best model: {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"\n[INFO] Early stopping after {epoch+1} epochs (no improvement)")
                break
        
        scheduler.step(avg_loss)
    
    print("\n" + "="*70)
    print("[OK] Training complete!")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Best model saved to: {os.path.join(args.save_dir, 'sync_model_havdf_best.pth')}")
    print("="*70)

    # Optional evaluation
    if args.eval_only or args.eval_videos or args.make_report:
        eval_dir = args.eval_videos or args.train_videos
        print(f"\n[Eval] Evaluating on: {eval_dir}")
        # Build filtered list from metadata for files present in eval_dir
        import csv
        eval_dataset_videos = []
        eval_dataset_labels = []
        with open(args.metadata_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_name = row.get('video_name') or row.get('filename')
                if not video_name:
                    continue
                video_path = os.path.join(eval_dir, video_name)
                if os.path.exists(video_path):
                    eval_dataset_videos.append(video_path)
                    label = 1 if row['label'].upper() == 'FAKE' else 0
                    eval_dataset_labels.append(label)
        if len(eval_dataset_videos) == 0:
            print("[Eval] No labeled videos found in evaluation directory; skipping metrics.")
            return
        # Simple evaluation loop
        def run_eval(deblur=False):
            model.eval()
            correct = 0
            total = 0
            batch = args.batch_size
            for i in tqdm(range(0, len(eval_dataset_videos), batch), desc=("Eval-Deblur" if deblur else "Eval")):
                paths = eval_dataset_videos[i:i+batch]
                labels = torch.tensor(eval_dataset_labels[i:i+batch], dtype=torch.float32, device=device)
                videos = []
                audios = []
                for p in paths:
                    v = dataset.load_video(p)
                    a = dataset.load_audio(p)
                    if v is None:
                        v = torch.zeros(3, dataset.num_frames, dataset.resize, dataset.resize)
                    if deblur:
                        vt = v.permute(1, 2, 3, 0).cpu().numpy()
                        out = []
                        for t in range(vt.shape[0]):
                            im = (vt[t] * 255.0).astype('uint8')
                            blur = cv2.GaussianBlur(im, (0, 0), 1.0)
                            sharp = cv2.addWeighted(im, 1.0 + args.deblur_amount, blur, -args.deblur_amount, 0)
                            out.append(sharp.astype('float32') / 255.0)
                        vt2 = torch.from_numpy(np.stack(out)).permute(3, 0, 1, 2)
                        v = vt2.float()
                    videos.append(v)
                    audios.append(a)
                video_tensor = torch.stack(videos).to(device)
                audio_tensor = torch.stack(audios).to(device)
                audio_tensor = audio_tensor.unsqueeze(1) if audio_tensor.dim()==3 else audio_tensor.unsqueeze(0).unsqueeze(1)
                with torch.no_grad():
                    outputs = model(video_tensor, audio_tensor)
                    outputs = outputs.squeeze()
                    preds = (torch.sigmoid(outputs) > 0.5).long()
                total += labels.size(0)
                correct += (preds.cpu() == labels.cpu().long()).sum().item()
            return 100.0 * correct / total
        acc_base = run_eval(deblur=False)
        print(f"[Eval] Accuracy (no deblur) on {len(eval_dataset_videos)} videos: {acc_base:.2f}%")
        if args.make_report:
            acc_deblur = run_eval(deblur=True)
            print(f"[Eval] Accuracy (deblur) on {len(eval_dataset_videos)} videos: {acc_deblur:.2f}%")
            out_dir = os.path.join(os.path.dirname(__file__), 'havdf_results')
            os.makedirs(out_dir, exist_ok=True)
            report_path = os.path.join(out_dir, 'benchmark_report.txt')
            with open(report_path, 'w') as rf:
                rf.write("==========================================================================================\n")
                rf.write("HAV-DF Benchmark Report (AV-Sync Model)\n")
                rf.write("==========================================================================================\n")
                rf.write(f"Videos evaluated: {len(eval_dataset_videos)}\n")
                rf.write(f"Accuracy (no deblur): {acc_base:.2f}%\n")
                rf.write(f"Accuracy (deblur): {acc_deblur:.2f}%\n")
                rf.write(f"Deblur amount: {args.deblur_amount}\n")
            print(f"[Eval] Report written to {report_path}")


if __name__ == '__main__':
    main()
