"""
Video Deblurring Inference Script with Audio Preservation
Processes video frame-by-frame using trained UNet model and preserves audio
"""

import cv2
import torch
from torchvision import transforms
from PIL import Image
import argparse
from tqdm import tqdm
import os
import sys
import subprocess
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import model
from model import UNetDeblur, UNetDeblurLight

def deblur_video(input_path, output_path, checkpoint_path, device='cuda', resize_dim=256, use_light=False):
    """
    Deblur a video using trained model and preserve audio
    
    Args:
        input_path: Path to input video
        output_path: Path to save deblurred video
        checkpoint_path: Path to model checkpoint
        device: 'cuda' or 'cpu'
        resize_dim: Processing dimension (default 256)
        use_light: Use lightweight model (default False - uses full UNet)
    """
    
    print(f"Loading model from {checkpoint_path}...")
    
    # Load model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    if use_light:
        model = UNetDeblurLight().to(device)
        print("Using Lightweight UNet")
    else:
        model = UNetDeblur().to(device)
        print("Using Standard UNet")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
            
        print(f"[OK] Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False
    
    model.eval()
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Create temporary output path (without audio)
    temp_output = output_path.replace('.mp4', '_temp_no_audio.mp4')
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Cannot create output video {temp_output}")
        return False
    
    # Transform for preprocessing
    transform = transforms.Compose([
        transforms.Resize((resize_dim, resize_dim)),
        transforms.ToTensor(),
    ])
    
    print(f"Processing video frames...")
    
    # Process frame by frame
    with torch.no_grad():
        for _ in tqdm(range(total_frames), desc="Deblurring"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = transform(frame_pil).unsqueeze(0).to(device)
            
            # Run through model
            output_tensor = model(frame_tensor)
            output_tensor = torch.clamp(output_tensor, 0, 1)
            
            # Postprocess
            output_np = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
            output_np = (output_np * 255).astype('uint8')
            
            # Resize back to original dimensions
            output_resized = cv2.resize(output_np, (width, height))
            output_bgr = cv2.cvtColor(output_resized, cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(output_bgr)
    
    cap.release()
    out.release()
    
    print(f"[OK] Video frames processed")
    
    # Now copy audio from original video using ffmpeg
    print(f"Copying audio track from original video...")
    
    audio_copied = False
    
    try:
        # Check if ffmpeg is available
        ffmpeg_check = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True
        )
        
        if ffmpeg_check.returncode == 0:
            # Use ffmpeg to combine deblurred video with original audio
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', temp_output,      # Deblurred video (no audio)
                '-i', input_path,       # Original video (with audio)
                '-c:v', 'copy',         # Copy video without re-encoding
                '-c:a', 'aac',          # Encode audio as AAC
                '-map', '0:v:0',        # Take video from first input
                '-map', '1:a:0?',       # Take audio from second input (? makes it optional)
                '-shortest',            # Match shortest stream
                '-y',                   # Overwrite output file
                output_path
            ]
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"[OK] Audio track copied successfully")
            audio_copied = True
            
            # Remove temporary file
            try:
                os.remove(temp_output)
            except:
                pass
                
    except FileNotFoundError:
        print(f"[WARNING] FFmpeg not found in system PATH")
        print(f"   Install FFmpeg to preserve audio: https://ffmpeg.org/download.html")
        
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Could not copy audio track")
        print(f"   FFmpeg error: {e.stderr}")
        
    except Exception as e:
        print(f"[WARNING] Error during audio copying: {e}")
    
    # If audio copying failed, just rename temp file to output
    if not audio_copied:
        print(f"   Saving video without audio...")
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
            shutil.move(temp_output, output_path)
        except Exception as e:
            print(f"   Error moving file: {e}")
            if os.path.exists(temp_output):
                os.remove(temp_output)
            return False
    
    print(f"[OK] Deblurred video saved to: {output_path}")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deblur video using trained model')
    parser.add_argument('--src', required=True, help='Input video path')
    parser.add_argument('--dst', required=True, help='Output video path')
    parser.add_argument('--ckpt', required=True, help='Model checkpoint path')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--resize', type=int, default=256, help='Processing dimension')
    parser.add_argument('--light', action='store_true', help='Use lightweight model (use if checkpoint trained with --light flag)')
    
    args = parser.parse_args()
    
    success = deblur_video(
        args.src, 
        args.dst, 
        args.ckpt, 
        device=args.device, 
        resize_dim=args.resize,
        use_light=args.light
    )
    
    if success:
        print("\n[OK] Deblurring completed successfully!")
    else:
        print("\n[FAIL] Deblurring failed!")
        exit(1)