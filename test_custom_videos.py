#!/usr/bin/env python3
"""
Custom Video Folder Benchmark Testing Script
Works exactly like havdf_flat_structure.py but for any folder of test videos
Supports optional metadata CSV for ground truth labels
"""

import os
import sys
import subprocess
import csv
import time
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse

class CustomFolderBenchmarkTester:
    def __init__(self, videos_folder, output_dir, deblur_checkpoint, detection_script, 
                 deblur_script, metadata_file=None, device='cuda'):
        self.videos_folder = videos_folder
        self.output_dir = output_dir
        self.deblur_checkpoint = deblur_checkpoint
        self.detection_script = detection_script
        self.deblur_script = deblur_script
        self.metadata_file = metadata_file
        self.device = device
        
        self.python_exe = sys.executable
        print(f"Using Python: {self.python_exe}")
        
        # Create output directories
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/deblurred_videos").mkdir(exist_ok=True)
        Path(f"{output_dir}/original_results").mkdir(exist_ok=True)
        Path(f"{output_dir}/deblurred_results").mkdir(exist_ok=True)
        
        self.results = []
        self.ground_truth = {}
    
    def load_ground_truth(self):
        """Load ground truth labels from metadata CSV if provided"""
        if not self.metadata_file or not os.path.exists(self.metadata_file):
            print("\n[INFO] No metadata file provided - testing without ground truth")
            return False
        
        try:
            print(f"\n[INFO] Loading metadata from: {self.metadata_file}")
            df = pd.read_csv(self.metadata_file)
            
            print(f"  Columns found: {list(df.columns)}")
            
            # Identify filename and label columns
            filename_col = None
            label_col = None
            
            for col in ['filename', 'video', 'file', 'video_name', 'name', 'path']:
                if col in df.columns:
                    filename_col = col
                    break
            
            for col in ['label', 'class', 'type', 'category', 'ground_truth']:
                if col in df.columns:
                    label_col = col
                    break
            
            if not filename_col or not label_col:
                print(f"  ⚠️  Could not identify columns automatically")
                print(f"  Please ensure CSV has 'filename' and 'label' columns")
                return False
            
            print(f"  Using: filename='{filename_col}', label='{label_col}'")
            
            # Load labels
            for _, row in df.iterrows():
                video_name = str(row[filename_col])
                label = str(row[label_col]).upper().strip()
                
                # Normalize labels
                if label in ['FAKE', '1', 'DEEPFAKE', 'SYNTHETIC']:
                    label = 'FAKE'
                elif label in ['REAL', '0', 'GENUINE', 'AUTHENTIC']:
                    label = 'REAL'
                
                if not video_name.endswith('.mp4'):
                    video_name = video_name + '.mp4'
                
                self.ground_truth[video_name] = label
            
            print(f"  ✓ Loaded {len(self.ground_truth)} ground truth labels")
            
            # Count labels
            fake_count = sum(1 for v in self.ground_truth.values() if v == 'FAKE')
            real_count = sum(1 for v in self.ground_truth.values() if v == 'REAL')
            print(f"    FAKE: {fake_count}, REAL: {real_count}")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error loading metadata: {e}")
            return False
    
    def get_video_files(self):
        """Get all video files from folder"""
        print(f"\n[INFO] Scanning for videos in: {self.videos_folder}")
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
        videos = []
        
        for ext in video_extensions:
            videos.extend(Path(self.videos_folder).glob(f'*{ext}'))
        
        videos = sorted(list(set(videos)))
        print(f"  ✓ Found {len(videos)} video files")
        
        return videos
    
    def run_deblurring(self, input_video, output_video):
        """Run deblurring on a video"""
        try:
            cmd = [
                self.python_exe, self.deblur_script,
                "--src", str(input_video),
                "--dst", str(output_video),
                "--ckpt", self.deblur_checkpoint,
                "--device", self.device
            ]
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=os.environ.copy())
            deblur_time = time.time() - start_time
            
            return True, deblur_time, None
        except Exception as e:
            return False, 0, str(e)
    
    def run_detection(self, video_path, output_subdir):
        """Run detection on a video"""
        try:
            output_path = os.path.join(self.output_dir, output_subdir)
            Path(output_path).mkdir(parents=True, exist_ok=True)
            Path(f"{output_path}/visualizations").mkdir(exist_ok=True)
            
            cmd = [
                self.python_exe, self.detection_script,
                "--test_video_path", str(video_path),
                "--output_dir", output_path,
                "--device", self.device
            ]
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=os.environ.copy())
            detection_time = time.time() - start_time
            
            # Read results
            summary_path = os.path.join(output_path, "detection_summary.csv")
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        parts = lines[1].strip().split(',')
                        score = float(parts[1])
                        label = parts[2]
                        return True, detection_time, score, label, None
            
            return False, 0, None, None, "Could not read results"
            
        except Exception as e:
            return False, 0, None, None, str(e)
    
    def process_video(self, video_path, video_index, total_videos):
        """Process a single video with both original and deblurred detection"""
        video_name = video_path.name
        ground_truth_label = self.ground_truth.get(video_name, 'UNKNOWN')
        
        print(f"\n{'='*70}")
        print(f"Processing [{video_index}/{total_videos}]: {video_name}")
        if ground_truth_label != 'UNKNOWN':
            print(f"Ground Truth: {ground_truth_label}")
        print(f"{'='*70}")
        
        result = {
            'video_name': video_name,
            'video_path': str(video_path),
            'ground_truth': ground_truth_label,
            'timestamp': datetime.now().isoformat()
        }
        
        # Step 1: Test ORIGINAL video
        print(f"\n[1/3] Testing ORIGINAL video...")
        success, det_time, score, label, error = self.run_detection(
            video_path, 
            f"original_results/{video_name}_original"
        )
        
        if success:
            result['original_detection_time'] = round(det_time, 2)
            result['original_score'] = score
            result['original_label'] = label
            result['original_status'] = 'success'
            result['original_correct'] = (label == ground_truth_label) if ground_truth_label != 'UNKNOWN' else None
            
            print(f"  ✓ Original: {label} (Score: {score:.4f}) in {det_time:.2f}s")
            if ground_truth_label != 'UNKNOWN':
                correct = '✓' if label == ground_truth_label else '✗'
                print(f"  {correct} Prediction: {label} vs Ground Truth: {ground_truth_label}")
        else:
            result['original_status'] = 'failed'
            result['original_error'] = error
            print(f"  ✗ Original detection failed: {error}")
        
        # Step 2: Run DEBLURRING
        print(f"\n[2/3] Applying DEBLURRING...")
        deblurred_video = os.path.join(
            self.output_dir, 
            "deblurred_videos", 
            f"deblurred_{video_name}"
        )
        
        success, deblur_time, error = self.run_deblurring(video_path, deblurred_video)
        
        if success:
            result['deblur_time'] = round(deblur_time, 2)
            result['deblur_status'] = 'success'
            print(f"  ✓ Deblurring completed in {deblur_time:.2f}s")
            
            # Step 3: Test DEBLURRED video
            print(f"\n[3/3] Testing DEBLURRED video...")
            success, det_time, score, label, error = self.run_detection(
                deblurred_video,
                f"deblurred_results/{video_name}_deblurred"
            )
            
            if success:
                result['deblurred_detection_time'] = round(det_time, 2)
                result['deblurred_score'] = score
                result['deblurred_label'] = label
                result['deblurred_status'] = 'success'
                result['deblurred_correct'] = (label == ground_truth_label) if ground_truth_label != 'UNKNOWN' else None
                
                print(f"  ✓ Deblurred: {label} (Score: {score:.4f}) in {det_time:.2f}s")
                if ground_truth_label != 'UNKNOWN':
                    correct = '✓' if label == ground_truth_label else '✗'
                    print(f"  {correct} Prediction: {label} vs Ground Truth: {ground_truth_label}")
                
                # Calculate improvements
                if result.get('original_score') is not None:
                    score_diff = score - result['original_score']
                    result['score_improvement'] = round(score_diff, 4)
                    result['labels_match'] = (label == result['original_label'])
                    
                    # Check accuracy improvement
                    if ground_truth_label != 'UNKNOWN':
                        orig_correct = result.get('original_correct', False)
                        deblur_correct = result.get('deblurred_correct', False)
                        
                        if not orig_correct and deblur_correct:
                            result['accuracy_improvement'] = 'FIXED'
                            print(f"  🎯 Deblurring FIXED incorrect prediction!")
                        elif orig_correct and not deblur_correct:
                            result['accuracy_improvement'] = 'BROKE'
                            print(f"  ⚠️  Deblurring BROKE correct prediction!")
                        else:
                            result['accuracy_improvement'] = 'SAME'
                    
                    print(f"\n  📊 Comparison:")
                    print(f"     Score change: {score_diff:+.4f} ({score_diff*100:+.2f}%)")
            else:
                result['deblurred_status'] = 'failed'
                result['deblurred_error'] = error
                print(f"  ✗ Deblurred detection failed: {error}")
        else:
            result['deblur_status'] = 'failed'
            result['deblur_error'] = error
            print(f"  ✗ Deblurring failed: {error}")
        
        return result
    
    def run_benchmark(self):
        """Run benchmark on all videos in folder"""
        
        # Get videos
        videos = self.get_video_files()
        
        if not videos:
            print(f"\n❌ No videos found in {self.videos_folder}")
            return
        
        # Load ground truth if available
        has_labels = self.load_ground_truth()
        
        print(f"\n{'='*70}")
        print(f"CUSTOM FOLDER BENCHMARK TEST")
        print(f"{'='*70}")
        print(f"Videos folder: {self.videos_folder}")
        print(f"Output directory: {self.output_dir}")
        print(f"Total videos: {len(videos)}")
        print(f"Ground truth labels: {len(self.ground_truth) if has_labels else 0}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")
        
        # Process each video
        for idx, video_path in enumerate(videos, 1):
            try:
                result = self.process_video(video_path, idx, len(videos))
                self.results.append(result)
                
                # Save intermediate results every 5 videos
                if idx % 5 == 0:
                    self.save_results(intermediate=True)
                    print(f"\n✓ Saved intermediate results ({idx}/{len(videos)} videos)")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user!")
                print("Saving partial results...")
                self.save_results(intermediate=True)
                sys.exit(0)
            except Exception as e:
                print(f"\n  ✗ Error processing {video_path.name}: {e}")
                self.results.append({
                    'video_name': video_path.name,
                    'ground_truth': self.ground_truth.get(video_path.name, 'UNKNOWN'),
                    'error': str(e),
                    'status': 'failed'
                })
        
        # Save final results
        self.save_results()
        self.generate_report()
    
    def save_results(self, intermediate=False):
        """Save detailed results to CSV"""
        suffix = "_intermediate" if intermediate else ""
        csv_path = os.path.join(self.output_dir, f"benchmark_results{suffix}.csv")
        
        fieldnames = [
            'video_name', 'ground_truth', 'video_path', 'timestamp',
            'original_status', 'original_score', 'original_label', 'original_correct', 'original_detection_time', 'original_error',
            'deblur_status', 'deblur_time', 'deblur_error',
            'deblurred_status', 'deblurred_score', 'deblurred_label', 'deblurred_correct', 'deblurred_detection_time', 'deblurred_error',
            'score_improvement', 'labels_match', 'accuracy_improvement'
        ]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(result)
        
        if not intermediate:
            print(f"\n✓ Detailed results saved to: {csv_path}")
    
    def generate_report(self):
        """Generate benchmark report"""
        report_path = os.path.join(self.output_dir, "benchmark_report.txt")
        
        total_videos = len(self.results)
        successful_original = sum(1 for r in self.results if r.get('original_status') == 'success')
        successful_deblurred = sum(1 for r in self.results if r.get('deblurred_status') == 'success')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HAV-DF Benchmark Report (AV-Sync Model)\n")
            f.write("="*80 + "\n")
            f.write(f"Videos evaluated: {total_videos}\n")
            
            # Calculate accuracy if ground truth available
            if any(r.get('ground_truth') != 'UNKNOWN' for r in self.results):
                orig_correct = sum(1 for r in self.results if r.get('original_correct') == True)
                orig_total = sum(1 for r in self.results if r.get('original_correct') is not None)
                
                deblur_correct = sum(1 for r in self.results if r.get('deblurred_correct') == True)
                deblur_total = sum(1 for r in self.results if r.get('deblurred_correct') is not None)
                
                if orig_total > 0:
                    orig_accuracy = orig_correct / orig_total * 100
                    f.write(f"Accuracy (no deblur): {orig_accuracy:.2f}%\n")
                
                if deblur_total > 0:
                    deblur_accuracy = deblur_correct / deblur_total * 100
                    f.write(f"Accuracy (deblur): {deblur_accuracy:.2f}%\n")
            else:
                f.write("Accuracy: N/A (no ground truth)\n")
            
            # Score improvements
            improvements = [r['score_improvement'] for r in self.results if 'score_improvement' in r]
            if improvements:
                avg_improvement = sum(improvements) / len(improvements)
                f.write(f"Average score improvement: {avg_improvement:+.4f}\n")
        
        print(f"✓ Report saved to: {report_path}")
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"BENCHMARK SUMMARY")
        print(f"{'='*70}\n")
        print(f"Total videos: {total_videos}")
        print(f"Successful tests: {successful_deblurred}/{total_videos}")
        
        if any(r.get('ground_truth') != 'UNKNOWN' for r in self.results):
            orig_correct = sum(1 for r in self.results if r.get('original_correct') == True)
            orig_total = sum(1 for r in self.results if r.get('original_correct') is not None)
            deblur_correct = sum(1 for r in self.results if r.get('deblurred_correct') == True)
            deblur_total = sum(1 for r in self.results if r.get('deblurred_correct') is not None)
            
            if orig_total > 0 and deblur_total > 0:
                orig_accuracy = orig_correct / orig_total * 100
                deblur_accuracy = deblur_correct / deblur_total * 100
                print(f"\nAccuracy:")
                print(f"  No deblur: {orig_accuracy:.2f}%")
                print(f"  With deblur: {deblur_accuracy:.2f}%")
                print(f"  Improvement: {deblur_accuracy - orig_accuracy:+.2f}%")
        
        print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Benchmark testing for any folder of videos')
    parser.add_argument('--videos_folder', type=str, required=True, help='Folder containing test videos')
    parser.add_argument('--output_dir', type=str, default='benchmark_results', help='Output directory')
    parser.add_argument('--metadata_file', type=str, help='Optional CSV with ground truth labels')
    parser.add_argument('--deblur_checkpoint', type=str, required=True, help='Deblur checkpoint path')
    parser.add_argument('--detection_script', type=str, required=True, help='Detection script path')
    parser.add_argument('--deblur_script', type=str, required=True, help='Deblur script path')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.videos_folder):
        print(f"Error: Videos folder not found: {args.videos_folder}")
        sys.exit(1)
    
    if not os.path.exists(args.deblur_checkpoint):
        print(f"Error: Deblur checkpoint not found: {args.deblur_checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(args.detection_script):
        print(f"Error: Detection script not found: {args.detection_script}")
        sys.exit(1)
    
    if not os.path.exists(args.deblur_script):
        print(f"Error: Deblur script not found: {args.deblur_script}")
        sys.exit(1)
    
    # Create tester and run
    tester = CustomFolderBenchmarkTester(
        videos_folder=args.videos_folder,
        output_dir=args.output_dir,
        deblur_checkpoint=args.deblur_checkpoint,
        detection_script=args.detection_script,
        deblur_script=args.deblur_script,
        metadata_file=args.metadata_file,
        device=args.device
    )
    
    tester.run_benchmark()


if __name__ == "__main__":
    main()