import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import json
import pandas as pd
from fake_celeb_dataset import FakeAVceleb
from model import MP_AViT, MP_av_feature_AViT
from subprocess import call
from backbone.select_backbone import select_backbone
from torch.optim import Adam
from config_deepfake import load_opts, save_opts
from torch.utils.data import DataLoader
import argparse
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data.sampler import RandomSampler, Sampler
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Optional, Union
from audio_process import AudioEncoder
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import math
import h5py
import os
import glob
import time
from deep_fake_data import prepocess_video
import logging
from load_audio import wav2filterbanks, wave2input
from torch.utils.tensorboard import SummaryWriter
from transformer_component import transformer_decoder
import pickle
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger
from tqdm.contrib.logging import logging_redirect_tqdm

opts = load_opts()
device = opts.device
device = torch.device(device)

with open('pca.pkl', 'rb') as pickle_file:
    pca = pickle.load(pickle_file)

def get_logger(filename, verbosity=1, name=__name__):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

class network(nn.Module):
    def __init__(self, vis_enc, aud_enc, transformer):
        super().__init__()
        self.vis_enc = vis_enc
        self.aud_enc = aud_enc
        self.transformer = transformer

    def forward(self, video, audio, phase=0, train=True):
        if train:
            if phase == 0:
                vid_emb = self.vis_enc(video)
                batch_size,c,t, h, w= vid_emb.shape
                vid_emb = vid_emb[:, None]
                vid_emb = vid_emb.expand(-1, batch_size, -1, -1, -1, -1)
                vid_emb = vid_emb.reshape(-1, c, t, h, w)
                aud_emb = self.aud_enc(audio)
                batch_size, c_aud, t_aud = aud_emb.shape
                aud_emb = aud_emb[None, :]
                aud_emb = aud_emb.expand(batch_size, -1, -1, -1).reshape(-1, c_aud, t_aud)
                cls_emb = self.transformer(vid_emb, aud_emb)
            elif phase == 1:
                vid_emb = self.vis_enc(video)
                batch_size,c,t, h, w= vid_emb.shape
                vid_emb = vid_emb[:, None]
                vid_emb = vid_emb.expand(-1, opts.number_sample, -1, -1, -1, -1)
                vid_emb = vid_emb.reshape(-1, c, t, h, w)
                aud_emb = self.aud_enc(audio)
                batch_size, c_aud, t_aud = aud_emb.shape
                aud_emb_new = torch.zeros_like(aud_emb)
                aud_emb_new = aud_emb_new[None, :]
                aud_emb_new = aud_emb_new.expand(opts.number_sample, -1, -1, -1).reshape(-1, c_aud, t_aud)
                num_sample = opts.number_sample
                if batch_size == num_sample*(opts.bs2):
                    for k in range(opts.bs2):
                        aud_emb_new[k*num_sample*num_sample:(k+1)*num_sample*num_sample] = (aud_emb[k*num_sample:(k+1)*num_sample][None, :]).expand(opts.number_sample, -1, -1, -1).reshape(-1, c_aud, t_aud)
                else:
                    bs2 = int(batch_size / num_sample)
                    assert batch_size == bs2 * num_sample
                    for k in range(bs2):
                        aud_emb_new[k*num_sample*num_sample:(k+1)*num_sample*num_sample] = (aud_emb[k*num_sample:(k+1)*num_sample][None, :]).expand(opts.number_sample, -1, -1, -1).reshape(-1, c_aud, t_aud)
                aud_emb = aud_emb_new
                cls_emb = self.transformer(vid_emb, aud_emb)
            
        else:
            vid_emb = self.vis_enc(video)
            aud_emb = self.aud_enc(audio)
            cls_emb = self.transformer(vid_emb, aud_emb)
        return cls_emb


def visualize_results(output_dir, video_name, predict_set, predict_set_avfeature, 
                     prob, prob_avfeature, final_score, max_len=50):
    """Create comprehensive visualization of detection results"""
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, 'visualizations')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Clean video name for filename
    clean_name = os.path.splitext(os.path.basename(video_name))[0]
    
    # 1. Delay Distribution Heatmap
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Audio-Visual Forensics Analysis: {clean_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Raw delay distribution
    ax = axes[0, 0]
    if predict_set.shape[0] > 0:
        im = ax.imshow(predict_set.T, aspect='auto', cmap='RdYlGn_r', interpolation='bilinear')
        ax.set_xlabel('Time Frame Index')
        ax.set_ylabel('Delay Offset (-15 to +15 frames)')
        ax.set_title('Delay Distribution Over Time\n(Darker = More Likely Delay)')
        y_ticks = np.arange(0, 31, 5)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{i-15}' for i in y_ticks])
        plt.colorbar(im, ax=ax, label='Anomaly Score')
    
    # Plot 2: Average delay distribution
    ax = axes[0, 1]
    avg_delay = np.mean(predict_set, axis=0)
    offset_labels = np.arange(-15, 16, 1)
    ax.bar(offset_labels, avg_delay, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Perfect Sync (0)')
    ax.set_xlabel('Delay Offset (frames)')
    ax.set_ylabel('Average Score')
    ax.set_title('Average Delay Distribution\n(Real videos peak at 0, Fakes are noisy)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: AV Feature Distribution Heatmap
    ax = axes[1, 0]
    if predict_set_avfeature.shape[0] > 0:
        im = ax.imshow(predict_set_avfeature.T, aspect='auto', cmap='viridis', interpolation='bilinear')
        ax.set_xlabel('Time Frame Index')
        ax.set_ylabel('Feature Index')
        ax.set_title('Audio-Visual Feature Correlation Over Time\n(PCA-reduced)')
        plt.colorbar(im, ax=ax, label='Feature Score')
    
    # Plot 4: Final Scores Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create summary text
    summary_text = f"""
    DETECTION RESULTS SUMMARY
    {'='*50}
    
    Video Name: {clean_name}
    
    SCORES:
    ├─ Distribution-based Score: {prob:.4f}
    ├─ AV-Feature Score: {prob_avfeature:.4f}
    ├─ Lambda (blending factor): {opts.lam}
    └─ Final Combined Score: {final_score:.4f}
    
    INTERPRETATION:
    ├─ Score Range: [0, ∞) - Higher = More Likely Fake
    ├─ Threshold (typical): ~0.5-1.0
    ├─ Verdict: {"[ALERT] LIKELY DEEPFAKE" if final_score > 0.5 else "[OK] LIKELY REAL"}
    
    DELAY ANALYSIS:
    ├─ Peak Delay: {offset_labels[np.argmax(avg_delay)]} frames
    ├─ Max Delay Score: {np.max(avg_delay):.4f}
    ├─ Delay Variance: {np.var(avg_delay):.4f}
    └─ Entropy: {-np.sum(np.exp(-predict_set) * predict_set + 1e-10):.4f}
    
    TEMPORAL CONSISTENCY:
    ├─ Sequence Length: {predict_set.shape[0]} frames
    ├─ Max Length: {max_len} frames
    └─ Coverage: {(predict_set.shape[0]/max_len)*100:.1f}%
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save figure
    output_path = os.path.join(vis_dir, f'{clean_name}_analysis.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Visualization saved: {output_path}")
    
    # Save detailed data as CSV
    csv_path = os.path.join(vis_dir, f'{clean_name}_delays.csv')
    df = pd.DataFrame(predict_set, columns=[f'delay_{i-15}' for i in range(31)])
    df['timestamp'] = df.index
    df.to_csv(csv_path, index=False)
    print(f"[OK] Delay data saved: {csv_path}")


def test2(dist_model, avfeature_model, loader, dist_reg_model, avfeature_reg_model, max_len=50):
    output_dir = opts.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger_path = os.path.join(output_dir, 'output.log')
    score_file_path = os.path.join(output_dir, 'testing_scores.npy')
    logger = get_logger(logger_path)
    logger.info('Start testing!')
    score_list = []
    video_names = []
    
    with logging_redirect_tqdm():
        with tqdm(total=len(loader), position=0, leave=False, colour='green', ncols=150) as pbar:
            for nm, aud_vis in enumerate(loader):
                video_set = aud_vis['video']
                audio_set = aud_vis['audio']
                path_for_detect = aud_vis['sample'][0]  # Get video path
                video_names.append(path_for_detect)
                
                pbar.set_postfix(data_path=path_for_detect)
                time_len = video_set.shape[2]
                
                if (time_len - 5 + 1) < max_len:
                    max_seq_len = time_len - 5 + 1
                else:
                    max_seq_len = max_len
                    
                predict_set = np.zeros((max_seq_len, 31))
                predict_set_avfeature = np.zeros((max_seq_len, 31))
                
                for k in tqdm(range(max_seq_len), position=1, leave=False, colour='red', ncols=80):
                    video = video_set[:, :, k:k+5, :, :]
                    audio = audio_set[:, (k+15-15)*opts.aud_fact:(k+5+15+15)*opts.aud_fact]
                    
                    dist_model.eval()
                    avfeature_model.eval()
                    with torch.no_grad():
                        batch_size = video.shape[0]
                        b, c, t, h, w = video.shape
                        video = video[:, None]
                        video = video.repeat(1, 31, 1, 1, 1, 1).reshape(-1, c, t, h, w).to(device)
                        audio_list = []
                        for j in range(batch_size):
                            for i in range(31):
                                audio_list.append(audio[j:j+1, i*opts.aud_fact:(i+5)*opts.aud_fact])
                        audio = torch.cat(audio_list, dim=0).to(device)
                        audio, _, _, _ = wav2filterbanks(audio.to(device), device=device)
                        audio = audio.permute([0, 2, 1])[:, None]
                        score = dist_model(video, audio, train=False)
                        avfeature = avfeature_model(video, audio, train=False)[15]
                        score = score.reshape(batch_size, 31)
                        avfeature = avfeature.cpu().numpy()[None, ...]
                        avfeature = pca.transform(avfeature)
                        predict_set[k] = score.squeeze(0).cpu().numpy()
                        predict_set_avfeature[k] = avfeature
                
                # Regression analysis
                dist_reg_model.eval()
                avfeature_reg_model.eval()
                mask = torch.zeros(max_len)
                predict_set_tensor = torch.from_numpy(predict_set)
                predict_set_avfeature_tensor = torch.from_numpy(predict_set_avfeature)
                criterion = nn.KLDivLoss(reduce=False)
                criterion_av_feature = nn.MSELoss(reduction="none")
                
                if predict_set_tensor.shape[0] < max_len:
                    pad_len = max_len - predict_set_tensor.shape[0]
                    mask[predict_set_tensor.shape[0]:] = 1.0
                    seq = F.pad(predict_set_tensor, (0, 0, 0, pad_len))
                    seq_avfeature = F.pad(predict_set_avfeature_tensor, (0, 0, 0, pad_len))
                else:
                    start = 0
                    seq = predict_set_tensor[start:start+max_len, :]
                    seq_avfeature = predict_set_avfeature_tensor[start:start+max_len, :]
                    pad_len = 0
                    
                seq = seq[None, :, :]
                seq_avfeature = seq_avfeature[None, :, :]
                seq = seq.to(device)
                seq_avfeature = seq_avfeature.to(device)
                mask = mask.to(device)
                mask = mask[None, :]
                
                with torch.no_grad():
                    target = seq[:, 1:, :]
                    target = nn.functional.softmax(target.float(), dim=2)
                    target_av_feature = seq_avfeature[:, 1:, :]
                    target_av_feature = F.normalize(target_av_feature, p=2.0, dim=2)
                    input = seq[:, :-1, :]
                    input_av_feature = seq_avfeature[:, :-1, :]
                    input_av_feature = F.normalize(input_av_feature, p=2.0, dim=2)
                    input_mask_ = mask[:, :-1]
                    
                    logit = dist_reg_model(input.float(), input_mask_)
                    logit_avfeature = avfeature_reg_model(input_av_feature.float(), input_mask_)
                    logit = nn.functional.log_softmax(logit, dim=2)
                    
                    prob_total = criterion(logit, target)
                    prob_total_avfeature = criterion_av_feature(logit_avfeature, target_av_feature)
                    
                    prob = prob_total[0, :(max_len - pad_len - 1)]
                    prob_avfeature = prob_total_avfeature[0, :(max_len - pad_len - 1)]
                    prob = torch.sum(prob, dim=1)
                    prob = torch.mean(prob)
                    prob_avfeature = torch.sum(prob_avfeature, dim=1)
                    prob_avfeature = torch.mean(prob_avfeature)
                    
                    final_score = (opts.lam) * prob_avfeature + prob
                
                logger.info(f"Video: {path_for_detect} | Dist Score: {prob.item():.4f} | AV Score: {prob_avfeature.item():.4f} | Final: {final_score.item():.4f}")
                score_list.append(final_score.item())
                
                # Generate visualization
                visualize_results(output_dir, path_for_detect, predict_set, predict_set_avfeature,
                                prob.item(), prob_avfeature.item(), final_score.item(), max_len)
                
                pbar.update(1)
        
        np.save(score_file_path, np.array(score_list))
        logger.info('Finished!')
        
        # Create summary report
        summary_path = os.path.join(output_dir, 'detection_summary.csv')
        summary_df = pd.DataFrame({
            'video_path': video_names,
            'final_score': score_list,
            'verdict': ['FAKE' if s > 0.5 else 'REAL' for s in score_list]
        })
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[OK] Summary report saved: {summary_path}")


def main():
    # -----------------------------------------
    # LOAD OPTIONS FROM COMMAND LINE
    # -----------------------------------------
    opts_loaded = load_opts()
    
    # Override global device
    global device
    device = torch.device(opts_loaded.device)

    # -----------------------------------------
    # LOAD MODELS
    # -----------------------------------------
    vis_enc, _ = select_backbone(network='r18')
    aud_enc = AudioEncoder()
    
    Transformer = MP_AViT(
        image_size=14, patch_size=0, num_classes=1, dim=512, depth=3, heads=4, 
        mlp_dim=512, dim_head=128, dropout=0.1, emb_dropout=0.1, 
        max_visual_len=5, max_audio_len=4
    )

    avfeature_Transformer = MP_av_feature_AViT(
        image_size=14, patch_size=0, num_classes=1, dim=512, depth=3, heads=4, 
        mlp_dim=512, dim_head=128, dropout=0.1, emb_dropout=0.1, 
        max_visual_len=5, max_audio_len=4
    )

    sync_model = network(vis_enc=vis_enc, aud_enc=aud_enc, transformer=Transformer)
    avfeature_sync_model = network(vis_enc=vis_enc, aud_enc=aud_enc, transformer=avfeature_Transformer)

    sync_model_weight = torch.load('sync_model.pth', map_location='cpu')
    sync_model.load_state_dict(sync_model_weight)
    avfeature_sync_model.load_state_dict(sync_model_weight)

    sync_model.to(device)
    avfeature_sync_model.to(device)

    dist_regressive_model = transformer_decoder(
        input_dim_old=31, input_dim=256, compress_factor=1, num_heads=16, 
        dropout_prob=0.1, max_len=49, layers=2
    )

    avfeature_regressive_model = transformer_decoder(
        input_dim_old=31, input_dim=256, compress_factor=1, num_heads=16,
        dropout_prob=0.1, max_len=49, layers=2
    )

    dist_regressive_model.load_state_dict(
        torch.load('dist_regressive_model.pth', map_location="cpu")
    )
    avfeature_regressive_model.load_state_dict(
        torch.load('avfeature_regressive_model.pth', map_location="cpu")
    )

    dist_regressive_model.to(device)
    avfeature_regressive_model.to(device)

    # -----------------------------------------
    # VIDEO LOADING
    # -----------------------------------------
    if opts_loaded.test_video_path.endswith(".mp4"):
        test_video = FakeAVceleb(
            [opts_loaded.test_video_path], opts_loaded.resize, opts_loaded.fps, opts_loaded.sample_rate,
            vid_len=opts_loaded.vid_len, phase=0, train=False, number_sample=1, 
            lrs2=False, need_shift=False, lrs3=False, kodf=False, 
            lavdf=False, robustness=False, test=True
        )
    else:
        with open(opts_loaded.test_video_path) as f:
            paths = f.readlines()
        test_video = FakeAVceleb(
            paths, opts_loaded.resize, opts_loaded.fps, opts_loaded.sample_rate,
            vid_len=opts_loaded.vid_len, phase=0, train=False, number_sample=1,
            lrs2=False, need_shift=False, lrs3=False, kodf=False,
            lavdf=False, robustness=False, test=True
        )

    loader_test = DataLoader(
        test_video, batch_size=opts_loaded.bs, num_workers=opts_loaded.n_workers, shuffle=False
    )

    # Run detection
    test2(sync_model, avfeature_sync_model, loader_test,
          dist_reg_model=dist_regressive_model,
          avfeature_reg_model=avfeature_regressive_model,
          max_len=opts_loaded.max_len)


if __name__ == '__main__':
    main()


# ============================================================
# USAGE EXAMPLES
# ============================================================
#
# 1. Single video with all defaults:
#    python detect_with_visualization.py --test_video_path "C:\path\to\video.mp4"
#
# 2. Single video with custom device and output:
#    python detect_with_visualization.py --test_video_path "C:\path\to\video.mp4" --device cuda:0 --output_dir "save"
#
# 3. Single video with all parameters:
#    python detect_with_visualization.py --test_video_path "C:\path\to\video.mp4" --device cuda:0 --output_dir "save" --max-len 50 --n_workers 4 --bs 1 --lam 0
#
# 4. Multiple videos (text file list):
#    python detect_with_visualization.py --test_video_path "videos_list.txt" --device cuda:0 --output_dir "save"
#
# 5. With CPU only:
#    python detect_with_visualization.py --test_video_path "C:\path\to\video.mp4" --device cpu --output_dir "save"
#
# ============================================================
# OUTPUTS
# ============================================================
# All results saved in <output_dir>/visualizations/ folder:
#   - {video_name}_analysis.png    (4-panel visualization with delays + scores)
#   - {video_name}_delays.csv      (raw delay distribution data)
#   - detection_summary.csv        (summary of all processed videos)
#   - output.log                   (detection logs with scores)

