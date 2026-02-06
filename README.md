```
# Audio-Visual Forensics — Improved (PyTorch)

This repository is an improved scaffold meant for thesis work: it adds a deblurring U-Net and a deepfake detector built on a ResNet backbone.

## Quick start
1. Prepare datasets:
   - For deblurring: `paired_dir/blurred/*` and `paired_dir/sharp/*` (matching file counts)
   - For detection: `data_dir/real/*` and `data_dir/fake/*`
2. Train deblurring:
   bash
   python deblurring/train.py --paired_dir ./paired --out ./checkpoints_deblur --epochs 20
`

3. Deblur a video:

   bash
   python deblurring/infer.py --src input.mp4 --dst deblurred.mp4 --ckpt ./checkpoints_deblur/unet_epoch20.pth
   
4. Train detector:

   bash
   python deepfake_detection/train.py --data_dir ./detector_dataset --epochs 10 --save_prefix detector
   
5. Run detector on video:

   bash
   python deepfake_detection/infer.py --ckpt detector_epoch10.pth --src deblurred.mp4 --out results.csv
   

## Suggestions to improve for publication

* Replace U-Net with DeblurGANv2 or a transformer-based deblurring network.
* Use perceptual (VGG) + L1 losses for deblurring; add adversarial loss for higher realism.
* For detection, use face-aligned crops, augmentations, and stronger backbones (EfficientNet / ViT).
* Provide thorough ablation study: (i) deblur before detect vs detect on original; (ii) effect of deblurring strength; (iii) performance metrics (AUC, EER).

## License

Add your institution's license / attribution.



---

## Next steps I can do for you (pick any):
- Hook the project to your uploaded repository and adapt file paths / naming to match your existing code.
- Replace the U-Net with DeblurGAN-lite and include adversarial training code (more complex).
- Add sample Jupyter notebook with visual comparisons and metrics for thesis figures.


If you want any of the "Next steps" implemented, tell me exactly which one and I'll add it.

```