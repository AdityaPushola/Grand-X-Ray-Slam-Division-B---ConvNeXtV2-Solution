# 🩻 Grand X-Ray SLAM (Division A) — ConvNeXtV2 + Heavy Attention + GeM [SOTA Solution]

This repository contains my **top-ranked Kaggle solution** for the **Grand X-Ray SLAM Division A** competition.  
Final Leaderboard Results: 🏆 **8 / 113 (Private)** • **12 / 113 (Public)**

It implements a **ConvNeXtV2-based multi-label image classification pipeline** designed for robust X-ray abnormality detection.

---

## 🚀 Key Highlights (SOTA Features)

- 🧠 **Backbone:** `ConvNeXtV2-Base (FCMAE Fine-tuned IN22K→IN1K)` via [timm](https://github.com/huggingface/pytorch-image-models)
- 💎 **Pooling:** [GeM (Generalized Mean Pooling)](https://arxiv.org/abs/1711.02512)
- ⚡ **Head Architecture:** Custom **Heavy Multi-Head Attention** block for feature refinement
- 🎯 **Loss Function:** Weighted **Focal Loss** for class imbalance
- 🔄 **Optimizer:** `AdamW` with **cosine LR schedule** and warmup
- 🧘 **Regularization:** Gradient clipping, dropout, and **EMA (Exponential Moving Average)** weights
- 🏁 **Early Stopping:** Based on validation **AUC**
- 🔁 **TTA (Test-Time Augmentation):** Horizontal flip averaging
- 🧩 **Multi-GPU Support:** `torch.nn.DataParallel`
- 💾 **Auto-Checkpointing:** Saves best model by validation AUC

---
# Dataset Structure
/kaggle/input/grand-xray-slam-division-a/
│
├── train1.csv
├── train1/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
└── test1/
    ├── 0001.png
    ├── 0002.png
    └── ...
