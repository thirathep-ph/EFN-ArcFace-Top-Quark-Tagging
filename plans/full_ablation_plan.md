# Complete Ablation Plan — All Models & Parameters

> ทุก model, ทุก parameter, ทุก ablation ที่ต้องรัน

---

## 1. Full Model (Current — d=64, p=128)

### Architecture

```
EFN Backbone (58,816 params) — เหมือนกันทุก model:
  Phi (per-particle MLP):
    Linear(2, 128)     -> 256 + 128 bias = 384
    BatchNorm1d(128)   -> 128 + 128 = 256
    ReLU
    Linear(128, 128)   -> 16,384 + 128 = 16,512
    BatchNorm1d(128)   -> 256
    ReLU
    Linear(128, 128)   -> 16,384 + 128 = 16,512
    -> Sum pooling weighted by z_i

  Rho (jet-level MLP):
    Linear(128, 128)   -> 16,384 + 128 = 16,512
    BatchNorm1d(128)   -> 256
    ReLU
    Linear(128, 64)    -> 8,192 + 64 = 8,256
    -> Output: 64-dim embedding

Classification Heads (แตกตางกัน):
  ArcFace:   class_centers(2, 64) = 128         -> Total: 59,072
  Linear:    weight(2, 64) + bias(2) = 130      -> Total: 59,074
  CosLinear: class_centers(2, 64) = 128         -> Total: 59,072
```

### Full Model Runs (Phase 1 — Baselines)

| Run | Model | Seeds | Epochs | Params | Time |
|:---:|:------|:-----:|:------:|:------:|:----:|
| 1 | Linear | 42 | 50 | 59,074 | ~1.7 hr |
| 2 | Linear | 123 | 50 | 59,074 | ~1.7 hr |
| 3 | Linear | 7 | 50 | 59,074 | ~1.7 hr |
| 4 | CosLinear | 42 | 50 | 59,072 | ~1.7 hr |
| 5 | CosLinear | 123 | 50 | 59,072 | ~1.7 hr |
| 6 | CosLinear | 7 | 50 | 59,072 | ~1.7 hr |
| | **Total** | | | | **~10 hr** |

### Full Model Runs (Phase 2 — Robustness)

| Run | Scale (s) | Margin (m) | Epochs | Params | Time |
|:---:|:---------:|:----------:|:------:|:------:|:----:|
| 7 | 8 | 0.0 | 50 | 59,072 | ~2 hr |
| 8 | 8 | 0.3 | 50 | 59,072 | ~2 hr |
| 9 | 8 | 0.5 | 50 | 59,072 | ~2 hr |
| 10 | 16 | 0.0 | 50 | 59,072 | ~2 hr |
| 11 | 16 | 0.3 | 50 | 59,072 | ~2 hr |
| 12 | 16 | 0.5 | 50 | 59,072 | ~2 hr |
| 13 | 32 | 0.0 | 50 | 59,072 | ~2 hr |
| 14 | 32 | 0.3 | 50 | 59,072 | ~2 hr |
| 15 | 32 | 0.5 | 50 | 59,072 | ~2 hr |
| | **Total** | | | | **~18 hr** |

---

## 2. Ablation: Dimension Sweep (เพิ่มเติม)

เพื่อตอบ reviewer: "ทำไม 64 dims? ลดเหลือ 16 หรือ 32 ไดไหม?"

| Run | embedding_dim | particle_dim | Params (ArcFace) | Expected Acc | Time |
|:---:|:-------------:|:------------:|:----------------:|:------------:|:----:|
| A1 | **8** | **16** | **~3,500** | ~88-89% | ~1.5 hr |
| A2 | 16 | 32 | ~7,200 | ~90% | ~1.5 hr |
| A3 | 32 | 64 | ~29,000 | ~91% | ~1.5 hr |
| A4 | 64 | 128 | 59,072 | ~91.7% | ~2 hr |
| A5 | 128 | 256 | ~120,000 | ~91.8% | ~2.5 hr |

---

## 3. สรุป: สิ่งที่ตองรันทั้งหมด

| Phase | อะไร | จำนวน | เวลา | อยู่ใน run_all.py? |
|:------|:-----|:-----:|:----:|:------------------:|
| **1** | Baselines (Linear + CosLinear x 3 seeds) | 6 runs | ~10 hr | Yes |
| **2** | Robustness (9 trials sxm) | 9 runs | ~18 hr | Yes |
| **3** | SVD analysis | 1 run | ~5 min | Yes |
| **4** | Bootstrap | 1 run | ~10 min | Yes |
| **A** | Dimension sweep (d=8,16,32,64,128) | 5 runs | ~8 hr | No |
| | **รวม** | **21 runs** | **~36 hr** | |
