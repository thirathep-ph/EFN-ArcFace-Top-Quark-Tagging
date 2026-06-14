# Baseline Implementation Plan (Minimal)

**เป้าหมาย**: เพิ่ม baseline model แค่ที่จำเป็น เพื่อเปรียบเทียบกับ EFN + ArcFace โดยใช้ทรัพยากรที่มีอยู่

**ข้อตกลง**: 
- ไม่รายงาน spec computer หรือ training time ใน paper
- ใช้เฉพาะ metrics ที่วัดได้จากผลลัพธ์ (Accuracy, AUC, F1)
- Train เฉพาะที่จำเป็น ตัดของที่หนักเกินไป

---

## 1. Plain EFN + Softmax (Ablation Control)

### ความสำคัญ: 🔴 **ต้องมี**

นี่คือ ablation ที่สำคัญที่สุด เพราะวัดว่า **ArcFace ทำให้ performance drop หรือไม่** เมื่อเทียบกับ classifier head ทั่วไป

### การเปลี่ยนแปลงโค้ด

**ไฟล์ที่มีอยู่**: [`jet_tagging/models/efn_linear.py`](jet_tagging/models/efn_linear.py)

```python
"""
EFN + Standard Softmax (Ablation Baseline)
เหมือน EFN + ArcFace ทุกประการ แต่ใช้ Linear + CrossEntropy แทน ArcFace
"""
import torch.nn as nn
from jet_tagging.models.efn_arcface import EFN

class TopTaggingSoftmax(nn.Module):
    def __init__(self, embedding_dim=64, particle_dim=128):
        super().__init__()
        self.efn = EFN(input_dim=2, particle_dim=particle_dim, jet_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 2)
    
    def forward(self, constituents, mask=None):
        coords, z = get_efn_features(constituents, mask)
        jet_emb = self.efn(coords, z, mask)
        logits = self.classifier(jet_emb)
        return logits, jet_emb
```

### Training

ใช้โค้ดจาก [`scripts/train_top.py`](scripts/train_top.py) ได้เลย โดยแก้แค่:
- เปลี่ยนจาก `TopTaggingModel` → `TopTaggingSoftmax`
- เอา ArcFace margin warmup ออก
- ใช้ CrossEntropyLoss ปกติ

**จำนวน epochs**: 30 (15 epochs warmup-like + 15 convergence)
**เวลาประมาณ**: เท่ากับ main model (~2.5 ชม. สำหรับ 100 epochs → 45 นาทีสำหรับ 30 epochs)

---

## 2. PFN / Deep Sets (Simple Permutation-Invariant Baseline)

### ความสำคัญ: 🟡 **มีดีกว่าไม่มี**

PFN (Particle Flow Network) เป็นอีก baseline ที่สำคัญ เพราะ:
- ต่างจาก EFN ตรงที่ **ไม่รับประกัน IRC-safety**
- ใช้ **mean pooling** แทน energy-weighted sum pooling
- ทำให้เห็นว่า IRC-safety constraint ส่งผลต่อ performance แค่ไหน

### การเปลี่ยนแปลงโค้ด

**ไฟล์ที่มีอยู่**: [`jet_tagging/models/efn_coslinear.py`](jet_tagging/models/efn_coslinear.py) (ArcFace without margin)

```python
"""
PFN (Particle Flow Network) — Simple Permutation-Invariant Baseline
ไม่รับประกัน IRC-safety (ใช้ mean pooling และเห็นทั้ง E, eta, phi)
"""
class PFN(nn.Module):
    def __init__(self, input_dim=4, particle_dim=128, jet_dim=64):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, particle_dim),  # เห็น E, px, py, pz (ไม่ใช่แค่ angular)
            nn.BatchNorm1d(particle_dim),
            nn.ReLU(),
            nn.Linear(particle_dim, particle_dim),
            nn.BatchNorm1d(particle_dim),
            nn.ReLU(),
            nn.Linear(particle_dim, jet_dim),
        )
        self.classifier = nn.Linear(jet_dim, 2)
    
    def forward(self, constituents, mask=None):
        # ใช้ mean pooling (ไม่รับประกัน IRC-safety)
        phi_out = self.phi(constituents)  # (B, N, jet_dim)
        if mask is not None:
            phi_out = phi_out * mask.unsqueeze(-1).float()
            jet_emb = phi_out.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-10)
        else:
            jet_emb = phi_out.mean(dim=1)
        logits = self.classifier(jet_emb)
        return logits, jet_emb
```

### Training
ใช้ pipeline เดียวกับ Softmax variant
**จำนวน epochs**: 30
**เวลาประมาณ**: ~45 นาที - 1 ชม.

---

## 3. ParticleNet — **ข้ามไปก่อน** ⏭️

ParticleNet (EdgeConv + Dynamic Graph CNN) เป็น SOTA จริง แต่:
- 🔴 **เทรนนานมาก** (~6-8 ชม. บน GPU ธรรมดา)
- 🔴 **ต้องการ GPU memory สูง** (เทียบเท่า 2-3x ของ EFN)
- 🔴 **Implementation ซับซ้อน** (ต้องเขียน ParticleNet block เอง หรือหา package)

**ข้อเสนอ**: เก็บไว้ทำทีหลัง หรือใช้ตัวเลขจาก literature อ้างอิงแทน (quoted)

---

## 4. สรุป Training Protocol

### Hyperparameters และจำนวน parameters

| Model | Params | Epochs | เวลาโดยประมาณ |
|:------|:------:|:------:|:--------------|
| EFN + ArcFace (main) | 46,592 | 30 | ~45 นาที |
| EFN + Softmax (ablation) | 46,338 | 30 | ~45 นาที |
| PFN / Deep Sets (baseline) | ~33,000 | 30 | ~45 นาที |

### Evaluation Protocol (ใช้ [`scripts/evaluate_models.py`](scripts/evaluate_models.py))

```bash
# สำหรับทุก model
python scripts/evaluate_models.py \
    --model_path checkpoints/baseline/<model_name>/best.pt \
    --split test \
    --max_events 5000
```

### Metrics ที่เปรียบเทียบ

| Metric | วัดอะไร |
|:-------|:--------|
| **Test Accuracy** | Overall classification correctness |
| **ROC AUC** | Class separation quality |
| **Precision** | Top jet identification purity |
| **Recall** | Top jet identification efficiency |
| **F1-Score** | Harmonic mean of precision & recall |
| **Confusion Matrix** | TP/TN/FP/FN counts |

---

## 5. ข้อควรระวัง

1. **ใช้ dataset split เดียวกัน** — train.h5 / val.h5 / test.h5 (อย่าใช้คนละชุด)
2. **ใช้ preprocessing pipeline เดียวกัน** — centering, normalization
3. **ไม่ต้อง optimize hyperparameters** ของ baselines มาก — แค่ default ก็พอ เพราะ paper ต้องการแสดง trend ไม่ใช่ beating SOTA
4. **ไม่รายงาน training time / hardware spec** — เห็นด้วยกับคุณว่าไม่จำเป็น
5. **Focus ที่ interpretability comparison** มากกว่า absolute performance — ข้อความหลักของ paper คือ "EFN+ArcFace gives interpretability with minimal performance cost"

---

## 6. สิ่งที่ต้องเขียนใน Code

### จะได้ไฟล์เพิ่ม

| File | Description |
|:-----|:------------|
| `jet_tagging/models/efn_linear.py` | EFN backbone + Linear classifier (Model A) |
| `jet_tagging/models/efn_coslinear.py` | EFN backbone + Cosine Linear (Model B, ArcFace without margin) |
| `scripts/train_baselines.py` | Training script สำหรับทั้ง 2 baselines (ใช้ shared logic) |

### จะได้ checkpoint ใหม่

```
checkpoints/baseline/
├── efn_softmax/best.pt
└── pfn/best.pt
```

### จะได้ results ใหม่

```
results/baseline/
├── efn_softmax/
│   ├── metrics.json
│   └── performance.png
└── pfn/
    ├── metrics.json
    └── performance.png
```
