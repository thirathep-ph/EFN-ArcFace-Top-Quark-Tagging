# Architecture & Design Decisions

## Overview
Top Tagging with DeepSets + Prototype Layer (xAI).
Classify jets as Top (signal) vs QCD (background).

## Why This Architecture?

### DeepSets
- **Role**: Learns permutation-invariant jet embedding from raw constituents (E, px, py, pz)
- **Why needed**: Captures non-linear patterns that no fixed set of observables can express
- **Not redundant**: Provides the "extra dimensions" beyond the 6 FastJet features
- **Without it**: The prototype layer would only see 6 features → linear model in feature space

### FastJet Features (6 observables)
- **Role**: Physics-interpretable observables for prototype explanations
- **Why 6**: mass, pT, eta, tau21, tau32, D2 — well-established for top tagging
- **Expandable**: Can add SoftDrop mass, splitting scales, Lund plane for multi-class

### Prototype Layer
- **Role**: Learns class centers (NOT decision boundaries)
- **20 prototypes**: 10 for Top, 10 for QCD — captures substructure within classes
- **Interpretation**: "This jet is Top because it's similar to Top prototypes"
- **Novelty detection**: Low similarity to ALL prototypes → out-of-distribution / anomaly

### Why Not Just FastJet Features Alone?
- 6 features can't capture all the information in raw constituents
- Two jets with identical {mass, pT, eta, tau21, tau32, D2} can have different substructure
- DeepSets captures the "unexplained residual"

### Why Not Just DeepSets Alone?
- DeepSets is a black box — no explanation for decisions
- Prototypes provide physics-interpretable explanations
- "This jet is similar to Prototype 7 (mass=170, tau21=0.3, D2=1.8)"

## Data Flow
```
Constituents (E, px, py, pz) 
    │
    ├──→ DeepSets ──→ Jet Embedding (128-dim) ──┐
    │                                            │
    ├──→ FastJet ──→ 6 Features (mass, pT, ...) ─┤
    │                                            │
    │                                     [concatenate]
    │                                            │
    │                                    Combined (134-dim)
    │                                            │
    │                                     Prototype Layer
    │                                            │
    │                                     Classification
```

## Key Design Decisions

### 1. Cartesian (E, px, py, pz) for DeepSets Input
- **Chosen over cylindrical** (Δη, Δφ, pT_rel) for now
- FastJet features already provide the cylindrical/physics view
- Can switch later — it's just a data transformation

### 2. Exclusive C/A Clustering (R=10.0)
- NOT inclusive jets
- Cluster all constituents into exactly 1 jet per event
- Vectorized API: `cs.exclusive_jets(n_jets=1)`

### 3. Vectorized AwkwardClusterSequence API
- NOT per-event Python loop
- All events processed in one C++ call
- ~780x speedup over per-event loop
- Input: float64 Awkward Array with {px, py, pz, E}

### 4. 6 Core Features (Expandable)
- mass, pT, eta — basic kinematics
- tau21, tau32 — N-subjettiness ratios (prong structure)
- D2 — energy correlation ratio
- Future: SoftDrop mass, splitting scales, Lund plane

### 5. Multiple Prototypes Per Class
- Captures substructure within classes
- Top: different decay topologies (hadronic, leptonic, boosted)
- QCD: gluon jets, light quark jets, heavy flavor jets

## xAI Capabilities

### Prototype Interpretation
After training, each prototype can be described by:
- Which class it represents
- Average FastJet feature values of jets that activate it
- How many Top vs QCD jets activate it

### Novelty/Anomaly Detection
```python
max_similarity = similarities.max()
if max_similarity < threshold:
    # Out-of-distribution — not like anything seen in training
```

### Future: LHCO2020 Anomaly Detection
- Train on QCD only → prototypes learn "normal" QCD
- Anomalous jets (new physics) → low similarity to all prototypes
- Natural extension of this architecture

## Model Specs
- Parameters: 88,536
- Input: (200, 4) padded constituents
- Particle embedding: 128-dim
- Jet embedding: 128-dim
- FastJet features: 6-dim (expandable)
- Prototypes: 20 (10 per class)
- Temperature: 1.0

## Training
- 30 epochs, batch_size=256, lr=1e-3
- Adam optimizer, ReduceLROnPlateau scheduler
- Class weights for imbalance
- CPU only (PyTorch 2.11.0+cpu)
