# Jet Tagging with DeepSets + Prototype (xAI)

## Pipeline Design

```
H5 Dataset
    ↓
Awkward Array (jagged particles)
    ↓
Vector (kinematics: eta, phi, pT, mass)
    ↓
FastJet (τ₃₂, D₂, mass, pT)
    ↓
DeepSets (particle → embedding → Σ pooling)
    ↓
Concat: [jet_embedding, fastjet_features]
    ↓
Prototype Layer (EDM similarity)
    ↓
Classification (probability scores)
```

## Architecture Choices

### 1. DeepSets
- Particle encoder: MLP(node_dim → hidden_dim)
- Pooling: Sum or Mean over particles
- Jet encoder: MLP(particle_emb → jet_emb)

### 2. FastJet Features
- Jet mass, pT, eta, phi
- N-subjettiness ratio: τ₂₁ = τ₂/τ₁
- N-subjettiness ratio: τ₃₂ = τ₃/τ₂
- Energy correlation: D₂
-ECF ratios: C₂, C₃

### 3. Prototype Layer (xAI)
- Similarity metric: **Energy Distance (EDM)**
- EDM(z, p) = √(E[z] + E[p] - 2·E[√(pz)])
- 20 prototypes (10 Top, 10 QCD)
- Output: similarity to each prototype

### 4. Classification
- Combine prototype similarities
- Output: P(Top), P(QCD)

## Why This Approach?

| Criteria | Reason |
|----------|--------|
| **Fast** | DeepSets ~O(N), no edge iteration |
| **Lightweight** | No graph edges, just pooling |
| **Accurate** | DeepSets proven on jet tagging |
| **Explainable** | Prototype layer shows "why" |
| **Physics** | FastJet observables + vector math |

## References

- DeepSets: Zaheer et al. (2017) - permutation invariance
- Energy Distance: Székely (2002) - statistical distance
- Jet Tagging: Qu & Gouskos (ParticleNet), Moreno et al.
