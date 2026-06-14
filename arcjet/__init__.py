"""
ArcJet: Ante-hoc Interpretable Jet Tagging with EFN and ArcFace
================================================================
EFN (Energy Flow Network) backbone + ArcFace hyperspherical prototypes
for intrinsically interpretable top quark tagging.

Core components:
    - TopTaggingModel: Main EFN + ArcFace model
    - EFN: Energy Flow Network (IRC-safe)
    - ArcFaceHead: Hyperspherical prototype classifier
    - TopTaggingLinear: EFN + Linear baseline
    - TopTaggingCosLinear: EFN + Cosine Linear baseline
"""

from .models.efn_arcface import TopTaggingModel, EFN, ArcFaceHead, arcface_loss, count_parameters
from .models.efn_linear import TopTaggingLinear
from .models.efn_coslinear import TopTaggingCosLinear

__all__ = [
    'TopTaggingModel', 'EFN', 'ArcFaceHead',
    'TopTaggingLinear', 'TopTaggingCosLinear',
    'arcface_loss', 'count_parameters',
]
