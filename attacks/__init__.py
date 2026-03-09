"""
Attack modules initialization.
"""

from attacks.base import BaseAttack, SpatialOnlyAttack, FrequencyOnlyAttack, HybridAttack
from attacks.hybrid import SequentialHybridAttack, JointHybridAttack, AdaptiveBandAttack
from attacks.sota import FGSM, PGD, SFA, FACL

__all__ = [
    'BaseAttack',
    'SpatialOnlyAttack',
    'FrequencyOnlyAttack',
    'HybridAttack',
    'SequentialHybridAttack',
    'JointHybridAttack',
    'AdaptiveBandAttack',
    'FGSM',
    'PGD',
    'SFA',
    'FACL',
]
