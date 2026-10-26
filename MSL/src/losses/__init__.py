"""Loss functions for MSL"""

from .msl_ranking_loss import MSLRankingLoss, BinaryCrossEntropyLoss

__all__ = [
    'MSLRankingLoss',
    'BinaryCrossEntropyLoss'
]

