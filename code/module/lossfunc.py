import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss Function"""
    def __init__(self, gamma=0, pos_weight=1, logits=False, reduction='sum', name='FocalLoss'):
        super(FocalLoss, self).__init__()
        self.name = name
        self.gamma = gamma
        self.weight = pos_weight
        self.logits = logits
        self.reduce = reduction
    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        p = torch.sigmoid(inputs)
        pt = p * targets + (1 - p) * (1 - targets)
        focal_loss = BCE_loss * ((1 - pt) ** self.gamma)
        weight = self.weight * targets + 1 - targets
        focal_loss = weight * focal_loss
        if self.reduce == 'mean':
            return torch.mean(focal_loss)
        elif self.reduce == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss