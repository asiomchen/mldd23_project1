# Concious Cross-Entropy

import torch
import torch.nn as nn

class CCE(nn.Module):
    def __init__(self, ignore_index=40):
        super(CCE, self).__init__()
        self.idx_ignore = ignore_index # index of token to ignore
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, target, predictions):
        # target.shape [batch_size, seq_len, token_idx]
        batch_size = predictions.shape[0]
        one_hot = target.argmax(dim=-1)
        mask = (one_hot != self.idx_ignore)
        weights = (mask.T / mask.sum(axis=1)).T[mask]
        loss = torch.nn.functional.cross_entropy(predictions[mask], one_hot[mask], reduction='none')
        return (weights * loss).sum() / batch_size