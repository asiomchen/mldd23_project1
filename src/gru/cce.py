# Concious Cross-Entropy

import torch
import torch.nn as nn

class CCE(nn.Module):
    def __init__(self, ignore_index=40):
        """
        Concious Cross-Entropy

        Parameters:
            ignore_index: index of the padding token to ignore
                          when calculating loss
        """
        super(CCE, self).__init__()
        self.idx_ignore = ignore_index # index of token to ignore
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, target, predictions):
        """
        Forward pass
        Args:
            target (torch.Tensor): target tensor of shape [batch_size, seq_len, token_idx]
            predictions (torch.Tensor): predictions tensor of shape [batch_size, seq_len, token_idx]

        Returns:
            loss (torch.Tensor): loss value
        """
        # target.shape [batch_size, seq_len, token_idx]
        batch_size = predictions.shape[0]
        one_hot = target.argmax(dim=-1)
        mask = (one_hot != self.idx_ignore)
        weights = (mask.T / mask.sum(axis=1)).T[mask]
        loss = torch.nn.functional.cross_entropy(predictions[mask], one_hot[mask], reduction='none')
        loss = (weights * loss).sum() / batch_size
        return loss

    import numpy as np

    class CCE_old(nn.Module):
        def __init__(self):
            '''
            Concious Cross-Entropy (deprecated). Use CCC() instead.
            '''
            super(CCE, self).__init__();
            self.alphabet_len = 42
            self.seq_len = 128
            self.idx_ignore = 40  # index of token to ignore
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.ignore = torch.zeros(self.alphabet_len).to(self.device)
            self.ignore[self.idx_ignore] = 1
            self.ignore = self.ignore.unsqueeze(0).repeat(128, 1)

        def forward(self, target, predictions):
            # target.shape [batch_size, seq_len, token_idx]
            batch_size = predictions.shape[0]
            cross_entropy_loss = 0
            for y_true, y in zip(target, predictions):
                mask = self.prep_mask(y_true)
                nops = torch.sum(mask)
                prob = torch.sum(y_true * y, dim=1)
                loss = -torch.log(prob)
                sequence_loss = torch.sum(loss * mask) / (self.seq_len - nops)
                cross_entropy_loss += sequence_loss
            loss_value = cross_entropy_loss / batch_size
            return loss_value

        def prep_mask(self, y_true):
            '''
            look through target SELFIES sequence and prepare mask
            as a tensor of size [128] with 0s on [nop] symbol
            and 1s for all the other tokens
            '''
            v1 = torch.zeros(self.alphabet_len).to(self.device)
            v1[self.idx_ignore] = 1
            m1 = v1.unsqueeze(0).repeat(y_true.shape[0], 1)
            output = ~torch.sum(y_true * m1, dim=1).bool()
            return output.float()