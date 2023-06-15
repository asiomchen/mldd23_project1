import torch
import torch.nn as nn

class ConsciousCrossEntropy(nn.Module):
    def __init__(self, device):
        super(ConsciousCrossEntropy, self).__init__();
        self.batch_size = 256
        self.alphabet_len = 42
        self.seq_len = 128
        self.idx_ignore = 40 # index of token to ignore
        self.ignore = self.prep_token_to_ignore()
        self.device = device

    def forward(self, target, predictions):
        cross_entropy_loss = 0
        for y_true, y in zip(target, predictions):
            sequence_loss = 0
            mask = self.prep_mask(y_true)
            nops = torch.sum(mask)
            product = torch.mul(y_true, y)
            prob = torch.sum(product, dim=1)
            loss = -torch.log(prob)
            loss_masked = torch.mul(loss, mask)
            sequence_loss = torch.sum(loss_masked)/(self.seq_len - nops)
            cross_entropy_loss += sequence_loss
        loss_value = cross_entropy_loss/self.batch_size
        return loss_value

    def prep_token_to_ignore(self):
        ignore = torch.zeros(self.alphabet_len).to(self.device)
        ignore[self.idx_ignore] = 1
        ignore = ignore.unsqueeze(0).repeat(128,1)
        return ignore

    def prep_mask(self, y_true):
        # look through target SELFIES sequence and prepare mask
        # as a tensor of size [128] with 0s on [nop] symbol
        # and 1s for all the other tokens
        v1 = torch.zeros(self.alphabet_len).to(self.device)
        v1[self.idx_ignore] = 1
        m1 = v1.unsqueeze(0).repeat(y_true.shape[0], 1)
        product = torch.mul(y_true, m1)
        output = ~torch.sum(product, dim=1).bool()
        output = output.float()
        return output