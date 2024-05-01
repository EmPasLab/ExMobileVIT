import torch.nn.functional as F
import torch.nn as nn

class LabelSmoothingCrossEntropyLoss(nn.Module):

    # Label Smoothing Croos Entropy Loss

    # Like paper, use smoothing as 0.1
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, input, target):
        log_probs = F.log_softmax(input, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()