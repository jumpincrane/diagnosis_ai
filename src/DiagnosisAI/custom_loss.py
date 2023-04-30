import torch.nn as nn
import torch


class DiceLoss(nn.Module):
    """This definition generalize to real valued pred and target vector. This should be differentiable.
    :param float smooth,

    In forward docstring:
    :param torch.Tensor inputs: tensor with first dimension as batch,
    :param torch.Tensor targets: tensor with first dimension as batch.
    """

    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # have to use contiguous since they may from a torch.view op
        iflat = inputs.contiguous().view(-1)
        tflat = targets.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(tflat * iflat)
        B_sum = torch.sum(tflat * tflat)

        return 1 - ((2. * intersection + self.smooth) / (A_sum + B_sum + self.smooth))
