import torch.nn as nn
import torch

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()

        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
             
        inputs = nn.functional.softmax(inputs, dim=1)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)  
        
        return 1 - dice