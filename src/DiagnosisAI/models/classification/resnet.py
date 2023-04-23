import torch
import torch.nn as nn

from diagnosisai.models._encoders import resnet_3d


class ResNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        