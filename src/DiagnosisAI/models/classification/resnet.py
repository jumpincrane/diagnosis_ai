from typing import Union

import torch
import torch.nn as nn

from diagnosisai.models._encoders import resnet_encoder


def _choose_avg_pool(mode: str) -> Union[nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]:
    if mode == "2D":
        avg_pool = nn.AdaptiveAvgPool2d(1)
    elif mode == "3D":
        avg_pool = nn.AdaptiveAvgPool3d(1)
    else:
        raise ValueError(f"Choose 2D or 3D mode, but not {mode}")

    return avg_pool


def _choose_activ_func(activ_func_mode: str) -> Union[nn.Sigmoid, nn.Softmax, nn.Identity]:
    if activ_func_mode == "sigmoid":
        activ_func = nn.Sigmoid()
    elif activ_func_mode == "softmax":
        activ_func = nn.Softmax(dim=1)
    elif activ_func_mode == "none":
        activ_func = nn.Identity()
    else:
        raise ValueError("Choose from {sigmoid, softmax, none}")

    return activ_func


class ResNet(nn.Module):
    """
    ResNet model where are:
    - encoder,
    - average pooling,
    - fully connected layer,
    - activaction function.

    :param int num_classes: number of classes to predict,
    :param int resnet_model: Depth of ResNet, choose from {18, 34, 50, 101, 152},
    :param int in_channels: number of input channels to the first layer,
    :param str mode: 2D or 3D ResNet type,
    :param str activ_func_mode: selection of the activation function on the result from {sigmoid, softmax, none}.
    """

    def __init__(self, num_classes: int, resnet_depth: int = 34, in_channels: int = 3, mode: str = "2D",
                 activ_func_mode: str = 'softmax'):
        super().__init__()

        self.resnet = resnet_encoder(resnet_depth, in_channels, mode)
        self.avg_pool = _choose_avg_pool(mode)
        self.fc = nn.Linear(self.resnet.output_features, num_classes)
        self.activ_func = _choose_activ_func(activ_func_mode)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.activ_func(x)

        return x
    
    def step(self, inputs: torch.Tensor, labels: torch.Tensor, loss_func: nn.Module) -> tuple[torch.Tensor]:
        outputs = self.forward(inputs)
        loss = loss_func(outputs, labels)

        return outputs, loss
