from typing import Union, Any

import torch
import torch.nn as nn


def _conv1(in_planes: int, planes: int, stride: int = 1, mode: str = "2D") -> Union[nn.Conv2d, nn.Conv3d]:
    if mode == "2D":
        return nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
    elif mode == "3D":
        return nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride, bias=False)


def _conv3(in_planes: int, planes: int, stride: int = 1, mode: str = "2D") -> Union[nn.Conv2d, nn.Conv3d]:
    if mode == "2D":
        return nn.Conv2d(in_channels=in_planes, out_channels=planes,
                         kernel_size=3, stride=stride, bias=False, padding=1)
    elif mode == "3D":
        return nn.Conv3d(in_channels=in_planes, out_channels=planes,
                         kernel_size=3, stride=stride, bias=False, padding=1)


def _batch_norm(planes: int, mode: str = "2D") -> Union[nn.BatchNorm2d, nn.BatchNorm3d]:
    if mode == "2D":
        return nn.BatchNorm2d(planes)
    elif mode == "3D":
        return nn.BatchNorm3d(planes)


def _max_pool(kernel_size: int = 3, stride: int = 2, padding: int = 1, mode: str = "2D") -> Union[nn.MaxPool2d,
                                                                                                  nn.MaxPool3d]:
    if mode == "2D":
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    elif mode == "3D":
        return nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Union[nn.Module, None] = None,
                 mode: str = "2D"):
        super().__init__()
        self.conv1 = _conv3(in_planes, planes, stride, mode)
        self.bn1 = _batch_norm(planes, mode)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3(planes, planes, stride=1, mode=mode)
        self.bn2 = _batch_norm(planes, mode)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class _Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Union[nn.Module, None] = None,
                 mode: str = "2D"):
        super().__init__()

        self.conv1 = _conv1(in_planes, planes, stride=1, mode=mode)
        self.bn1 = _batch_norm(planes, mode)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = _conv3(planes, planes, stride=stride, mode=mode)
        self.bn2 = _batch_norm(planes, mode)

        self.conv3 = _conv1(planes, planes * self.expansion, stride=1, mode=mode)
        self.bn3 = _batch_norm(planes * self.expansion, mode=mode)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetEncoder(nn.Module):
    """
    Encoder based on resnet architecture.
    """

    def __init__(self,
                 block: Union[_BasicBlock, _Bottleneck],
                 layers: list[int],
                 block_inplanes: list[int],
                 n_in_channels: int = 3,
                 mode: str = "2D",
                 conv1_t_size: int = 7,
                 return_all_layers: bool = False):
        super().__init__()

        self.in_planes = block_inplanes[0]
        self.return_all_layers = return_all_layers

        if mode == "2D":
            self.conv1 = nn.Conv2d(n_in_channels,
                                   self.in_planes,
                                   kernel_size=(conv1_t_size, 7),
                                   stride=2,
                                   padding=(conv1_t_size // 2, 3),
                                   bias=False)
        elif mode == "3D":
            self.conv1 = nn.Conv3d(n_in_channels,
                                   self.in_planes,
                                   kernel_size=(conv1_t_size, 7, 7),
                                   stride=2,
                                   padding=(conv1_t_size // 2, 3, 3),
                                   bias=False)
        else:
            raise ValueError(f"Choose 2D or 3D mode, but not {mode}")

        self.bn1 = _batch_norm(self.in_planes, mode)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = _max_pool(kernel_size=3, stride=2, padding=1, mode=mode)

        self.layer0 = nn.Sequential(self.conv1, self.bn1, self.relu)
        self.layer1 = nn.Sequential(self.maxpool, self._make_layer(block, block_inplanes[0], layers[0], mode))
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], mode, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], mode, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], mode, stride=2)

        self.output_features = block.expansion * block_inplanes[-1]

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Union[_BasicBlock, _Bottleneck],
                    planes: int, blocks: int, mode: str = "2D", stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1(self.in_planes, planes * block.expansion, stride=stride, mode=mode),
                _batch_norm(planes * block.expansion, mode=mode))
        layers = []
        layers.append(block(in_planes=self.in_planes, planes=planes, stride=stride, downsample=downsample, mode=mode))

        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, mode=mode))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, list[torch.Tensor]]:
        if not self.return_all_layers:

            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            return x
        else:
            x0 = nn.Identity()(x)
            x1 = self.layer0(x0)
            x2 = self.layer1(x1)
            x3 = self.layer2(x2)
            x4 = self.layer3(x3)
            x5 = self.layer4(x4)

            return [x0, x1, x2, x3, x4, x5]


def resnet(resnet_model: int, n_in_channels: int, mode="2D", **kwargs) -> ResNetEncoder:
    """
    :param int resnet_model: Depth of ResNet, choose from {18, 34, 50, 101, 152},
    :param int n_in_channels: input channels,
    :param str mode: 2D or 3D ResNet type.

    :return ResNetEncoder.
    """
    if resnet_model == 18:
        model = ResNetEncoder(_BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512], n_in_channels, mode, **kwargs)
    if resnet_model == 34:
        model = ResNetEncoder(_BasicBlock, [3, 4, 6, 3], [64, 128, 256, 512], n_in_channels, mode, **kwargs)
    if resnet_model == 50:
        model = ResNetEncoder(_Bottleneck, [3, 4, 6, 3], [64, 128, 256, 512], n_in_channels, mode, **kwargs)
    if resnet_model == 101:
        model = ResNetEncoder(_Bottleneck, [3, 4, 23, 3], [64, 128, 256, 512], n_in_channels, mode, **kwargs)
    if resnet_model == 152:
        model = ResNetEncoder(_Bottleneck, [3, 8, 36, 3], [64, 128, 256, 512], n_in_channels, mode, **kwargs)

    return model
