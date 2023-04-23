import torch
import torch.nn as nn


def _conv1(in_planes, planes, stride=1, mode="2D"):
    if mode == "2D":
        return nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
    elif mode == "3D":
        return nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride, bias=False)


def _conv3(in_planes, planes, stride=1, mode="2D"):
    if mode == "2D":
        return nn.Conv2d(in_channels=in_planes, out_channels=planes,
                         kernel_size=3, stride=stride, bias=False, padding=1)
    elif mode == "3D":
        return nn.Conv3d(in_channels=in_planes, out_channels=planes,
                         kernel_size=3, stride=stride, bias=False, padding=1)


def _batch_norm(planes, mode="2D"):
    if mode == "2D":
        return nn.BatchNorm2d(planes)
    elif mode == "3D":
        return nn.BatchNorm3d(planes)


def _max_pool(kernel_size=3, stride=2, padding=1, mode="2D"):
    if mode == "2D":
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    elif mode == "3D":
        return nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding)


class _BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample=None, mode="2D"):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_planes, out_channels=planes,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, bias=False, padding=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
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


class _Bottleneck3D(nn.Module):
    expansion: int = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample=None, mode="2D") -> None:
        super().__init__()

        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)

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


class ResNetEncoder3D(nn.Module):
    """
    Encoder based on resnet architecture.
    """

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_in_channels=3,
                 mode="2D",
                 conv1_t_size=7,
                 return_all_layers=False):
        super().__init__()

        self.in_planes = block_inplanes[0]
        self.return_all_layers = return_all_layers

        self.conv1 = nn.Conv3d(n_in_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=2,
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)

        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer0 = nn.Sequential(self.conv1, self.bn1, self.relu)
        self.layer1 = nn.Sequential(self.maxpool, self._make_layer(block, block_inplanes[0], layers[0]))
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: _BasicBlock3D, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(in_planes=self.in_planes, planes=planes, stride=stride, downsample=downsample))

        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


def resnet_3d(resnet_model: int, n_in_channels: int, mode="2D", **kwargs):
    """
    :param int n_in_channels: resnet input channels,
    :param str mode: "2D" or "3D"
    """

    if resnet_model == 18:
        model = ResNetEncoder3D(_BasicBlock3D, [2, 2, 2, 2], [64, 128, 256, 512], n_in_channels, mode, **kwargs)
    if resnet_model == 34:
        model = ResNetEncoder3D(_BasicBlock3D, [3, 4, 6, 3], [64, 128, 256, 512], n_in_channels, mode, **kwargs)
    if resnet_model == 50:
        model = ResNetEncoder3D(_Bottleneck3D, [3, 4, 6, 3], [64, 128, 256, 512], n_in_channels, mode, **kwargs)
    if resnet_model == 101:
        model = ResNetEncoder3D(_Bottleneck3D, [3, 4, 23, 3], [64, 128, 256, 512], n_in_channels, mode, **kwargs)
    if resnet_model == 152:
        model = ResNetEncoder3D(_Bottleneck3D, [3, 8, 36, 3], [64, 128, 256, 512], n_in_channels, mode, **kwargs)

    return model
