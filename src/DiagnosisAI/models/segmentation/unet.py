import torch.nn as nn
import torch
import torch.nn.functional as F

from diagnosisai.models._encoders import _conv3, _batch_norm, resnet_encoder


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: str = "2D"):
        super().__init__()

        self.conv1 = _conv3(in_channels, out_channels, mode=mode)
        self.bn1 = _batch_norm(out_channels, mode=mode)
        self.relu = nn.ReLU(inplace=True)
        self.attention1 = nn.Identity()
        self.conv2 = _conv3(out_channels, out_channels, mode=mode)
        self.bn2 = _batch_norm(out_channels, mode=mode)
        self.attention2 = nn.Identity()

    def forward(self, x: torch.Tensor, skip=None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.attention2(x)

        return x


class _Decoder(nn.Module):
    def __init__(self, mode: str = "2D"):
        super(_Decoder, self).__init__()

        self.center = nn.Identity()
        self._in_channels = [768, 384, 192, 128, 32]
        self._out_channels = [256, 128, 64, 32, 16]

        self.decoderblocks = [_DecoderBlock(in_channels=in_ch, out_channels=out_ch, mode=mode)
                              for in_ch, out_ch in zip(self._in_channels, self._out_channels)]
        self.blocks = nn.ModuleList(self.decoderblocks)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class UNet(nn.Module):
    # TODO: some problems with padding and 3D
    """
    UNet model with skip connections with encoder based on ResNet architecture.

    :param int in_channels: number of input channels to the first layer,
    :param int out_channels: number of output channels,
    :param str mode: 2D or 3D ResNet type,
    :param int resnet_model: Depth of ResNet, choose from {18, 34, 50, 101, 152}.
    """
    def __init__(self, in_channels: int = 4, out_channels=1, mode: str = "2D", resnet_model: int = 34):

        super(UNet, self).__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels

        self.enc = resnet_encoder(resnet_model, in_channels, mode, return_all_layers=True)
        self.dec = _Decoder(mode=mode)
        self.final_conv = _conv3(16, out_channels, mode=mode)
        self.id1 = nn.Identity()
        self.id2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_features = self.enc(x)
        output = self.dec(enc_features)
        output = self.final_conv(output)
        output = self.id1(output)
        output = self.id2(output)

        return output
    
    def step(self, inputs: torch.Tensor, labels: torch.Tensor, loss_func: nn.Module) -> tuple[torch.Tensor]:
        outputs = self.forward(inputs)

        loss = loss_func(outputs, labels)

        return outputs, loss
