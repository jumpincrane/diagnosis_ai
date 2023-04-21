import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet3d import resnet18_3d, resnet34_3d, resnet50_3d, resnet101_3d, resnet152_3d


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels, eps=1e-05, momentum=0.1,
                                  affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.attention1 = nn.Identity()
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels, eps=1e-05, momentum=0.1,
                                  affine=True, track_running_stats=True)
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=(2, 2, 2), mode="nearest")
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


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.center = nn.Identity()
        self._in_channels = [768, 384, 192, 128, 32]
        self._out_channels = [256, 128, 64, 32, 16]

        self.decoderblocks = [DecoderBlock(in_channels=in_ch, out_channels=out_ch)
                              for in_ch, out_ch in zip(self._in_channels, self._out_channels)]
        self.blocks = nn.ModuleList(self.decoderblocks)

    def forward(self, features):
        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class Unet3d(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, resnet_model_depth=34):
        super(Unet3d, self).__init__()

        assert resnet_model_depth in [18, 34, 50, 101, 152]

        self.enc = get_encoder(resnet_model_depth, as_encoder=True, n_input_channels=in_channels)
        self.dec = Decoder()
        self.final_conv = nn.Conv3d(16, out_channels, kernel_size=3, stride=1, padding=1)
        self.id1 = nn.Identity()
        self.id2 = nn.Identity()

    def forward(self, x):
        enc_features = self.enc(x)
        output = self.dec(enc_features)
        output = self.final_conv(output)
        output = self.id1(output)
        output = self.id2(output)

        return output


def get_encoder(resnet_model_depth, **kwargs):

    model = None

    if resnet_model_depth == 18:
        model = resnet18_3d(**kwargs)
    elif resnet_model_depth == 34:
        model = resnet34_3d(**kwargs)
    elif resnet_model_depth == 50:
        model = resnet50_3d(**kwargs)
    elif resnet_model_depth == 101:
        model = resnet101_3d(**kwargs)
    elif resnet_model_depth == 152:
        model = resnet152_3d(**kwargs)
    else:
        raise ValueError(f"Resnet of {resnet_model_depth} doesn't exists")

    return model
