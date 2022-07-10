import torch as t
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_channels = 1, nfilter = 16, kernel_size = 3, batchnorm = False):
        super().__init__()

        self.batchnorm = batchnorm

        self.blockconv1 = nn.Conv2d(in_channels=in_channels, out_channels=nfilter, kernel_size=kernel_size, padding='same')
        self.blockconv2 = nn.Conv2d(in_channels=nfilter, out_channels=nfilter, kernel_size=kernel_size, padding='same')
        self.batch_norm = nn.BatchNorm2d(num_features=nfilter)
        self.activ_func = nn.ReLU()
    
    def forward(self, x):
        x = self.blockconv1(x)
        if self.batchnorm:
            x = self.batch_norm(x)
        x = self.activ_func(x)

        x = self.blockconv2(x)
        if self.batchnorm:
            x = self.batch_norm(x)
        x = self.activ_func(x)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, n_filters=16, kernel_size=(3, 3), dropout=0.1, maxpool=(2, 2), batchnorm=False):
        super().__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.batchnorm = batchnorm

        self.conv1 = conv_block(self.in_channels, self.n_filters * 1, self.kernel_size, self.batchnorm)
        self.maxpool1 = nn.MaxPool2d(maxpool)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = conv_block(self.n_filters * 1, self.n_filters * 2, self.kernel_size, self.batchnorm)
        self.maxpool2 = nn.MaxPool2d(maxpool)
        self.dropout2 = nn.Dropout(dropout)
        self.conv3 = conv_block(self.n_filters * 2, self.n_filters * 4, self.kernel_size, self.batchnorm)
        self.maxpool3 = nn.MaxPool2d(maxpool)
        self.dropout3 = nn.Dropout(dropout)
        self.conv4 = conv_block(self.n_filters * 4, self.n_filters * 8, self.kernel_size, self.batchnorm)
        self.maxpool4 = nn.MaxPool2d(maxpool)
        self.dropout4 = nn.Dropout(dropout)
        self.conv5 = conv_block(self.n_filters * 8, self.n_filters * 16, self.kernel_size, self.batchnorm)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.maxpool1(c1)
        p1 = self.dropout1(p1)

        c2 = self.conv2(p1)
        p2 = self.maxpool2(c2)
        p2 = self.dropout2(p2)

        c3 = self.conv3(p2)
        p3 = self.maxpool3(c3)
        p3 = self.dropout3(p3)

        c4 = self.conv4(p3)
        p4 = self.maxpool4(c4)
        p4 = self.dropout4(p4)

        c5 = self.conv5(p4)

        enc_conv_outs = [c1, c2, c3, c4, c5]

        return enc_conv_outs

class Decoder(nn.Module):
    def __init__(self, out_channels, n_filters=16, kernel_size=(3, 3), dropout=0.1, upsample_scale=2, padding=1, batchnorm=False):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.batchnorm = batchnorm
        self.out_channels = out_channels

        self.convt6 = nn.ConvTranspose2d(self.n_filters * 16, self.n_filters * 8, kernel_size=self.kernel_size, padding=self.padding)
        self.upsample6 = nn.UpsamplingBilinear2d(scale_factor=upsample_scale)
        self.dropout6 = nn.Dropout(dropout)
        self.conv6 = conv_block(self.n_filters * 16, self.n_filters * 8, kernel_size = self.kernel_size, batchnorm = self.batchnorm)

        self.convt7 = nn.ConvTranspose2d(self.n_filters * 8, self.n_filters * 4, kernel_size=self.kernel_size, padding=self.padding)
        self.upsample7 = nn.UpsamplingBilinear2d(scale_factor=upsample_scale)
        self.dropout7 = nn.Dropout(dropout)
        self.conv7 = conv_block(self.n_filters * 8, self.n_filters * 4, kernel_size = self.kernel_size, batchnorm = self.batchnorm)

        self.convt8 = nn.ConvTranspose2d(self.n_filters * 4, self.n_filters * 2, kernel_size=self.kernel_size, padding=self.padding)
        self.upsample8 = nn.UpsamplingBilinear2d(scale_factor=upsample_scale)
        self.dropout8 = nn.Dropout(dropout)
        self.conv8 = conv_block(self.n_filters * 4, self.n_filters * 2, kernel_size = self.kernel_size, batchnorm = self.batchnorm)

        self.convt9 = nn.ConvTranspose2d(self.n_filters * 2, self.n_filters * 1, kernel_size=self.kernel_size, padding=self.padding)
        self.upsample9 = nn.UpsamplingBilinear2d(scale_factor=upsample_scale)
        self.dropout9 = nn.Dropout(dropout)
        self.conv9 = conv_block(self.n_filters * 2, self.n_filters * 1, kernel_size = self.kernel_size, batchnorm = self.batchnorm)

    def forward(self, encFeatures):
        c1, c2, c3, c4, c5 = encFeatures

        u6 = self.convt6(c5)
        u6 = self.upsample6(u6)
        u6 = t.cat([u6, c4], dim=1)
        u6 = self.dropout6(u6)
        c6 = self.conv6(u6)

        u7 = self.convt7(c6)
        u7 = self.upsample7(u7)
        u7 = t.cat([u7, c3], dim=1)
        u7 = self.dropout7(u7)
        c7 = self.conv7(u7)

        u8 = self.convt8(c7)
        u8 = self.upsample8(u8)
        u8 = t.cat([u8, c2], dim=1)
        u8 = self.dropout8(u8)
        c8 = self.conv8(u8)

        u9 = self.convt9(c8)
        u9 = self.upsample9(u9)
        u9 = t.cat([u9, c1], dim=1)
        u9 = self.dropout9(u9)
        c9 = self.conv9(u9)

        return c9


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters = n_filters

        self.encoder = Encoder(self.in_channels, self.n_filters, batchnorm=True)
        self.decoder = Decoder(self.out_channels, self.n_filters, batchnorm=True)

        self.last_layer = nn.Conv2d(self.n_filters * 1, self.out_channels, kernel_size=1)
    def forward(self, x):
        encFeatures = self.encoder(x)
        output = self.decoder(encFeatures)
        output = self.last_layer(output)
        

        return output