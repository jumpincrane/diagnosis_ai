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


class fe_block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels

        self.conv11 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(15, 1), padding='same')
        self.conv12 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 15), padding='same')

        self.conv21 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(13, 1), padding='same')
        self.conv22 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 13), padding='same')


        self.conv31 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(11, 1), padding='same')
        self.conv32 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 11), padding='same')

        self.conv41 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(9, 1), padding='same')
        self.conv42 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 9), padding='same')

        self.conv5 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding='same')
        self.conv6 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1), padding='same')


    def forward(self, x):
        
        c1 = self.conv11(x)
        c1 = self.conv12(c1)

        c2 = self.conv21(x)
        c2 = self.conv22(c2)

        c3 = self.conv31(x)
        c3 = self.conv32(c3)

        c4 = self.conv41(x)
        c4 = self.conv42(c4)
     
        output = c1 + c2 + c3 + c4 + x
        output = self.conv5(output)
        output = self.conv6(output)

        return output




class Encoder(nn.Module):
    def __init__(self, in_channels=4, n_filters=64, kernel_size=(3, 3), dropout=0.3, maxpool=(2, 2), batchnorm=False):
        super().__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.batchnorm = batchnorm

        self.conv1 = conv_block(self.in_channels, self.n_filters * 1, self.kernel_size, self.batchnorm) # (64, 240, 240)
        self.maxpool1 = nn.MaxPool2d(maxpool) # (64, 120, 120)
        # self.dropout1 = nn.Dropout(dropout)
        self.conv2 = conv_block(self.n_filters * 1, self.n_filters * 2, self.kernel_size, self.batchnorm) # (128, 120, 120)
        self.maxpool2 = nn.MaxPool2d(maxpool) # (128, 60, 60)
        # self.dropout2 = nn.Dropout(dropout)
        self.conv3 = conv_block(self.n_filters * 2, self.n_filters * 4, self.kernel_size, self.batchnorm)# (256, 60, 60)
        self.maxpool3 = nn.MaxPool2d(maxpool) # (256, 30, 30)
        # self.dropout3 = nn.Dropout(dropout)
        self.conv4 = conv_block(self.n_filters * 4, self.n_filters * 8, self.kernel_size, self.batchnorm) # (512, 30, 30)
        self.maxpool4 = nn.MaxPool2d(maxpool) # (512, 15, 15)
        # self.dropout4 = nn.Dropout(dropout)
        self.conv5 = conv_block(self.n_filters * 8, self.n_filters * 16, self.kernel_size, self.batchnorm) # (1024, 15, 15)

    def forward(self, x):
        c1 = self.conv1(x)
  
        p1 = self.maxpool1(c1)
        c2 = self.conv2(p1)

        p2 = self.maxpool2(c2)
        c3 = self.conv3(p2)
  
        p3 = self.maxpool3(c3)
        c4 = self.conv4(p3)
      
        p4 = self.maxpool4(c4)
        c5 = self.conv5(p4)

        enc_conv_outs = [c1, c2, c3, c4, c5]

        return enc_conv_outs

class Decoder(nn.Module):
    def __init__(self, out_channels=4, n_filters=64, kernel_size=(3, 3), dropout=0.3, upsample_scale=2, padding=1, batchnorm=False):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.batchnorm = batchnorm
        self.out_channels = out_channels

        self.fe1 = fe_block(self.n_filters * 1)
        self.convt6 = nn.ConvTranspose2d(self.n_filters * 16, self.n_filters * 8, kernel_size=self.kernel_size, padding=self.padding)
        self.upsample6 = nn.UpsamplingBilinear2d(scale_factor=upsample_scale)
        self.dropout6 = nn.Dropout(dropout)
        self.conv6 = conv_block(self.n_filters * 16, self.n_filters * 8, kernel_size = self.kernel_size, batchnorm = self.batchnorm)


        self.fe2 = fe_block(self.n_filters * 2)
        self.convt7 = nn.ConvTranspose2d(self.n_filters * 8, self.n_filters * 4, kernel_size=self.kernel_size, padding=self.padding)
        self.upsample7 = nn.UpsamplingBilinear2d(scale_factor=upsample_scale)
        self.dropout7 = nn.Dropout(dropout)
        self.conv7 = conv_block(self.n_filters * 16, self.n_filters * 4, kernel_size = self.kernel_size, batchnorm = self.batchnorm)


        self.fe3 = fe_block(self.n_filters * 4)
        self.convt8 = nn.ConvTranspose2d(self.n_filters * 4, self.n_filters * 2, kernel_size=self.kernel_size, padding=self.padding)
        self.upsample8 = nn.UpsamplingBilinear2d(scale_factor=upsample_scale)
        self.dropout8 = nn.Dropout(dropout)
        self.conv8 = conv_block(self.n_filters * 16, self.n_filters * 2, kernel_size = self.kernel_size, batchnorm = self.batchnorm)


        self.fe4 = fe_block(self.n_filters * 8)
        self.convt9 = nn.ConvTranspose2d(self.n_filters * 2, self.n_filters * 1, kernel_size=self.kernel_size, padding=self.padding)
        self.upsample9 = nn.UpsamplingBilinear2d(scale_factor=upsample_scale)
        self.dropout9 = nn.Dropout(dropout)
        self.conv9 = conv_block(self.n_filters * 16, self.n_filters * 1, kernel_size = self.kernel_size, batchnorm = self.batchnorm)

    def forward(self, encFeatures):
        c1, c2, c3, c4, c5 = encFeatures

        fe1 = self.fe1(c1) # (64, 240, 240)
        fe2 = self.fe2(c2) # (128, 120, 120)
        fe3 = self.fe3(c3) # (256, 60, 60)
        fe4 = self.fe4(c4) # (512, 30, 30)

        u6 = self.convt6(c5)
        u6 = self.upsample6(u6)
        u6 = t.cat([u6, fe4], dim=1) # (512, 30, 30) + (512, 30, 30) = (1024, 30, 30)
        u6 = self.dropout6(u6)
        c6 = self.conv6(u6) # (512, 30, 30)

        fe44 = self.upsample6(fe4)
        u7 = self.convt7(c6) # (256, 30, 30)
        u7 = self.upsample7(u7) # (256, 60, 60)
        u7 = t.cat([u7, fe3, fe44], dim=1) # (256, 60, 60) + (256, 60, 60) + (512, 60, 60)= (1024, 60, 60)
        u7 = self.dropout7(u7)
        c7 = self.conv7(u7) # (256, 60, 60)

        fe444 = self.upsample6(fe44)
        fe33 = self.upsample6(fe3)
        u8 = self.convt8(c7) # (128, 60, 60)
        u8 = self.upsample8(u8) # (128, 120, 120)
        u8 = t.cat([u8, fe2, fe33, fe444], dim=1) # (128, 120, 120) + (128, 120, 120) + (256, 60, 60) + (512, 30, 30) = (1024, 120, 120)
        u8 = self.dropout8(u8)
        c8 = self.conv8(u8) # (128, 120, 120)

        fe4444 = self.upsample6(fe444)
        fe333 = self.upsample6(fe33)
        fe22 = self.upsample6(fe2)
        u9 = self.convt9(c8) # (64, 120, 120)
        u9 = self.upsample9(u9) # (64, 240, 240)
        u9 = t.cat([u9, fe1, fe22, fe333, fe4444], dim=1) # (64, 240, 240) + (64, 240, 240) + (128, 120, 120) + (256, 60, 60) + (512, 30, 30) = (1024, 240, 240)
        u9 = self.dropout9(u9)
        c9 = self.conv9(u9) # (6, 240, 240)

        return c9


class BrainSegUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, n_filters=64):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters = n_filters

        self.encoder = Encoder(self.in_channels, self.n_filters, batchnorm=True)
        self.decoder = Decoder(self.out_channels, self.n_filters, batchnorm=True)

        self.last_layer = nn.Conv2d(self.n_filters * 1, self.out_channels, kernel_size=(1, 1))
        self.last_activ = nn.Sigmoid()
    def forward(self, x):
        encFeatures = self.encoder(x)
        output = self.decoder(encFeatures)
        output = self.last_layer(output)
        output = self.last_activ(output)

        return output