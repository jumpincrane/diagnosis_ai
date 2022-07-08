import torch
import torch.nn as nn
from pathlib import Path
import torchmetrics
from segmentation_models_pytorch import Unet
import pytorch_lightning as pl

def conv2d_block(input_tensor, in_channels, nfilter, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    # input = batch_size, 1, 240, 240
    x = nn.Conv2d(in_channels=in_channels, out_channels=nfilter, kernel_size=kernel_size, padding=1)(input_tensor)
    if batchnorm:
        x = nn.BatchNorm2d(num_features=nfilter)(x)
    x = nn.ReLU()(x)
    
    # second layer
    x = nn.Conv2d(in_channels=nfilter, out_channels=nfilter, kernel_size=kernel_size, padding=1)(x)
    if batchnorm:
        x = nn.BatchNorm2d(num_features=nfilter)(x)
    x = nn.ReLU()(x)
    
    return x


  # TODO: DONT NEED TO FLATTEN IN BOTTLENECK?
  # TODO: STRIDES IN DECODER?
def get_unet(input_img, in_channels = 1, n_filters = 16, dropout = 0.1, batchnorm = True):
    # Contracting Path
    c1 = conv2d_block(input_img, in_channels, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = nn.MaxPool2d((2, 2))(c1)
    p1 = nn.Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = nn.MaxPool2d((2, 2))(c2)
    p2 = nn.Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = nn.MaxPool2d((2, 2))(c3)
    p3 = nn.Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 4, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = nn.MaxPool2d((2, 2))(c4)
    p4 = nn.Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters * 8, n_filters * 16, kernel_size = 3, batchnorm = batchnorm)

    # Expansive Path
    u6 = nn.ConvTranspose2d(n_filters * 16, n_filters * 8, kernel_size=3, strides = 2, padding=1)(c5)
    u6 = torch.cat([u6, c4], dim=1)
    u6 = nn.Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 16, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, padding=1)

    u7 = nn.ConvTranspose2d(n_filters * 4, (3, 3), strides = 2, padding = 'same')(c6)
    u7 = torch.cat([u7, c3], dim=1)
    u7 = nn.Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)

    u8 = nn.ConvTranspose2d(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = torch.cat([u8, c2], dim=1)
    u8 = nn.Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)

    u9 = nn.ConvTranspose2d(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = torch.cat([u9, c1], dim=1)
    u9 = nn.Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

    outputs = nn.Conv2d(1, (1, 1), activation='sigmoid')(c9)
