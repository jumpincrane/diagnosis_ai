import torch
import torch.nn as nn
from pathlib import Path
import torchmetrics
from segmentation_models_pytorch import Unet
import pytorch_lightning as pl



class Segmenter(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.network = Unet(encoder_name='resnet18', encoder_weights='imagenet', classes=3)
        self.loss_function = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)

        self.accuracy(outputs, labels.type(torch.int64))
        self.log('train_acc', self.accuracy, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        
        loss = self.loss_function(outputs, labels)
        self.log('val_loss', loss, prog_bar=True)

        self.accuracy(outputs, labels.type(torch.int64))
        self.log('val_acc', self.accuracy, prog_bar=True)

    def configure_optimizers(self):
        # Tym razem użyjmy optimizera Adam - uczenie powinno być szybsze
        return torch.optim.Adam(self.parameters(), lr=1e-3)