import pytorch_lightning as pl
from segmentation_models_pytorch import Unet
import torchmetrics
from DiagnosisAI.models.BrainSegNet import BrainSegUNet
import torch.nn as nn
import torch as t

class BinarySegmenterUnet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.network = Unet(encoder_name='resnet34', classes=1, in_channels=4)
        self.loss_function = nn.BCEWithLogitsLoss() # for binary 
        metrics = torchmetrics.MetricCollection([torchmetrics.Precision(num_classes=1, average='macro', mdmc_average='samplewise'),
                                                 torchmetrics.Recall(num_classes=1, average='macro', mdmc_average='samplewise'),
                                                 torchmetrics.F1Score(num_classes=1, average='macro', mdmc_average='samplewise'),
                                                 torchmetrics.Accuracy(num_classes=1, average='macro', mdmc_average='samplewise'),
                                                 torchmetrics.JaccardIndex(num_classes=2)
                                                ])
        self.train_metrics = metrics.clone('train_')
        self.val_metrics = metrics.clone('val_')

    def forward(self, x):
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)
        self.log('train_loss', loss)
        self.log_dict(self.train_metrics(outputs.view(-1), labels.type(t.int32).view(-1)))
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log_dict(self.val_metrics(outputs.view(-1), labels.type(t.int32).view(-1)))

    def configure_optimizers(self):
        return t.optim.Adam(self.parameters(), lr=1e-3)



class BinarySegmenterBrainSegNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.network = BrainSegUNet(in_channels=4, out_channels=1, n_filters=64)
        self.loss_function = nn.BCEWithLogitsLoss() # for binary 
        metrics = torchmetrics.MetricCollection([torchmetrics.Precision(num_classes=1, average='macro', mdmc_average='samplewise'),
                                                 torchmetrics.Recall(num_classes=1, average='macro', mdmc_average='samplewise'),
                                                 torchmetrics.F1Score(num_classes=1, average='macro', mdmc_average='samplewise'),
                                                 torchmetrics.Accuracy(num_classes=1, average='macro', mdmc_average='samplewise'),
                                                 torchmetrics.JaccardIndex(num_classes=2)
                                                ])
        self.train_metrics = metrics.clone('train_')
        self.val_metrics = metrics.clone('val_')

    def forward(self, x):
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)
        self.log('train_loss', loss)
        self.log_dict(self.train_metrics(outputs.view(-1), labels.type(t.int32).view(-1)))
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log_dict(self.val_metrics(outputs.view(-1), labels.type(t.int32).view(-1)))

    def configure_optimizers(self):
        return t.optim.Adam(self.parameters(), lr=1e-3)