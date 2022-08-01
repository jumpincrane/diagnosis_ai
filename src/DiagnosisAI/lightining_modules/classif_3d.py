import pytorch_lightning as pl
import torchmetrics
import torch.nn as nn
import torch as t
from DiagnosisAI.utils.metrics import calc_metrics, calculate_type_errors
from DiagnosisAI.models.resnet3d import generate_model

class Classif3D(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.n_classes = 11
        self.network = generate_model(model_depth=18, n_classes=self.n_classes, n_input_channels=1)
        self.loss_function = nn.CrossEntropyLoss() # starting
        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None

    def forward(self, x):
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs.type(t.float32))

        outputs = nn.Softmax(dim=1)(outputs)
        labels = nn.functional.one_hot(labels, num_classes=self.n_classes)
        labels = labels.squeeze()

        loss = self.loss_function(outputs, labels.type(t.float32))


        
        self.log('train_loss', loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs.type(t.float32))

        outputs = nn.Softmax(dim=1)(outputs)
        labels = nn.functional.one_hot(labels, num_classes=self.n_classes)
        labels = labels.squeeze()

        loss = self.loss_function(outputs, labels.type(t.float32))
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        return t.optim.AdamW(self.parameters(), lr=3e-4)



