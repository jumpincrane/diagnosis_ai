import torchmetrics
import torch.nn as nn
import torch as t
import pytorch_lightning as pl
from torchvision.models import resnet50

model = resnet50(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
network = nn.Sequential(
                        model,
                        nn.Linear(in_features=1000, out_features=3))

class ChestModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        
        self.network = nn.Sequential(
                        model,
                        nn.Linear(in_features=1000, out_features=3))
        self.loss_function = nn.CrossEntropyLoss() # dice
        metrics = torchmetrics.MetricCollection([torchmetrics.Precision(num_classes=3, average='macro', mdmc_average='samplewise'),
                                                 torchmetrics.Recall(num_classes=3, average='macro', mdmc_average='samplewise'),
                                                 torchmetrics.F1Score(num_classes=3, average='macro', mdmc_average='samplewise'),
                                                 torchmetrics.Accuracy(num_classes=3, average='macro', mdmc_average='samplewise'),
                                                ])
        self.train_metrics = metrics.clone('train_')
        self.val_metrics = metrics.clone('val_')


    # metoda forward bardzo przydatnaa do sieci, gdzie nie wszystkie warstwy są sekwencyjne
    def forward(self, x):  # x - tensor z danymi wejściowymi
        return self.network(x)

    def training_step(self, batch, batch_idx): # konieczna funkcja
        inputs, labels = batch # brak przenoszenia na karte graficzną
        outputs = self(inputs) # wywołanie self spowoduje również wywołanie metdy forward
        
        x = nn.Softmax(dim=1)(outputs)
        loss = self.loss_function(x, labels.long())

        out = t.argmax(x, dim=1)


        self.log('train_loss', loss)
        self.log_dict(self.train_metrics(out.view(-1), labels.view(-1)))

        return loss

    def validation_step(self, batch, batch_idx): 
        inputs, labels = batch 
        outputs = self(inputs) 

        x = nn.Softmax(dim=1)(outputs)
        loss = self.loss_function(x, labels.long())

        self.log('val_loss', loss, prog_bar=True)
        out = t.argmax(x, dim=1)
        self.log_dict(self.val_metrics(out.view(-1), labels.view(-1)))
    

    def configure_optimizers(self):
        # Tym razem użyjmy optimizera Adam - uczenie powinno być szybsze
        return t.optim.Adam(self.parameters(), lr=1e-4)
