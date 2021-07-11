import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class LitModel(pl.LightningModule):

    def __init__(self, batch_size: int = 64):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
        )
        self.batch_size = batch_size
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.flatten(x)
        y_hat = self.linear_relu_stack(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('Loss/train', loss, on_step=False, on_epoch=True)
        # correct = (y_hat.argmax(1) == y).type(torch.float).sum().item()
        # self.log('Accuracy/train', correct, on_step=False, on_epoch=True)
        self.accuracy(y_hat, y)
        return loss

    def training_epoch_end(self, outs):
        self.log('Accuracy/train', self.accuracy.compute())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log('Loss/val', val_loss, on_step=False, on_epoch=True)
        correct = (y_hat.argmax(1) == y).type(torch.float).sum().item()
        self.log('Accuracy/val', correct, on_step=False, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('Loss/test', loss, on_step=False, on_epoch=True)
        correct = (y_hat.argmax(1) == y).type(torch.float).sum().item()
        self.log('Accuracy/test', correct, on_step=False, on_epoch=True)
        return loss