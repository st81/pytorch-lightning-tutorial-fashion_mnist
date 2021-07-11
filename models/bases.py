from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class LitModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        return parent_parser

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
        )
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/lr": self.hparams.learning_rate})

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.flatten(x)
        y_hat = self.linear_relu_stack(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("Loss/train", loss, on_step=False, on_epoch=True)
        self.train_acc(y_hat, y)
        return loss

    def training_epoch_end(self, outs):
        self.log("Accuracy/train", self.train_acc.compute())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("Loss/val", val_loss, on_step=False, on_epoch=True)
        self.val_acc(y_hat, y)
        return val_loss

    def validation_epoch_end(self, outs):
        self.log("Accuracy/val", self.val_acc.compute())

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("Test/loss", loss, on_step=False, on_epoch=True)
        self.test_acc(y_hat, y)
        return loss

    def test_epoch_end(self, outs):
        self.log("Test/accuracy", self.test_acc.compute())
