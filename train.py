from argparse import ArgumentParser
from pytorch_lightning import Trainer

from models.bases import LitModel
from datamodules.bases import FashionMNISTDataModule


parser = ArgumentParser()

parser = LitModel.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

model = LitModel(args)
trainer = Trainer.from_argparse_args(args)
trainer.fit(model, datamodule=FashionMNISTDataModule())
