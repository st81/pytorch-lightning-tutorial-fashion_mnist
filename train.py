from argparse import ArgumentParser
import pytorch_lightning as pl

import models.bases as base_models
import datamodules.bases as base_datamodule


parser = ArgumentParser()

parser = base_models.LitModel.add_model_specific_args(parser)
args = parser.parse_args()
print(args)

model = base_models.LitModel(args)
trainer = pl.Trainer(
    gpus=1, limit_train_batches=1.0, limit_val_batches=1.0, max_epochs=1
)
trainer.fit(model, datamodule=base_datamodule.FashionMNISTDataModule())
