import pytorch_lightning as pl

import models.bases as base_models
import datamodules.bases as base_datamodule


model = base_models.LitModel()
trainer = pl.Trainer(
    gpus=1, limit_train_batches=1.0, limit_val_batches=1.0, max_epochs=10
)
trainer.fit(model, datamodule=base_datamodule.FashionMNISTDataModule())
