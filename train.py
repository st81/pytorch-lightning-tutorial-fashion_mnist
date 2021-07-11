from torch.utils.data import DataLoader
import pytorch_lightning as pl

import utils.data as data_utils
import models.bases as base_models
import datamodules.bases as base_datamodule


train_dataset, val_dataset, _ = data_utils.load(0.7, 0.3)

train_dataloader = DataLoader(train_dataset, batch_size=64)
val_dataloader = DataLoader(val_dataset, batch_size=64)

model = base_models.LitModel()
trainer = pl.Trainer(gpus=1, limit_train_batches=1.0, limit_val_batches=1.0, max_epochs=10)
trainer.fit(model, datamodule=base_datamodule.FashionMNISTDataModule())