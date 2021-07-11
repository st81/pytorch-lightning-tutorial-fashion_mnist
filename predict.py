import pytorch_lightning as pl
import models.bases as base_models
import datamodules.bases as base_datamodules


model = base_models.LitModel.load_from_checkpoint(
    "lightning_logs/version_22/checkpoints/epoch=9-step=6569.ckpt"
)
trainer = pl.Trainer(gpus=1)
trainer.test(model, datamodule=base_datamodules.FashionMNISTDataModule())
