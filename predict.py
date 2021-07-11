import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import models.bases as base_models
import utils.data as data_utils

model = base_models.LitModel.load_from_checkpoint(
    "lightning_logs/version_3/checkpoints/epoch=28-step=121799.ckpt"
)

_, _, test_dataset = data_utils.load(.9, .1)
test_dataloader = DataLoader(test_dataset)

trainer = pl.Trainer(gpus=1)
trainer.test(model, test_dataloader)

# image, label = test_[0]
# pred = model(image)
# plt.imshow(image[0])
# plt.show()
# print(label)
# print(pred.argmax(1))