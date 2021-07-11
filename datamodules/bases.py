from typing import Optional
import torch
from torch.utils.data import random_split
from torch.utils.data import dataloader
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl


class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.3,
        seed: int = 42,
        batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.train_size = train_size
        self.val_size = val_size
        self.seed = seed
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        FashionMNIST(
            root="data", train=True, download=True,
        )
        FashionMNIST(root="data", train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            dataset = FashionMNIST(
                root="data", train=True, download=False, transform=ToTensor(),
            )
            self.train_dataset, self.val_dataset = random_split(
                dataset,
                [
                    int(len(dataset) * self.train_size),
                    int(len(dataset) * self.val_size),
                ],
                generator=torch.Generator().manual_seed(self.seed),
            )
        if stage in (None, "test"):
            self.test_dataset = FashionMNIST(
                root="data", train=False, download=False, transform=ToTensor()
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
