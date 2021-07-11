import typing
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import ToTensor


def load(train_size: float, val_size: float) -> typing.Tuple[Dataset, Dataset, Dataset]:
    dataset = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=ToTensor(),
    )
    train_dataset, val_dataset = random_split(
        dataset,
        [int(len(dataset) * train_size), int(len(dataset) * val_size)],
        generator=torch.Generator().manual_seed(42),
    )
    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return train_dataset, val_dataset, test_dataset
