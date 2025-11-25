import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_dataloader(batch_size: int = 64, data_root: str = "./data") -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
