import torch
from torchvision import datasets, transforms


class BinaryMNIST:
    def __init__(self, data_dir, dynamic: bool = True, train: bool = True):
        """
        :param data_dir: location to save
        :param dynamic: dynamic / static binarisation
        :param train:
        """
        self.dataset = datasets.MNIST(
            data_dir,
            train=train,
            transform=transforms.ToTensor(),
            download=True
        )

        self.dynamic = dynamic

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]

        if self.dynamic:
            x = x.bernoulli()
        else:
            x = torch.round(x)

        return x, y
