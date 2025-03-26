from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import FashionMNIST


class FashionMNISTDataset(Dataset):
    def __init__(self, loading="torchvision", root_dir="./data"):
        super().__init__()
        self.train_set = None
        self.test_set = None
        self.root_dir = root_dir
        self.loading = loading

        self.train_set = self.get_dataset(
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )

        self.test_set = self.get_dataset(
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )

    def get_dataset(self, train, transform, download=True):
        if self.loading == "torchvision":
            dataset = FashionMNIST(
                root=self.root_dir,
                train=train,
                transform=transform,
                download=download,
            )
        elif self.loading == "custom":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return dataset
