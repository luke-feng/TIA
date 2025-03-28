from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose
from torchvision.datasets import SVHN
import sys
import os


class SVHNdataset(Dataset):
    # Singleton
    svhn_train = None
    svhn_val = None

    def __init__(
            self
    ):
        super().__init__()
        self.train_set = None
        self.test_set = None

        # Singletons of MNIST train and test datasets
        if not os.path.exists(f"{sys.path[0]}/data"):
            os.makedirs(f"{sys.path[0]}/data")

        if SVHNdataset.svhn_train is None:
            SVHNdataset.svhn_train = SVHN(
                f"{sys.path[0]}/data", split='train', download=True, transform=Compose([
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize((0.5, ), (0.5, ))
                                                                    ])
            )

        if SVHNdataset.svhn_val is None:
            SVHNdataset.svhn_val = SVHN(
                f"{sys.path[0]}/data", split='test', download=True, transform=Compose([
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize((0.5, ), (0.5, ))
                                                                    ])
            )

        self.train_set = SVHNdataset.svhn_train
        self.test_set = SVHNdataset.svhn_val
