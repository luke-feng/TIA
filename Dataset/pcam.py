from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose
from torchvision.datasets import SVHN
import sys
import os
import h5py
import numpy as np
from PIL import Image

class PCamData(Dataset):
    def __init__(self, data_path, label_path, transform=None):
        self.data_h5 = h5py.File(data_path, 'r')
        self.label_h5 = h5py.File(label_path, 'r')
        self.images = self.data_h5['x']  # shape: [N, 96, 96, 3]
        self.labels = self.label_h5['y']  # shape: [N, 1]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx][0])

        if self.transform:
            img = self.transform(img)

        return img, label
    

class NumpyToPIL:
    def __call__(self, x):
        return Image.fromarray(x.astype(np.uint8))
    
        

class PCAMdataset(Dataset):
    # Singleton
    pcam_train = None
    pcam_val = None

    def __init__(
            self
    ):
        super().__init__()
        self.train_set = None
        self.test_set = None

        # Singletons of MNIST train and test datasets
        transform = transforms.Compose([
            NumpyToPIL(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

        if PCAMdataset.pcam_train is None:
            PCAMdataset.pcam_train = PCamData(
                data_path=f'{sys.path[0]}/data/PCAM/camelyonpatch_level_2_split_train_x.h5',
                label_path=f'{sys.path[0]}/data/PCAM/camelyonpatch_level_2_split_train_y.h5',
                transform=transform
            )
            
        if PCAMdataset.pcam_val is None:
            PCAMdataset.pcam_val = PCamData(
                data_path=f'{sys.path[0]}/data/PCAM/camelyonpatch_level_2_split_test_x.h5',
                label_path=f'{sys.path[0]}/data/PCAM/camelyonpatch_level_2_split_test_y.h5',
                transform=transform
            )

        self.train_set = PCAMdataset.pcam_train
        self.test_set = PCAMdataset.pcam_val
