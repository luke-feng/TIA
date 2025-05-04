from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os,sys


class ImageNet10Dataset(Dataset):
    # Singleton
    imagenet10_train = None
    imagenet10_val = None

    def __init__(self, img_size=64):
        super().__init__()

        self.train_set = None
        self.test_set = None


        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        
        if ImageNet10Dataset.imagenet10_train is None:
            ImageNet10Dataset.imagenet10_train = datasets.ImageFolder(
                root=os.path.join(f'{sys.path[0]}/data/imagenet10/', 'train.X'),
                transform=transform
            )
        
        if ImageNet10Dataset.imagenet10_val is None:
            ImageNet10Dataset.imagenet10_val = datasets.ImageFolder(
                root=os.path.join(f'{sys.path[0]}/data/imagenet10/', 'val.X'),
                transform=transform
            )

        self.train_set = ImageNet10Dataset.imagenet10_train
        self.test_set = ImageNet10Dataset.imagenet10_val