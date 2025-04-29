from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os,sys


class ImageNet100Dataset(Dataset):
    # Singleton
    imagenet100_train = None
    imagenet100_val = None

    def __init__(self, img_size=64):
        super().__init__()

        self.train_set = None
        self.test_set = None


        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        
        if ImageNet100Dataset.imagenet100_train is None:
            ImageNet100Dataset.imagenet100_train = datasets.ImageFolder(
                root=os.path.join(f'{sys.path[0]}/data/imagenet100/', 'train.X'),
                transform=transform
            )
        
        if ImageNet100Dataset.imagenet100_val is None:
            ImageNet100Dataset.imagenet100_val = datasets.ImageFolder(
                root=os.path.join(f'{sys.path[0]}/data/imagenet100/', 'val.X'),
                transform=transform
            )

        self.train_set = ImageNet100Dataset.imagenet100_train
        self.test_set = ImageNet100Dataset.imagenet100_val