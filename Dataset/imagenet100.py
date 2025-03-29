from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os,sys


class ImageNet100Dataset(Dataset):
    # Singleton
    imagenet100_train = None
    imagenet100_val = None

    def __init__(self):
        super().__init__()

        self.train_set = None
        self.test_set = None


        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 可调
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
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