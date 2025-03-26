from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CIFAR10Dataset(Dataset):
    def __init__(self, normalization="cifar10", loading="torchvision", root_dir="./data"):
        super().__init__()
        self.train_set = None
        self.test_set = None
        self.root_dir = root_dir
        self.loading = loading
        self.normalization = normalization
        self.mean = self.set_normalization(normalization)["mean"]
        self.std = self.set_normalization(normalization)["std"]

        self.train_set = self.get_dataset(
            train=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            ),
        )

        self.test_set = self.get_dataset(
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            ),
        )

    def set_normalization(self, normalization):
        # Image classification on the CIFAR10 dataset - Albumentations Documentation
        # https://albumentations.ai/docs/autoalbument/examples/cifar10/
        if normalization == "cifar10":
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2471, 0.2435, 0.2616)
        elif normalization == "imagenet":
            # ImageNet - torchbench Docs https://paperswithcode.github.io/torchbench/imagenet/
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        else:
            raise NotImplementedError
        return {"mean": mean, "std": std}

    def get_dataset(self, train, transform, download=True):
        if self.loading == "torchvision":
            dataset = CIFAR10(
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


class CIFAR10DatasetNoAugmentation(CIFAR10Dataset):
    def __init__(self, normalization="cifar10", loading="torchvision", root_dir="./data"):
        # Call the parent constructor to set up the test_set and other attributes
        super().__init__(normalization=normalization, loading=loading, root_dir=root_dir)

        # Override the train_set with no augmentation transformations
        self.train_set = self.get_dataset(
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            ),
        )


class CIFAR10DatasetExtendedAugmentation(CIFAR10Dataset):
    def __init__(self, normalization="cifar10", loading="torchvision", root_dir="./data"):
        # Call the parent constructor to set up the test_set and other attributes
        super().__init__(normalization=normalization, loading=loading, root_dir=root_dir)

        # Override the train_set with extended augmentation transformations
        self.train_set = self.get_dataset(
            train=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]),
        )
