from typing import Tuple
from pathlib import Path
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.io import read_image
from torchvision.datasets import ImageFolder, CIFAR10
CIFAR10.url="http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


# de facto values from ImageNet
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CROP_SIZE = 224
SIZE = 256


def get_default_transforms(mean:list=MEAN, std:list=STD, crop_size:int=CROP_SIZE, size:int=SIZE) -> dict:
    return {'train': transforms.Compose(
                                [
                                    transforms.RandomResizedCrop(crop_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std),
                                ]
                            ),
            'valid': transforms.Compose(
                        [
                            transforms.Resize(size),
                            transforms.CenterCrop(crop_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean, std=std),
                        ]
                    )
            }


class DatasetFromSubset(Dataset):
    def __init__(self, subset:Subset, transform=None, target_transform=None):
        self.dataset = subset
        if transform:
            self.dataset.dataset.transform = transform
        if target_transform:
            self.dataset.dataset.target_transform = target_transform
        
        # Extract and store targets if they are available in the original dataset
        if hasattr(subset.dataset, 'targets'):
            self.targets = [subset.dataset.targets[i] for i in subset.indices]
        else:
            self.targets = [subset.dataset[i][1] for i in subset.indices]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[Tensor, int]:
        return self.dataset[idx]

    @property
    def classes(self) -> list:
        if hasattr(self.dataset.dataset, 'classes'):
            return self.dataset.dataset.classes
        return None

    @property
    def class_to_idx(self) -> list:
        if hasattr(self.dataset.dataset, 'class_to_idx'):
            return self.dataset.dataset.class_to_idx
        return None

    @property
    def labels(self) -> list:
        if hasattr(self.dataset.dataset, 'labels'):
            return self.dataset.dataset.labels
        return None


class ImageDatasetCSV(Dataset):
    def __init__(self, img_dir, label_file, transform=None, target_transform=None):
        super().__init__()
        self.img_labels = pd.read_csv(label_file)   # filename, label
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx) -> Tuple[Tensor, int]:
        img_path = Path(self.img_dir) / self.img_labels.iloc[idx, 0]
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label