from pathlib import Path
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.datasets.utils import extract_archive
from torchvision.datasets import CIFAR10
from lightning import LightningDataModule
from data.common import DatasetFromSubset
from data.dataset import CropDisease
from utils.dataset import make_weighted_random_sampler
from config import reproducible
from config import PATH_DATA, PATH_DATA_FARMSALL


# For reproducibility
SEED = reproducible()
CWD = Path(__file__).parent


class OpensetDataModule(LightningDataModule):
    openset = CIFAR10

    def __init__(self, transform:dict=None,
                    batch_size:int=32, num_workers:int=0, pin_memory:bool=False):
        super().__init__()
        self.transform = transform or {'train': None, 'valid': None}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
        self.trainset = None
        self.valset = None
        self.testset = None

    def prepare_data(self, src:str=None):   # how to download, tokenize, etc…
        pass

    def setup(self, stage:str): # how to split, define dataset, etc…
        if stage in ['fit', 'valid']:
            trainset = self.openset(PATH_DATA, download=False, train=True)
            train_set_size = int(len(trainset) * 0.8)
            valid_set_size = len(trainset) - train_set_size
            # TODO: DatasetFromSubset 대신 transform을 치환하는 방법으로 바꿀 것
            train_set, valid_set = data.random_split(trainset, [train_set_size, valid_set_size], generator=SEED)
            self.trainset = DatasetFromSubset(train_set, transform=self.transform['train'])
            self.valset = DatasetFromSubset(valid_set, transform=self.transform['valid'])
        elif stage in ['test', 'predict']:
            self.testset = self.predict = self.openset(PATH_DATA, download=False, train=False, transform=self.transform['valid'])

    def train_dataloader(self, num_workers=None, pin_memory:bool=None, shuffle:bool=True, use_sampler:bool=False) -> DataLoader:
        sampler = make_weighted_random_sampler(self.trainset) if use_sampler else None
        shuffle = False if use_sampler else shuffle
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=shuffle,
                            num_workers=num_workers or self.num_workers,
                            pin_memory=pin_memory or self.pin_memory,
                            sampler=sampler)

    def val_dataloader(self, num_workers=None, pin_memory:bool=None) -> DataLoader:
        return DataLoader(self.valset, batch_size=self.batch_size,
                            num_workers=num_workers or self.num_workers,
                            pin_memory=pin_memory or self.pin_memory)

    def test_dataloader(self, num_workers=None, pin_memory:bool=None) -> DataLoader:
        return DataLoader(self.testset, batch_size=self.batch_size,
                            num_workers=num_workers or self.num_workers,
                            pin_memory=pin_memory or self.pin_memory)

    def predict_dataloader(self, num_workers=None, pin_memory:bool=None) -> DataLoader:
        return DataLoader(self.predict, batch_size=self.batch_size,
                            num_workers=num_workers or self.num_workers,
                            pin_memory=pin_memory or self.pin_memory)

    def teardown(self, stage:str):
        pass

    @property
    def classes(self) -> list:
        if hasattr(self.trainset, 'classes'):
            return self.trainset.classes
        return None

    @property
    def class_to_idx(self) -> list:
        if hasattr(self.trainset, 'class_to_idx'):
            return self.trainset.class_to_idx
        return None

    @property
    def labels(self) -> list:
        if hasattr(self.trainset, 'labels'):
            return self.trainset.labels
        return None


class CropDiseaseDataModule(OpensetDataModule):
    
    def prepare_data(self, src:str=None):   # how to download, tokenize, etc…
        if src is not None and PATH_DATA_FARMSALL.exists() is False:
            extract_archive(src, PATH_DATA_FARMSALL.parent)

    def setup(self, stage:str): # how to split, define dataset, etc…
        if stage == 'fit':
            self.trainset = CropDisease(PATH_DATA_FARMSALL, kind='train', transform=self.transform['train'])
        if stage == 'valid':
            self.valset = CropDisease(PATH_DATA_FARMSALL, kind='valid', transform=self.transform['valid'])
        if stage == 'test':
            self.testset = CropDisease(PATH_DATA_FARMSALL, kind='valid', transform=self.transform['valid'])
        if stage == 'predict':
            self.predict = CropDisease(PATH_DATA_FARMSALL, kind='valid', transform=self.transform['valid'])


def get_dataset_info(task_name:str, root:str=None) -> dict:

    if task_name != 'farmsall':
        root = root or PATH_DATA
        ds = OpensetDataModule.openset(root=root, download=False, train=False)
    else:
        root = root or PATH_DATA_FARMSALL
        ds = CropDisease(PATH_DATA_FARMSALL, kind='valid')

    info = {'num_classes': len(ds.classes),
            'class_to_idx': ds.class_to_idx,
            'classes': ds.classes,
            'labels': ds.labels if hasattr(ds, 'labels') else None
            }
    return info