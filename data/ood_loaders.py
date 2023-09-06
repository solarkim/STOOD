from pathlib import Path
from torch.utils.data import DataLoader
import torch.utils.data as data
from torch.utils.data.dataset import ConcatDataset, Subset
from torchvision.datasets import CIFAR10
from pytorch_ood.utils import ToRGB, ToUnknown
from pytorch_ood.dataset.img import (
    LSUNCrop,
    LSUNResize,
    Textures,
    TinyImageNetCrop,
    TinyImageNetResize,
)
from lightning import LightningDataModule
from data.dataset import CropDisease, Tomato104
from data.dataset import CucumberDisease, KoreanMelonDisease
from data.dataset import LettuceDisease, PapperDisease
from data.dataset import SquashDisease, WaterMelonDisease
from data.dataset import ZucchiniDisease, Zucchini2Disease
from utils.dataset import make_weighted_random_sampler, create_targets_list, make_subset
from data.common import DatasetFromSubset
from config import reproducible
from config import PATH_DATA, PATH_DATA_FARMSALL, PATH_DATA_FARMSALL_OOD


# For reproducibility
SEED = reproducible()


CWD = Path(__file__).parent


class OpensetOODDataModule(LightningDataModule):
    def __init__(self, transform:dict=None,
                    batch_size:int=32, num_workers:int=0, pin_memory:bool=False):
        super().__init__()
        self.transform = transform or {'train': None, 'valid': None}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
        self.fitset = None
        self.trainset = None
        self.valset = None
        self.testset = None
        
        self.ood_sets_train = []
        self.ood_sets_test = []
        self.ood_sets = [
                        Textures(PATH_DATA, transform=self.transform['valid'], target_transform=ToUnknown(), download=False)
                        # + LSUNResize(PATH_DATA, transform=self.transform['valid'], target_transform=ToUnknown(), download=True)
        ]
        self._split_ood_sets(self.ood_sets, 0.8)
        
    def _split_ood_sets(self, ood_sets:list, ratio:float=0.8):
        """ood_sets를 ratio로 분할하여, ood_sets_train 과 ood_sets_test에 할당함."""
        for ood_set in ood_sets:
            trainset_size = int(len(ood_set) * ratio)
            validset_size = len(ood_set) - trainset_size
            train, test = data.random_split(ood_set, [trainset_size, validset_size], generator=SEED)
            self.ood_sets_train.append(DatasetFromSubset(train))
            self.ood_sets_test.append(DatasetFromSubset(test))

    def prepare_data(self, src:str=None):   # how to download, tokenize, etc…
        pass

    def setup(self, stage:str): # how to split, define dataset, etc…
        if stage == 'fit':
            self.fitset = CIFAR10(root=PATH_DATA, train=True, transform=self.transform['valid'], download=False)
            self.trainset_in = self.fitset
        else:
            # Modify the candidate sets below
            # TODO: Using sampler to control the size of the OOD set.
            self.trainset_ood = ConcatDataset(self.ood_sets_train)
            self.trainset = self.trainset_in + self.trainset_ood
            self.trainset.targets = create_targets_list(self.trainset)   # preparing for sampling

            self.testset_in = CIFAR10(PATH_DATA, train=False, transform=self.transform['valid'], download=False)
            self.testset_ood = ConcatDataset(self.ood_sets_test)
            self.valset = self.testset_in + self.testset_ood
            self.testset = self.predict = self.valset
            self.testset.targets = create_targets_list(self.testset) # preparing for sampling

    def fit_dataloader(self, num_workers=None, pin_memory:bool=None) -> DataLoader:
        return DataLoader(self.fitset, batch_size=self.batch_size,
                            num_workers=num_workers or self.num_workers,
                            pin_memory=pin_memory or self.pin_memory)

    def train_dataloader(self, num_workers=None, pin_memory:bool=None, shuffle:bool=True, use_sampler:bool=False, subset:float=None) -> DataLoader:
        sampler = make_weighted_random_sampler(self.trainset) if use_sampler else None
        shuffle = False if use_sampler else shuffle
        trainset = make_subset(self.trainset, int(len(self.trainset)*subset)) if subset else self.trainset
        return DataLoader(trainset, batch_size=self.batch_size, shuffle=shuffle,
                            num_workers=num_workers or self.num_workers,
                            pin_memory=pin_memory or self.pin_memory,
                            sampler=sampler)

    def val_dataloader(self, num_workers=None, pin_memory:bool=None, subset:float=None) -> DataLoader:
        valset = make_subset(self.valset, int(len(self.valset)*subset)) if subset else self.valset
        return DataLoader(valset, batch_size=self.batch_size,
                            num_workers=num_workers or self.num_workers,
                            pin_memory=pin_memory or self.pin_memory)

    def test_dataloader(self, num_workers=None, pin_memory:bool=None, subset:float=None) -> DataLoader:
        testset = make_subset(self.testset, int(len(self.testset)*subset)) if subset else self.testset
        return DataLoader(testset, batch_size=self.batch_size,
                            num_workers=num_workers or self.num_workers,
                            pin_memory=pin_memory or self.pin_memory)

    def predict_dataloader(self, num_workers=None, pin_memory:bool=None, subset:float=None) -> DataLoader:
        predict = make_subset(self.predict, int(len(self.predict)*subset)) if subset else self.predict
        return DataLoader(predict, batch_size=self.batch_size,
                            num_workers=num_workers or self.num_workers,
                            pin_memory=pin_memory or self.pin_memory)

    def teardown(self, stage:str):
        pass

    def get_targets_from_subset(self, subset, concat_dataset):
        targets = []
        for index in subset.indices:
            # Find the dataset to which this index belongs
            dataset_idx = 0
            while index >= len(concat_dataset.datasets[dataset_idx]):
                index -= len(concat_dataset.datasets[dataset_idx])
                dataset_idx += 1
            # Retrieve the target from the underlying dataset
            _, target = concat_dataset.datasets[dataset_idx].samples[index]
            targets.append(target)
        return targets

    @property
    def counts(self):
        """Counts in-dist total and each class of ood sets"""
        def __collect(ood_sets):
            counts = {}
            for ood_set in ood_sets:
                if isinstance(ood_set, DatasetFromSubset):
                    name = ood_set.dataset.dataset.__class__.__name__
                elif isinstance(ood_set, Subset):
                    name = ood_set.dataset.__class__.__name__
                else:
                    name = ood_set.__class__.__name__
                counts[name] = len(ood_set)
            return counts

        counts_train = {self.fitset.__class__.__name__: len(self.fitset)}
        counts_test = {self.testset_in.__class__.__name__: len(self.testset_in)}
        counts_train.update(__collect(self.ood_sets_train))
        counts_test.update(__collect(self.ood_sets_test))
        return counts_train, counts_test


class CropDiseaseOODDataModule(OpensetOODDataModule):
    def __init__(self, transform:dict=None,
                    batch_size:int=32, num_workers:int=0, pin_memory:bool=False):
        super().__init__(transform=transform,
                        batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        self.ood_sets_train = [Tomato104(PATH_DATA_FARMSALL_OOD / 'tomato104', kind='train', transform=self.transform['valid'], target_transform=ToUnknown()),
                                CIFAR10(PATH_DATA_FARMSALL_OOD, train=True, transform=self.transform['valid'], target_transform=ToUnknown(), download=True),
                            ]
        self.ood_sets_test = [Tomato104(PATH_DATA_FARMSALL_OOD / 'tomato104', kind='valid', transform=self.transform['valid'], target_transform=ToUnknown()),
                                CIFAR10(PATH_DATA_FARMSALL_OOD, train=False, transform=self.transform['valid'], target_transform=ToUnknown(), download=True),
                            ]
        self.ood_sets = [
                    CucumberDisease(PATH_DATA_FARMSALL_OOD, kind=None, transform=self.transform['valid'], target_transform=ToUnknown()),
                    KoreanMelonDisease(PATH_DATA_FARMSALL_OOD, kind=None, transform=self.transform['valid'], target_transform=ToUnknown()),
                    LettuceDisease(PATH_DATA_FARMSALL_OOD, kind=None, transform=self.transform['valid'], target_transform=ToUnknown()),
                    PapperDisease(PATH_DATA_FARMSALL_OOD, kind=None, transform=self.transform['valid'], target_transform=ToUnknown()),
                    SquashDisease(PATH_DATA_FARMSALL_OOD, kind=None, transform=self.transform['valid'], target_transform=ToUnknown()),
                    WaterMelonDisease(PATH_DATA_FARMSALL_OOD, kind=None, transform=self.transform['valid'], target_transform=ToUnknown()),
                    ZucchiniDisease(PATH_DATA_FARMSALL_OOD, kind=None, transform=self.transform['valid'], target_transform=ToUnknown()),
                    Zucchini2Disease(PATH_DATA_FARMSALL_OOD, kind=None, transform=self.transform['valid'], target_transform=ToUnknown()),    
                    Textures(PATH_DATA, transform=self.transform['valid'], target_transform=ToUnknown(), download=True),
                    LSUNResize(PATH_DATA, transform=self.transform['valid'], target_transform=ToUnknown(), download=True)
                ]
        self._split_ood_sets(self.ood_sets, 0.8) # train/valid ratio
        
    def setup(self, stage:str=None): # how to split, define dataset, etc…
        if stage == 'fit':
            self.fitset = CropDisease(PATH_DATA_FARMSALL, kind='train', transform=self.transform['valid'])
            self.trainset_in = self.fitset
        else:
            # TODO: Using sampler to control the size of the OOD set.
            self.trainset_ood = ConcatDataset(self.ood_sets_train)
            self.trainset = self.trainset_in + self.trainset_ood
            # self.trainset.targets = create_targets_list(self.trainset)   # preparing for sampling

            self.testset_in = CropDisease(PATH_DATA_FARMSALL, kind='valid', transform=self.transform['valid'])
            self.testset_ood = ConcatDataset(self.ood_sets_test)
            self.valset = self.testset_in + self.testset_ood
            # self.valset.targets = create_targets_list(self.valset)   # preparing for sampling
            self.testset = self.predict = self.valset