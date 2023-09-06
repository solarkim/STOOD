import unittest
from pathlib import Path
from collections import Counter, defaultdict
import random
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData, MNIST
from torchvision import transforms
from utils.dataset import make_weighted_random_sampler, check_balanced, print_dataloader


CWD = Path(__file__).parent
TEMP = CWD / 'temp'
TEMP.mkdir(exist_ok=True, parents=True)


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.num_classes = 10
        # self.dataset = FakeData(100, image_size=(1,32,32), num_classes=self.num_classes, transform=transforms.ToTensor())
        self.dataset = MNIST(CWD.parent / 'examples' / 'data', train=True, download=False, transform=transforms.ToTensor())

    def tearDown(self):
        pass

    def test_make_weighted_random_sampler(self):
        batch_size = 1000
        num_workers = 3

        sample_probs = torch.rand(len(self.dataset.classes))
        idx_to_del = [idx for idx, (_, label) in enumerate(self.dataset) if random.random() > sample_probs[label]] 
        self.imbalanced_dataset = copy.deepcopy(self.dataset)
        self.imbalanced_dataset.targets = np.delete(self.dataset.targets, idx_to_del, axis=0)
        self.imbalanced_dataset.data = np.delete(self.dataset.data, idx_to_del, axis=0)

        print('Check dataset:', check_balanced(self.imbalanced_dataset))
        # loader = DataLoader(self.imbalanced_dataset, batch_size=batch_size, num_workers=num_workers)
        # print_dataloader(loader)

        print('Weighted Random Sampler - replacement')
        sampler = make_weighted_random_sampler(self.imbalanced_dataset)
        loader = DataLoader(self.imbalanced_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
        print_dataloader(loader)

        # The following is to check how it works - actually, it does not resolve imbalance
        # print('Weighted Random Sampler - no replacement')
        # sampler = make_weighted_random_sampler(self.imbalanced_dataset, False)
        # loader = DataLoader(self.imbalanced_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
        # print_dataloader(loader)

        # from torchsampler import ImbalancedDatasetSampler
        # print('Torchsampler')
        # sampler = ImbalancedDatasetSampler(self.imbalanced_dataset)
        # loader = DataLoader(self.imbalanced_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
        # print_dataloader(loader)
    