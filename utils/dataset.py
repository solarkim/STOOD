from typing import Tuple
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler


def optimize_num_workers(dataset:Dataset, batch_size:int=32, pin_memory:bool=True) -> int:
    """Finds the optimal num_workers for Dataloader

    Args:
        dataset (Dataset): dataset to test
        batch_size (int, optional): batch size. Defaults to 32.
        pin_memory (bool, optional): pin memory or not. Defaults to True.

    Returns:
        int: optimal num_workers
    """
    from time import time
    import multiprocessing as mp
    best_num = None
    best_time = np.inf
    for num_workers in range(0, mp.cpu_count(), 2):  
        train_loader = DataLoader(dataset, shuffle=True, num_workers=num_workers,
                                    batch_size=batch_size, pin_memory=pin_memory)
        start = time()
        for _ in range(2):
            for _ in enumerate(train_loader):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

        if best_time > end - start:
            best_time = end - start
            best_num = num_workers

    return best_num


def concat_batch_predictions(outputs:list) -> Tuple[Tensor, Tensor]:
    preds = torch.Tensor([])
    probs = torch.Tensor([])

    for pred, prob in outputs:
        preds = torch.concat([preds, pred])
        probs = torch.concat([probs, prob])
    return preds, probs


from tqdm import tqdm
import shutil
def copy_folder(src:Path, dest:Path):
    for file in tqdm(src.iterdir()):
        shutil.copy(file, dest)


# coding: utf-8
import os
import tarfile


# tarfile example
def tar_dir(path):
    tar = tarfile.open('test.tar.gz', 'w:gz')

    for root, dirs, files in os.walk(path):
        for file_name in files:
            tar.add(os.path.join(root, file_name))

    tar.close()


def check_balanced(dataset:Dataset, name:str='', show:bool=False) -> dict:
    """Dataset에서 class당 빈도수를 계산해 줌.

    Args:
        dataset (Dataset): 분석 대상 데이터셋
        name (str, optional): show시 표시할 내용. Defaults to ''.
        show (bool, optional): 계산 결과를 print함. Defaults to False.

    Returns:
        dict: _description_
    """

    # Check if dataset has 'targets' attribute
    if hasattr(dataset, 'targets'):
        targets = dataset.targets.tolist() if isinstance(dataset.targets, torch.Tensor) else dataset.targets
    else:
        print("WARNING - check_balanced:", "in slow process")
        targets = [label for _, label in dataset]

    total_per_class = dict(Counter(targets))
    num_samples = sum(total_per_class.values())

    if show:
        print(f'{name} Total Samples: {num_samples}')
        for label, class_sum in total_per_class.items():
            print(f'{label}: {class_sum} - {class_sum / num_samples * 100:0.1f}%')

    return total_per_class


def print_dataloader(loader:DataLoader, show:bool=False) -> None:
    """DataLoader의 데이터셋에 대한 내용을 출력함 - Total samples, num per class"""

    total_per_class = defaultdict(int)
    for idx, (_, target) in enumerate(loader):
        count = Counter(target.numpy())
        if show:
            print(f'batch-{idx}, {dict(count)}')
        for label, num in count.items():
            total_per_class[label] += num
    print('Loaded counts:', dict(total_per_class))
    print('Total Samples:', sum(total_per_class.values()))


from collections import Counter
import torch

def make_weighted_random_sampler(dataset: Dataset, replacement: bool = True) -> WeightedRandomSampler:
    """Makes a random sampler for DataLoader."""

    # Try to get targets directly if possible
    targets = None
    if hasattr(dataset, 'targets'):
        targets = dataset.targets.tolist() if isinstance(dataset.targets, torch.Tensor) else dataset.targets
    elif hasattr(dataset, 'labels'):  # Some datasets might use 'labels' instead of 'targets'
        targets = dataset.labels

    # If targets are still not available, use the slow process
    if targets is None:
        print("WARNING - make_weighted_random_sampler:", "in slow process")
        targets = [label for _, label in dataset]

    # Count per class
    total_per_class = dict(Counter(targets))

    # Compute weights
    weights = {label: 1. / count for label, count in total_per_class.items()}   # sample less by proportion
    sample_weights = [weights[label] for label in targets]  # assign per-sample weight to choose

    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=replacement)


def make_subset(dataset:Dataset, num_samples:int) -> Subset:
    """Makes a subset of a Torch Dataset with a fixed number of samples.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to make a subset of.
        num_samples (int): The number of samples to include in the subset.

    Returns:
        torch.utils.data.Subset: The subset of the dataset.
    """
    num_samples = int(num_samples)
    indices = torch.randperm(len(dataset))[:num_samples]
    return Subset(dataset, indices)


def create_targets_list(dataset:Dataset) -> list:
    """Dataset에 labels를 리턴함. Subset을 제외한 Dataset을 상속한 경우 사용."""

    if hasattr(dataset, 'targets'):
        return dataset.targets
    
    assert isinstance(dataset, Subset) is False

    targets = []
    if hasattr(dataset, 'datasets'):    # ConcatDataset
        datasets = dataset.datasets
    else:
        datasets = [dataset]
    for dataset in datasets:
        if hasattr(dataset, 'targets'):
            subset = dataset.targets
        else:
            subset = create_targets_list(dataset)
        targets.extend(subset)
    return targets