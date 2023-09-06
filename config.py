from pathlib import Path
import torch
import lightning as L


CWD = Path(__file__).parent

PATH_LOG = CWD / 'logs'
PATH_MODEL = PATH_LOG / 'saved_models'
PATH_DATA = CWD / 'dataset'

# additional settings
PATH_DATA_FARMSALL = PATH_DATA / 'farmsall'
PATH_DATA_FARMSALL_OOD = PATH_DATA_FARMSALL / 'ood'


def reproducible(seed:int=42) -> torch.Generator:
    # Setting the seed
    L.seed_everything(seed)

    if seed is None:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        return None

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator =  torch.Generator().manual_seed(seed)
    return generator
