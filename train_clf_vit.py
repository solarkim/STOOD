from pathlib import Path

import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from data.common import get_default_transforms
from data.clf_loaders import OpensetDataModule, CropDiseaseDataModule, get_dataset_info
from trainers import ClassificationTask, ConfusionMatrixCB, EvaluationCB
from models.transformer import get_model_instance
from config import reproducible
from config import PATH_LOG, PATH_MODEL, PATH_DATA_FARMSALL


# For reproducibility
SEED = reproducible()


CWD = Path(__file__).parent

task_name = 'farmsall' # 선택 - PoC:'cifar10', Farmsall:'farmsall'
model_name = 'vit16'
dsinfo = get_dataset_info(task_name=task_name)
num_classes = dsinfo['num_classes']
num_workers = 3     # Data Loader 동시 작업 수


model = get_model_instance(model_name, num_classes=num_classes)

# presets for the next step
input_params = model.input_params
size = input_params['size']
crop_size = input_params['crop_size']
mean = input_params['mean']
std = input_params['std']

transform = get_default_transforms(mean=mean, std=std, crop_size=crop_size, size=size)
if task_name != 'farmsall':    # For PoC
    dm = OpensetDataModule(transform=transform, num_workers=num_workers, pin_memory=True)
else:
    dm = CropDiseaseDataModule(transform=transform, num_workers=num_workers, pin_memory=True)
    if PATH_DATA_FARMSALL.exists() is False:
        dm.prepare_data('org/farmsall_dataset.tar.gz')

dm.setup('fit')
train_loader = dm.train_dataloader(use_sampler=True)
dm.setup('valid')
valid_loader = dm.val_dataloader()
dm.setup('test')
test_loader = dm.test_dataloader()


task = ClassificationTask(model, learning_rate=3e-4,
                            precision=True, recall=True, f1=True, confusion_matrix=True
                        )
logger = TensorBoardLogger(PATH_LOG, name=task_name,
                            log_graph=True,
                            # default_hp_metric=False   # if you want to turn it off
)
early_stop_callback = EarlyStopping(
                            monitor=task.VALIDATION_LOSS,
                            patience=3,
                            strict=False,
                            verbose=False,
                            mode='min'
)
trainer = Trainer(
                    max_epochs=40,
                    callbacks=[#early_stop_callback, 
                                ConfusionMatrixCB(),
                                EvaluationCB()],
                    devices=1, # the number of GPUs
                    accelerator='gpu',
                    # default_root_dir=CWD/'logs',   # location to save check points
                    logger=logger
                    # deterministic=True,
                    
)
trainer.fit(task, train_dataloaders=train_loader, val_dataloaders=valid_loader,
            # automatically restores model, epoch, step, LR schedulers, etc...
            # ckpt_path="some/path/to/my_checkpoint.ckpt"
)
trainer.save_checkpoint(PATH_MODEL / f"best_model_{task_name}_{model.name}.ckpt")

# Run in test phase
trainer.test(task, dataloaders=valid_loader)
