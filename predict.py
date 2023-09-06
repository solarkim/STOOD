#%%
from pathlib import Path
import json
import torch
from torchvision import transforms
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torchmetrics import functional as F

from config import PATH_MODEL, PATH_LOG
from data.common import get_default_transforms
from data.clf_loaders import OpensetDataModule, CropDiseaseDataModule, get_dataset_info
from trainers import ClassificationTask
from utils.dataset import concat_batch_predictions
from utils.metric import show_confusion_matrix
from models import get_model


CWD = Path(__file__).parent

#%% [markdown]
## Classification 모델 평가

#%%
model_name = 'vit16'
task_name = 'farmsall' # 선택 - PoC:'cifar10', Farmsall:'farmsall'
log_version = 16    # None: next avaiable version
log_path = PATH_LOG / task_name / f'version_{log_version}'
ckpt_file = log_path / 'checkpoints' / 'epoch=39-step=204360.ckpt' or PATH_MODEL / f'best_model_{task_name}_{model_name}.ckpt'
dsinfo = get_dataset_info(task_name=task_name)
num_classes = dsinfo['num_classes']
simple_transformation = False

#%%
model_instance = get_model(model_name, num_classes=num_classes)
task = ClassificationTask.load_from_checkpoint(ckpt_file,
                                        model=model_instance,
                                        # 추가 parameter들이 들어갈 수 있음
                                        )
model = task.model  # the loaded model

#%%
input_params = model.input_params
size = input_params['size']
crop_size = input_params['crop_size']
mean = input_params['mean']
std = input_params['std']

if simple_transformation:
    transform = {'train': transforms.ToTensor(), 'valid': transforms.ToTensor()}
else:
    transform = get_default_transforms(mean=mean, std=std, crop_size=crop_size, size=size)

if task_name != 'farmsall':    # For PoC
    dm = OpensetDataModule(transform=transform, num_workers=2)
else:
    dm = CropDiseaseDataModule(transform=transform, num_workers=2)

#%% [markdown]
### Test set 준비 및 prediction 실행

#%%
dm.setup('test')
test_loader = dm.test_dataloader()


if False:
    model.eval()
    device = 'cpu'
    if device == 'cuda':
        model = model.to(device)
    print('device:', model.device)
    preds = torch.Tensor([])
    # to_device
    with torch.no_grad():
        for batch, labels in test_loader:
            # img = transform(frame)
            # img = torch.from_numpy(img).float().unsqueeze(0).cuda(0)
            if device == 'cuda':
                batch = batch.to(device)
            logits = model(batch)
            pred = logits.argmax(-1).cpu()
            preds = torch.concat([preds, pred])
else:
    logger = TensorBoardLogger(PATH_LOG, name=task_name, version=log_version,
                            log_graph=True,
                            # default_hp_metric=False   # if you want to turn it off
    )
    trainer = Trainer(
                        logger=logger,
                        devices=1, # the number of GPUs
                        accelerator='gpu'
                    )
    outputs = trainer.predict(task, test_loader)
    preds, probs = concat_batch_predictions(outputs)

targets = torch.concat([labels for _, labels in test_loader])
accuracy = F.accuracy(preds, targets, task=model.task, num_classes=model.num_classes, top_k=1)
cm = F.confusion_matrix(preds, targets, task=model.task, num_classes=model.num_classes)

cm = cm.detach().cpu().tolist()
print('Accuracy:', accuracy.item())
print(cm)

with (log_path / 'evaluation.json').open('w') as fp:
    json.dump({'acc': accuracy.item(), 'cm': cm}, fp, indent=4)
show_confusion_matrix(cm, dsinfo['classes'], save_img=log_path / 'confusion_matrix')