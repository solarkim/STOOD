#%%
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import torch
from torchvision import transforms
import torchmetrics.functional as F
import lightning as L
from pytorch_ood.detector import EnergyBased
from data.common import get_default_transforms
from data.clf_loaders import get_dataset_info
from data.ood_loaders import OpensetOODDataModule, CropDiseaseOODDataModule
from trainers import ClassificationTask
from config import reproducible
from config import PATH_MODEL, PATH_LOG


# For reproducibility
SEED = reproducible()


CWD = Path(__file__).parent

task_name = 'farmsall'   # 선택 - PoC:'cifar10', Farmsall:'farmsall'
model_name = 'vit16'
log_version = 0    # None: next avaiable version
log_path = PATH_LOG / f'{task_name}_{model_name}' / f'version_{log_version}'
ckpt_file = log_path / 'checkpoints' / 'epoch=39-step=204360.ckpt'

dsinfo = get_dataset_info(task_name=task_name)
num_classes = dsinfo['num_classes']
device = 'cuda'  # 'cuda' or 'cpu'
num_workers = 8     # Data Loader 동시 작업 수
simple_transformation = False


#%%
print("STAGE 1: Creating or loading a trained model")

if False:
    from pytorch_ood.model import WideResNet
    model = WideResNet(num_classes=num_classes, pretrained="cifar10-pt")
else:
    from models import get_model

    model_instance = get_model(model_name, num_classes=num_classes)
    task = ClassificationTask.load_from_checkpoint(ckpt_file,
                                            model=model_instance
                                            # 추가 parameter들이 들어갈 수 있음
                                            )
    model = task.model  # the loaded model

# presets for the next step
model.eval()
fc = model.fc
input_params = model.input_params
size = input_params['size']
crop_size = input_params['crop_size']
mean = input_params['mean']
std = input_params['std']

#%%
print("STAGE 2: Creating DataLoaders")

if simple_transformation:
    transform = {'train': transforms.Compose([transforms.Resize(size),
                                    transforms.CenterCrop(crop_size),
                                    transforms.ToTensor()])}
    transform['valid'] = transform['train']
else:
    transform = get_default_transforms(mean=mean, std=std, crop_size=crop_size, size=size)

if task_name != 'farmsall':    # For PoC
    dm = OpensetOODDataModule(transform=transform, num_workers=num_workers, pin_memory=True)
else:
    dm = CropDiseaseOODDataModule(transform=transform, num_workers=num_workers, pin_memory=True)

dm.setup('fit')
fit_loader_in = dm.fit_dataloader()
dm.setup('train')
train_loader_in_ood = dm.train_dataloader(use_sampler=False)
test_loader_in_ood = dm.test_dataloader()

#%%
print("STAGE 3: Creating OOD Detectors")
from models.ood import get_detector_instance

detectors_to_search = {
            # "ASH": dict(),    # TODO: implement backbone
            'SHE': dict(head=model.fc),
            "ViM": dict(d=64, w=model.fc.weight, b=model.fc.bias),
            "KLMatching": dict(),
            "MaxLogit": dict(),
            "Entropy": dict(),
            'DICE': dict(w=model.fc.weight, b=model.fc.bias, p=0.65),
            "RMD": dict(),
            # "KNN": dict(),    # Bug
            # "ReAct": dict(head=model.fc, detector=EnergyBased.score), # TODO: implement backbone
            "EnergyBased": dict(),
            "Mahalanobis": dict(norm_std=std, eps=0.002),
            # "ODIN": dict(norm_std=std, eps=0.002),
            "MaxSoftmax": dict(),
}
detectors = {name: get_detector_instance(name, model, device=device, **kwargs) for name, kwargs in detectors_to_search.items()}
# fit detectors to training data (some require this, some do not)
for name, detector in detectors.items():
    print(f"--> Fitting {name} {'Start:' :<6} {datetime.now()}")
    detector.fit(fit_loader_in, device=device)
    print(f"--> Fitting {name} {'End:' :<6} {datetime.now()}")

#%%
print(f"STAGE 4: Searching optimal thresholds on {task_name} dataset.")

from models.ood import save_detector
from utils.metric import eval_ood_detector


PATH_RESULT = log_path / f"{task_name}_{model_name}_{datetime.now().strftime('%y-%m-%dT%H:%M:%S')}"
PATH_RESULT.mkdir(exist_ok=True, parents=True)

results = []
for detector_name, detector in detectors.items():
    print(f"> Training {detector_name} starts at {datetime.now()}")
    result = {"Detector": detector_name, "Model": model_name}
    eval, _, _ = eval_ood_detector(detector, train_loader_in_ood, device=device, search_by='f1')
    result.update(eval)
    path_detector = PATH_RESULT / f'ood_detector_{task_name}_{model_name}_{detector_name}.pkl'
    detector.threshold = result['threshold']
    save_detector(path_detector, detector)
    result.update({'Path': str(path_detector.relative_to(path_detector.parent.parent))})
    results.append(result)

#%%
df = pd.DataFrame(results)
df.to_csv(PATH_RESULT / f'train_ood_models_{task_name}_{model_name}.csv', index=False)
print('Evaluation Results:\n', df)


#%%
print(f"STAGE 5: Evaluating detectors on {task_name} dataset.")
from models.ood import load_detector

results = []
confusion_matrices = {}
for detector_name, detector in detectors.items():
    print(f"> Evaluating {detector_name} starts at {datetime.now()}")
    path_detector = PATH_RESULT / f'ood_detector_{task_name}_{model_name}_{detector_name}.pkl'
    detector = load_detector(path_detector)
    result = {"Detector": detector_name, "Model": model_name}
    eval, scores, labels = eval_ood_detector(detector, test_loader_in_ood, threshold=detector.threshold, device=device)
    result.update(eval)
    result.update({'Path': str(path_detector.relative_to(path_detector.parent.parent))})
    results.append(result)

    cm = F.confusion_matrix(scores, labels, task='binary', threshold=detector.threshold)
    cm = cm.detach().cpu().tolist()
    confusion_matrices[detector_name] = cm
    

#%%
df = pd.DataFrame(results)
df.to_csv(PATH_RESULT / f'evaluation_ood_models_{task_name}_{model_name}.csv', index=False)
print('Evaluation Results:\n', df)

with (PATH_RESULT / 'evaluation_confusion_matrices.json').open('w') as fp:
    json.dump(confusion_matrices, fp, indent=4)
# %%
