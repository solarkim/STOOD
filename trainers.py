# References
# https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/datamodules.html
# https://torchmetrics.readthedocs.io/en/latest/pages/lightning.html

from typing import Tuple
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
import torch
from torch import optim, nn, Tensor
from torch.optim import Optimizer
import torch.nn.functional as F
from torchmetrics import MetricCollection, Accuracy
from torchmetrics import Precision, Recall, F1Score
from lightning.pytorch import LightningModule, Callback, Trainer
from utils.vision import makegrid
from utils.metric import show_confusion_matrix
from torchmetrics import ConfusionMatrix


class ClassificationTask(LightningModule):
    
    TASK_BINARY = 'binary'
    TASK_MULTICLASS = 'multiclass'
    TASK_MULTILABLE = 'multilabel'
    DEFAULT_TASK = TASK_MULTICLASS
    
    TRAIN_LOSS = 'Train/Loss'
    TRAIN_ACC = 'Train/Top@1'
    VALIDATION_LOSS = 'Valid/Loss'
    VALIDATION_ACC = 'Valid/Top@1'
    TEST_LOSS = 'Test/Loss'
    TEST_ACC = 'Test/Top@1'

    def __init__(self,
        model:nn.Module,
        num_classes:int=None,
        optimizer:Optimizer=optim.Adam,
        learning_rate:float=1e-3,
        # adam_epsilon:float=1e-8,
        # warmup_steps:int=0,
        # weight_decay:float=0.0,
        precision:bool=False,
        recall:bool=False,
        f1:bool=False,
        confusion_matrix:bool=False,
        optional_top_k:int=None,
        optional_metrics:dict=None,
        **kwargs
        ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'precision', 'recall', 'f1', 'optional_top_k', 'optional_metrics']) # Saves all the hyper parameters passed in __init__

        self.model = model
        self.task = self.model.task if hasattr(self.model, 'task') else self.DEFAULT_TASK
        self.num_classes = self.model.num_classes if hasattr(self.model, 'num_classes') else num_classes

        self._define_metrics(precision, recall, f1, confusion_matrix, optional_top_k, optional_metrics)
        
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def _define_metrics(self, precision, recall, f1, confusion_matrix, optional_top_k, optional_metrics):
        metric = MetricCollection({'Top@1': Accuracy(task=self.task, num_classes=self.num_classes, top_k=1)})
        if optional_top_k:
            metric.add_metrics({f'Top@{optional_top_k}': Accuracy(task=self.task, num_classes=self.num_classes, top_k=optional_top_k)})
        if precision:
            metric.add_metrics({'Precision': Precision(task=self.task, num_classes=self.num_classes, average="macro", top_k=1)})
        if recall:
            metric.add_metrics({'Recall': Recall(task=self.task, num_classes=self.num_classes, average="macro", top_k=1)})
        if f1:
            metric.add_metrics({'F1-score': F1Score(task=self.task, num_classes=self.num_classes, average="macro", top_k=1)})
        if optional_metrics:
            metric.add_metrics(optional_metrics)
        
        self.train_metric = metric.clone(prefix='Train/')
        self.valid_metric = metric.clone(prefix='Valid/')
        self.test_metric = metric.clone(prefix='Test/')

        if confusion_matrix:
            self.train_cm = ConfusionMatrix(task=self.task, num_classes=self.num_classes, normalize='none')
            self.valid_cm = ConfusionMatrix(task=self.task, num_classes=self.num_classes, normalize='none')
            self.test_cm = ConfusionMatrix(task=self.task, num_classes=self.num_classes, normalize='none')
        else:
            self.train_cm = self.valid_cm = self.test_cm = None

    def on_train_start(self) -> None:
        # Add hyper parameters to keep
        # self.logger.log_hyperparams(self.hparams, {key: 0 for key in self.train_metric.keys()})
        # Add the model graph
        if hasattr(self.model, 'example_input_array'):
            with torch.no_grad():
                sampleImg = self.model.example_input_array.to(self.device)
                self.logger.experiment.add_graph(self.model, sampleImg)
                # self.logger.log_graph(self.model, sampleImg)
        super().on_train_start()

    def _shared_step(self, batch:Tensor, batch_index:int=0, metric=None, cm=None)-> Tuple[Tensor, Tensor]:
        inputs, labels = batch
        outputs = self.model(inputs)
        # HuggingFace 모델의 경우, outputs 객체를 리턴함
        logits = outputs if isinstance(outputs, Tensor) else outputs.logits 

        loss = F.cross_entropy(logits, labels)  # calculating the loss
        if metric is not None:
            self.log_dict(metric(logits, labels), on_epoch=True)
        if cm is not None:
            cm.update(logits, labels)
        return loss

    def training_step(self, batch:Tensor, batch_idx:int=0) -> Tensor:
        loss = self._shared_step(batch, batch_idx, self.train_metric, self.train_cm)
        self.log(self.TRAIN_LOSS, loss, on_epoch=True)
        return loss

    def validation_step(self, batch:Tensor, batch_idx:int=0) -> Tensor:
        loss = self._shared_step(batch, batch_idx, self.valid_metric, self.valid_cm)
        self.log(self.VALIDATION_LOSS, loss)
        return loss

    def test_step(self, batch:Tensor, batch_idx:int=0) -> Tensor:
        loss = self._shared_step(batch, batch_idx, self.test_metric, self.test_cm)
        self.log(self.TEST_LOSS, loss)
        return loss

    def predict_step(self, batch:Tensor, batch_idx:int=0, dataloader_idx=0) -> Tuple[Tensor, Tensor]:
        inputs, labels = batch
        logits = self.model(inputs)
        prob = F.softmax(logits, dim=-1)
        y_hats = logits.argmax(-1)
        return y_hats, prob

    def configure_optimizers(self) -> Optimizer:
        """Configure model specific optimizers and learning rates"""
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters
        # do not require weight_decay but just using AdamW out-of-the-box works fine.
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return optimizer

    # Visualization Tools
    def add_custom_histograms(self):
        # iterating through all parameters
        for name,params in self.named_parameters():
            self.logger.experiment.add_histogram(name,params,self.current_epoch)

    def showActivations(self, x:Tensor):
        """Adding images to tensorboard

        Args:
            x (Tensor): a reference image
        """
        # logging reference image        
        self.logger.experiment.add_image("input", Tensor.cpu(x[0][0]), self.current_epoch, dataformats="HW")

        # logging layer 1 activations        
        out = self.model.layer1(x)
        c = makegrid(out, 4)
        self.logger.experiment.add_image("layer 1", c, self.current_epoch, dataformats="HW")
            
        # logging layer 1 activations        
        out = self.model.layer2(out)
        c = makegrid(out, 8)
        self.logger.experiment.add_image("layer 2", c, self.current_epoch, dataformats="HW")

        # logging layer 1 activations        
        out = self.model.layer3(out)
        c = self.makegrid(out, 8)
        self.logger.experiment.add_image("layer 3", c, self.current_epoch, dataformats="HW")


class ConfusionMatrixCB(Callback):
    def _create_confution_matrix(self, trainer: Trainer, cm:list, name:str, classes:list=None) -> None:
        confmat = cm.compute().detach().cpu().numpy()
        # fig, ax = cm.plot(labels=classes)
        fig = show_confusion_matrix(confmat, classes=classes)
        plt.close(fig)
        cm.reset()  # This is necessary. Otherwise, it kept stacking the results after each epoch.
        trainer.logger.experiment.add_figure(name, fig, trainer.current_epoch)
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        cm = getattr(pl_module, 'train_cm', None)
        if cm:
            self._create_confution_matrix(trainer, cm, "Train/Confusion matrix", trainer.train_dataloader.dataset.classes)
        return super().on_validation_epoch_end(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        cm = getattr(pl_module, 'valid_cm', None)
        if cm:
            self._create_confution_matrix(trainer, cm, "Valid/Confusion matrix", trainer.val_dataloaders.dataset.classes)
        return super().on_validation_epoch_end(trainer, pl_module)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        cm = getattr(pl_module, 'test_cm', None)
        if cm:
            self._create_confution_matrix(trainer, cm, "Test/Confusion matrix", trainer.test_dataloaders.dataset.classes)
        return super().on_test_end(trainer, pl_module)


class EvaluationCB(Callback):
    def _save_evaluation(self, path:str, stat:dict, mode:str='w') -> None:
        with (Path(path) / 'evaluation.json').open(mode) as fp:
            json.dump({name: value.item() for name, value in stat.items()}, fp, indent=4)
    
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._save_evaluation(trainer.log_dir, trainer.logged_metrics)
        return super().on_train_end(trainer, pl_module)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._save_evaluation(trainer.log_dir, trainer.logged_metrics, 'a')
        return super().on_test_end(trainer, pl_module)


class CustomCallback(Callback):
    def __init__(self):
        """Additional callbacks - adding training time in minutes
        """
        self.time = 0
        self.acc_time = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.time = time.time()
    
    def on_train_epoch_end(self, trainer, pl_module):
        train_time = (time.time() - self.time) / 60 # minutes
        self.acc_time += train_time
        self.log("epoch/Time(min)", train_time, sync_dist=True)
        self.log("epoch/Time(Sum)", self.acc_time, sync_dist=True)
        self.time = 0