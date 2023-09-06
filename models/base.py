import torch
from torch import Tensor
from lightning.pytorch import LightningModule
from models import TASK_MULTICLASS


class BaseModel(LightningModule):
    def __init__(self,
                num_classes:int=10,
                input_shape:tuple=(3,224,224),
                name:str=None,
                **kwargs):
        super().__init__()
        self.save_hyperparameters() # saves all the hyperparameters passed to init
        self.name = name or self.__class__.__name__
        self.task = TASK_MULTICLASS
        self.num_classes = num_classes
        self._input_shape = input_shape

    def forward(self, batch) -> Tensor:
        return self.model(batch)

    @property
    def input_shape(self) -> tuple:
        return self._input_shape

    @input_shape.setter
    def input_shape(self, shape:tuple):
        self._input_shape = shape

    @property
    def example_input_array(self) -> Tensor:
        return torch.rand(1, *self.input_shape)
