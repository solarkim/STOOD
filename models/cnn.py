from typing import Tuple, Union
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torchvision import models as torchmodels
from lightning.pytorch import LightningModule
from transformers import AutoConfig
from transformers import AutoModelForImageClassification, AutoImageProcessor
from models.base import BaseModel


class CustomResnet50(BaseModel):
    # TODO: generalize to any resnet
    def __init__(self,
                num_classes:int=10,
                input_shape:tuple=(3,224,224),
                name:str=None,
                **kwargs):
        super().__init__(num_classes=num_classes,
                        input_shape=input_shape,
                        name=name,
                        **kwargs)

        # Use the pretrained model to classify arbitrary image classes
        self.model = torchmodels.resnet50(weights='DEFAULT')
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

        # Implement to provide features. Use 'get_graph_node_names' function to find out the return nodes.
        # https://pytorch.org/vision/stable/feature_extraction.html
        self.return_nodes = ['flatten']
    
    @property
    def input_params(self) -> dict:
        size = 256  # if xxx
        return {
                'size': size,
                'crop_size': self.input_shape[-1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
                }

    @property
    def fc(self):   # For OOD detector: ViM
        return self.model.fc


class CNNHF(BaseModel):
    def __init__(self,
                model_name_or_path:str,
                config:AutoConfig=None,
                num_classes:int=10,
                name:str=None,
                **kwargs):
        super().__init__(num_classes=num_classes,
                        name=name,
                        **kwargs)
        

        # Use the pretrained model to classify arbitrary image classes
        custom_config = kwargs
        custom_config.update({'num_labels': self.num_classes})
        self.config = config or AutoConfig.from_pretrained(model_name_or_path, **custom_config)
        self.model = AutoModelForImageClassification.from_pretrained(
                                            model_name_or_path,
                                            config=self.config,
                                            ignore_mismatched_sizes=True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )
        self.image_processor = AutoImageProcessor.from_pretrained(model_name_or_path)

    @property
    def input_params(self) -> dict:
        size = int(self.image_processor.size['shortest_edge'] / self.image_processor.crop_pct)
        return {
                'size': size,
                'crop_size': self.image_processor.size['shortest_edge'],
                'mean': self.image_processor.image_mean,
                'std': self.image_processor.image_std
                }

    @property
    def input_shape(self) -> tuple:
        return (3, self.image_processor.size['shortest_edge'], self.image_processor.size['shortest_edge'])

    def forward(self, batch) -> Tensor:
        outputs = self.model(batch) # HuggingFace 모델의 경우, outputs 객체를 리턴함
        return outputs.logits

    def features(self, batch) -> Tensor:
        outputs = self.model.base_model(batch, output_hidden_states=True)
        out = outputs.pooler_output
        return out.view(-1, np.prod(out.shape[1:]))

    @property
    def fc(self):   # For OOD detector: ViM
        return self.model.classifier[1]


MODELS = {'ms_resnet50': [CNNHF, dict(model_name_or_path='microsoft/resnet-50')],
            'resnet50': [CustomResnet50, dict()]
}

def get_model_instance(model_name, **kwargs):
    model_factory, model_kwargs = MODELS.get(model_name)
    if model_factory is None:
        raise NotImplementedError
    model_kwargs.update(kwargs)
    return model_factory(name=model_name, **model_kwargs)
    