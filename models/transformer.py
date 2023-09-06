import numpy as np
from torch import Tensor

from transformers import AutoConfig
from transformers import ViTForImageClassification, ViTFeatureExtractor, ViTImageProcessor
from models.base import BaseModel


class ViTClassifierHF(BaseModel):

    def __init__(self,
                model_name_or_path:str,
                config:AutoConfig=None,
                num_classes:int=10,
                # id2label:dict=None,
                name:str=None,
                **kwargs
    ):
        super().__init__(num_classes=num_classes,
                        input_shape=None,
                        name=name or name.replace('/','_'),
                        **kwargs)
        
        # self.lr = learning_rate
        # self.id2label = id2label
        # self.label2id = {value:key for key, value in self.id2label.items()}

        # Use the pretrained model to classify arbitrary image classes
        # self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        custom_config = kwargs
        custom_config.update({'num_labels': self.num_classes})
        try:
            self.config = config or AutoConfig.from_pretrained(model_name_or_path, **custom_config)
        except:
            self.config = custom_config
        self.model = ViTForImageClassification.from_pretrained(
                                    model_name_or_path,
                                    config=self.config,
                                    ignore_mismatched_sizes=True # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
                                    )
        self.image_processor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

    @property
    def input_params(self) -> dict:
        size = 256  # if xxx
        return {
                'size': size,
                'crop_size': self.image_processor.size['height'],
                'mean': self.image_processor.image_mean,
                'std': self.image_processor.image_std
                }

    @property
    def input_shape(self) -> tuple:
        return (3, self.image_processor.size['width'], self.image_processor.size['height'])

    def forward(self, batch, batch_idx:int=None) -> Tensor:
        outputs = self.model(pixel_values=batch) # HuggingFace 모델의 경우, outputs 객체를 리턴함
        return outputs.logits
    
    def features(self, batch) -> Tensor:    # For OOD detector: ViM, Malananobis
        outputs = self.model.base_model(batch, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state
        last_feature = last_hidden_state[:, 0, :]  # [CLS] token is at index 0
        return last_feature.view(-1, np.prod(last_feature.shape[1:]))

    @property
    def fc(self):   # For OOD detector: ViM
        return self.model.classifier


class DeitClassifierHF(ViTClassifierHF):
    def __init__(self,
                model_name_or_path:str,
                config:AutoConfig=None,
                num_classes:int=10,
                # id2label:dict=None,
                name:str=None,
                **kwargs):
        super().__init__(model_name_or_path,
                        config=config,
                        num_classes=num_classes,
                        name=name,
                        **kwargs)
        self.image_processor = ViTImageProcessor.from_pretrained(model_name_or_path)


MODELS = {'vit16': [ViTClassifierHF, dict(model_name_or_path='google/vit-base-patch16-224')],
            # 'vit16-21k': [ViTClassifierHF, dict(model_name_or_path='google/vit-base-patch16–224-in21k')],
            'deit16': [DeitClassifierHF, dict(model_name_or_path='facebook/deit-base-patch16-224')],
            'deit16-distill': [DeitClassifierHF, dict(model_name_or_path='facebook/deit-base-distilled-patch16-224')],
            # 'beit16': [ViTClassifierHF, dict(model_name_or_path='microsoft/beit-base-patch16–224')]
}

def get_model_instance(model_name, **kwargs):
    model_factory, model_kwargs = MODELS.get(model_name)
    if model_factory is None:
        raise NotImplementedError
    model_kwargs.update(kwargs)
    return model_factory(name=model_name, **model_kwargs)