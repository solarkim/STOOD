from typing import Tuple
import torch
from torch import nn
from PIL import Image
from collections import Counter
from torchvision import transforms
from pytorch_ood.api import Detector


TASK_BINARY = 'binary'
TASK_MULTICLASS = 'multiclass'
TASK_MULTILABLE = 'multilabel'


from models.cnn import get_model_instance as get_cnn_model
from models.transformer import get_model_instance as get_transfomer_model


def get_model(model_name:str, **kwargs):
    try:
        model_instance = get_cnn_model(model_name, **kwargs)
    except:
        try:
            model_instance = get_transfomer_model(model_name, **kwargs)
        except:
            return None
    return model_instance


def predict_img(model:nn.Module, img:Image, transform:transforms.Compose=None, device:str='cpu') -> int:
    
    model.eval()
    if device != str(model.device):
        model = model.to(device)
        print('device:', model.device)

    with torch.no_grad():
        if transform:
            input_img = transform(img)
            input_img = input_img.unsqueeze(0)  # increasing the dimension
        else:
            input_img = img
        if device:
            input_img = input_img.to(device)
        logits = model(input_img)
    pred = logits.argmax(-1).cpu()
    return pred.item()


def predict_ood(detector:Detector, img:Image, transform:transforms.Compose=None, device:str='cpu') -> bool:
    
    with torch.no_grad():
        if transform:
            input_img = transform(img)
            input_img = input_img.unsqueeze(0)  # increasing the dimension
        else:
            input_img = img
        if device:
            input_img = input_img.to(device)
        score = detector(input_img)
    prob = score.sigmoid()
    detection =  prob >= detector.threshold
    return detection.item(), prob.item()


def ensemble_oods(detectors:list, img:Image, transform:transforms.Compose=None, device:str='cpu') -> Tuple[bool, list]:
    
    with torch.no_grad():
        if transform:
            input_img = transform(img)
            input_img = input_img.unsqueeze(0)  # increasing the dimension
        else:
            input_img = img
        if device:
            input_img = input_img.to(device)
        predictions = [(detector(input_img).sigmoid() >= detector.threshold).item() for detector in detectors]
    # Count occurrences of each prediction
    vote_count = Counter(predictions)

    # Get the class with the highest vote
    majority_vote = vote_count.most_common(1)[0][0]
    return majority_vote, predictions


def predict(clf:nn.Module, detectors:list, img:Image, transform:transforms.Compose=None, device:str='cpu') -> Tuple[int, list]:
    
    input_img = transform(img) if transform else img
    input_img = input_img.unsqueeze(0)  # increasing the dimension
    input_img = input_img.to(device)
    if len(detectors) > 1:
        ood, scores = ensemble_oods(detectors, input_img, device=None)
    else:
        ood, scores = predict_ood(detectors[0], input_img, device=None)
    
    if ood:
        return -1, scores
    
    return predict_img(clf, input_img, device=None), None