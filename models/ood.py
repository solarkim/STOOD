import pickle
import inspect
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor
from pytorch_ood.api import Detector
from pytorch_ood.detector import (
    ASH,        # 2023
    SHE,        # 2023
    ViM,        # 2022
    KLMatching, # 2022
    MaxLogit,   # 2022
    Entropy,    # 2021
    DICE,       # 2021
    RMD,        # 2021
    KNN,        # 2022
    ReAct,      # 2021
    EnergyBased,    # 2020
    Mahalanobis,    # 2018
    ODIN,       # 2017
    MaxSoftmax, # 2017
    MCD,         # 2016 Monte Carlo Dropout
    TemperatureScaling, # 2017
    OpenMax,    # 2015
)


DETECTORS = {
            "ASH": [ASH, dict()],
            "SHE": [SHE, dict()],
            "ViM": [ViM, dict(d=64)],
            "DICE": [DICE, dict(p=0.65)],
            "RMD": [RMD, dict()],
            "KNN": [KNN, dict()],
            "ReAct": [ReAct, dict()],
            "EnergyBased": [EnergyBased, dict()],
            "KLMatching": [KLMatching, dict()],
            "MaxLogit": [MaxLogit, dict()],
            "Mahalanobis": [Mahalanobis, dict(eps=0.002)],
            "MaxSoftmax": [MaxSoftmax, dict()],
            "Entropy": [Entropy, dict()],
            "ODIN": [ODIN, dict(eps=0.002)],
            "MCD": [MCD, dict()],
            "TemperatureScaling": [TemperatureScaling, dict()],
            "OpenMax": [OpenMax, dict()],
            }


# Implement to provide features. Use 'get_graph_node_names' function to find out the return nodes.
# https://pytorch.org/vision/stable/feature_extraction.html
from functools import partial

def features(batch, feature_list, return_nodes:list):
    if return_nodes and isinstance(return_nodes, list) and len(return_nodes) == 1:
        out = feature_list(batch).get(return_nodes[0])
        return out.view(-1, np.prod(out.shape[1:]))
    else:
        out = feature_list(batch)
        out_list = []
        for node in out.values():
            out_list.append(node.view(-1, np.prod(node.shape[1:])))
        return torch.cat(out_list, dim=0)


def check_feature(model, return_nodes:list=None):
    if hasattr(model, 'features'):  # model has its own feature method
        return model.features
    elif hasattr(model, 'return_nodes') is False:   # model has no list to extract features. Use as it is.
        return model
    
    if return_nodes is None:
        return_nodes = model.return_nodes
    
    if hasattr(model, 'model'): # if the model has a pretrained CNN model inside
        model = model.model
    feature_list = create_feature_extractor(model, return_nodes=return_nodes)
    return partial(features, feature_list=feature_list, return_nodes=return_nodes)    


def get_detector_instance(name:str, model:nn.Module, device:str='cpu', **kwargs) -> Detector:
    model_factory, model_kwargs = DETECTORS.get(name)
    if model_factory is None:
        raise NotImplementedError
    model_kwargs.update(kwargs)
    model = model.eval().to(device)
    # detectors requiring features
    if name in ['Mahalanobis', 'ViM', 'SHE', 'DICE', 'RMD']:
        return model_factory(model=check_feature(model), **model_kwargs)
    elif name in ['ReAct']:
        return model_factory(backbone=check_feature(model), **model_kwargs)
    elif name in ['ASH']:
        raise NotImplementedError
    return model_factory(model=model, **model_kwargs)


def set_device(detector:Detector, device:str='cpu') -> Detector:
    for name, params in detector.__dict__.items():
        # check if methods are like model.features
        if name == 'model':
            if inspect.ismethod(params):    # 'features' method
                detector.__dict__[name] = params.__self__.to(device).features
            elif hasattr(params, 'keywords') and 'feature_list' in params.keywords:
                params.keywords['feature_list'] = params.keywords['feature_list'].to(device)
            elif isinstance(params, (torch.Tensor, nn.Module)):
                detector.__dict__[name] = params.to(device)
        elif isinstance(params, (torch.Tensor, nn.Module)):
            detector.__dict__[name] = params.to(device)
    return detector


def save_detector(path:str, detector, device:str='cpu'):
    detector = set_device(detector, device)
    with open(str(path), 'wb') as fp:
        pickle.dump(detector, fp)


def load_detector(path:str, device:str='cpu') -> Detector:
    with open(str(path), 'rb') as fp:
        detector = pickle.load(fp)
    return set_device(detector, device)