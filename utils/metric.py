from typing import Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import Tensor
from torchmetrics import functional as F
from pytorch_ood.utils import OODMetrics
from models.ood import set_device
from sklearn.metrics import auc


def show_confusion_matrix(cm:list, classes:list=None, normalize:bool=False, fmt:str='d', save_img:Path=None,
                          figsize:tuple=(12,7), ax=None, title:str=None, **kwargs):
    if normalize:
        normalized_data = cm / np.sum(cm, axis=1)[:, None]
        df_cm = pd.DataFrame(normalized_data)
        annot_data = np.where(normalized_data == 0, '0', normalized_data.round(3))
        fmt=''
    else:
        df_cm = pd.DataFrame(cm)
        annot_data = True
        
    if classes:
        df_cm.index = [i for i in classes]
        df_cm.columns = [i for i in classes]
    if ax is None:
        fig = plt.figure(figsize=figsize)
    ax = sns.heatmap(df_cm, annot=annot_data, cmap='Spectral', fmt=fmt, ax=ax, **kwargs)
    ax.set_title(title)
    fig = ax.get_figure()
    if save_img:
        fig.savefig(save_img)
    return fig


def get_binary_auroc_and_threshold(scores:Tensor, labels:Tensor, num_thresholds:int=100) -> Tuple[Tensor, Tensor]:
    """
    Compute the area under the receiver operating characteristic (AUROC) and the optimal threshold for binary classification.
    
    Args:
    - scores: Predicted scores from the model.
    - labels: Ground truth binary labels.
    - num_thresholds: Number of thresholds for the ROC curve computation.
    
    Returns:
    - AUROC value and the optimal threshold based on the difference between TPR and FPR.
    """
    # Compute FPR, TPR, and thresholds using the ROC function
    fpr, tpr, thresholds = F.roc(preds=scores, target=labels, num_thresholds=num_thresholds)
    
    # Find the optimal threshold
    optimal_idx = torch.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Compute the AUROC
    auc = F.auroc(preds=scores, target=labels)
    
    return auc, optimal_threshold


def get_binary_aupr(scores:Tensor, labels:Tensor, thresholds:list=None) -> Tuple[Tensor, Tensor]:
    """
    Compute the area under the precision-recall curve (AUPR) and the optimal threshold for binary classification.
    
    Args:
    - scores: Predicted scores from the model.
    - labels: Ground truth binary labels.
    - thresholds: Specific thresholds for the precision-recall curve computation (optional).
    
    Returns:
    - AUPR and the optimal threshold value based on the F1 score.
    """
    # Compute precision, recall, and thresholds
    precision, recall, thresholds = F.precision_recall_curve(preds=scores, target=labels, task='binary', thresholds=thresholds)
    
    # Compute F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # Adding a small value to prevent division by zero
    
    # Handle potential NaN values by setting them to zero
    f1_scores[torch.isnan(f1_scores)] = 0
    
    # Find the optimal threshold based on F1 score
    optimal_idx = torch.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Compute the area under the precision-recall curve (AUPR)
    aupr = auc(recall, precision)
    
    return aupr, optimal_threshold


def eval_ood_detector(detector, dataloader, threshold:float=None, search_by:str='roc', device='cpu') -> dict:
    detector = set_device(detector, device=device)
    with torch.no_grad():
        metrics = OODMetrics()
        for x, y in tqdm(dataloader, 'Loading the DataLoader'):
            metrics.update(detector(x.to(device)), y.to(device))
    eval = metrics.compute()
    scores = metrics.buffer.get("scores").view(-1)
    labels = metrics.buffer.get("labels").view(-1)
    if threshold is None:
        auc, optimal_threshold = get_binary_auroc_and_threshold(scores, labels) if search_by == 'roc' else get_binary_aupr(scores, labels)
        threshold = optimal_threshold.item()
    eval['ACC'] = F.accuracy(scores, labels, task='binary', threshold=threshold).item()
    eval['PRECIS'] = F.precision(scores, labels, task='binary', threshold=threshold).item()
    eval['RECALL'] = F.recall(scores, labels, task='binary', threshold=threshold).item()
    eval['F1'] = F.f1_score(scores, labels, task='binary', threshold=threshold).item()
    eval['threshold'] = threshold
    return eval, scores, labels