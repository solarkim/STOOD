from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset, Subset
import torchvision
from torchvision import transforms


def imshow(inputs:Tensor, title:str=None, mean:list=None, std:list=None):
    """Shows a list of images in grid format. It can revert the inputs again to show.

    Args:
        inputs (list): a list of images in tensors. The images are normalized
        title (str, optional): a title. Defaults to None.
        mean (list, optional): normalized mean - np.array([0.485, 0.456, 0.406]). Defaults to None.
        std (list, optional): normalized std - np.array([0.229, 0.224, 0.225]). Defaults to None.
    """
    grid = torchvision.utils.make_grid(inputs)
    grid = grid.numpy().transpose((1, 2, 0))
    # grid = grid.permute(1, 2, 0)
    if mean:    # revert it
        grid = std * grid + mean
    grid = np.clip(grid, 0, 1)
    plt.imshow(grid)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def show_imageset(dataset:Dataset, indices:Sequence[int], title:str=None, transform:transforms.Compose=None, mean:list=None, std:list=None):
    """Shows a subset of images in grid format. It can revert the inpute again to show.

    Args:
        dataset (Dataset): Dataset to retrieve
        indices (Sequence[int]): indices to select
        title (str, optional): a title. Defaults to None.
        mean (list, optional): normalized mean - np.array([0.485, 0.456, 0.406]). Defaults to None.
        std (list, optional): normalized std - np.array([0.229, 0.224, 0.225]). Defaults to None.
    """
    subset = Subset(dataset, indices)
    if transform:
        images = [transform(img) for img, _ in subset]
    else:
        images = [img for img, _ in subset]
    images = [img[:3, :, :] for img in images]  # 일부 이미지는 alpha 채널 포험하여 stack시 오류생김

    batch = torch.stack(images, dim=0)
    imshow(batch, title=title, mean=mean, std=std)


def visualize_model(model:nn.Module, dataloader, classes:list, num_images=6,):
    raise NotImplementedError
    device = model.device
    was_training = model.training
    model.eval()
    fig = plt.figure()

    with torch.no_grad():
        for idx_batch, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(-1)

            for j in range(inputs.size()[0]):
                idx_batch += 1
                ax = plt.subplot(num_images//2, 2, idx_batch)
                ax.axis('off')
                ax.set_title(f'predicted: {classes[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if idx_batch == num_images:
                    model.train(mode=was_training)
                    return
    model.train(mode=was_training)


# def get_mean_std(loader):
#     # var[X] = E[X**2] - E[X]**2
#     channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

#     for data, _ in loader:
#         channels_sum += torch.mean(data, dim=[0, 2, 3])
#         channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
#         num_batches += 1

#     mean = channels_sum / num_batches
#     std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

#     return mean, std

# train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
# mean, std = get_mean_std(train_loader)
# print(mean)
# print(std)


def makegrid(output:torch.Tensor, numrows:int, figsize:tuple=(20, 5)):
    outer = (torch.Tensor.cpu(output).detach())
    plt.figure(figsize=figsize)
    b = np.array([]).reshape(0, outer.shape[2])
    c = np.array([]).reshape( numrows * outer.shape[2], 0)
    i=0
    j=0
    while(i < outer.shape[1]):
        img=outer[0][i]
        b = np.concatenate((img,b),axis=0)
        j += 1
        if j == numrows:
            c = np.concatenate((c,b), axis=1)
            b = np.array([]).reshape(0,outer.shape[2])
            j = 0
        i+=1
    return c


from sklearn import metrics
def evaluate_f1_score(self):

    # Get the hypothesis and predictions
    all_target, all_preds = self.evaluate()

    table = metrics.classification_report(
        all_target,
        all_preds,
        labels = [int(a) for a in list(self.ids2labels.keys())],
        target_names = list(self.labels2ids.keys()),
        zero_division = 0,
        digits=self.classification_report_digits,
    )
    print(table)

    # Write logs
    self.__openLogs()
    self.logs_file.write(table + "\n")
    self.logs_file.close()

    print("Logs saved at: \033[93m" + self.logs_path + "\033[0m")
    print("\033[93m[" + self.model_name + "]\033[0m Best model saved at: \033[93m" + self.best_path + "\033[0m" + " - Accuracy " + "{:.2f}".format(self.best_acc*100))

    return all_target, all_preds