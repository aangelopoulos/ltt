import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import pathlib
import os
import pickle
import random
from tqdm import tqdm
import pdb

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def get_metrics_precomputed(est_labels,labels):
    corrects = (labels * est_labels).sum(dim=1)
    sizes = est_labels.sum(dim=1)
    corrects_temp = corrects
    corrects_temp[sizes==0] = 1
    sizes_temp = sizes
    sizes_temp[sizes==0] = 1
    fdr = 1-corrects/est_labels.float().sum(dim=1) # FDR 
    fdr[fdr > 1] = 1
    fdr[fdr < 0] = 0 # If -Inf, it means est_labels has size 0, which means FDR of 0.
    return fdr, sizes 

def get_correspondence(model_arr,dset_dict):
    corr = {}
    for i in range(model_arr.shape[0]):
        corr[i] = list(dset_dict.keys())[i]
    corr = {y:x for x,y in corr.items()}
    return corr

def fix_randomness(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

# Computes logits and targets from a model and loader
def get_scores_targets(model, loader, corr):
    scores = torch.zeros((len(loader.dataset), 80))
    labels = torch.zeros((len(loader.dataset), 80))
    i = 0
    print(f'Computing logits for model (only happens once).')
    with torch.no_grad():
        for x, label in tqdm(loader):
            if len(label)==0:
                print('I encountered an unlabeled image.')
                continue
            batch_scores = torch.sigmoid(model(x.cuda())).detach().cpu()
            scores[i:(i+x.shape[0]), :] = batch_scores

            annotations = [loader.dataset.coco.getAnnIds(imgIds=int(x)) for x in label[0]['image_id']]
            batch_labels = torch.zeros((x.shape[0],80))
            for j in range(len(annotations)):
                for annotation in loader.dataset.coco.loadAnns(annotations[j]):
                    batch_labels[j,corr[annotation['category_id']]] = 1

            labels[i:(i+x.shape[0]),:] = batch_labels 
            i = i + x.shape[0]

    keep = labels.sum(dim=1) > 0
    scores = scores[keep]
    labels = labels[keep]
    
    # Construct the dataset
    dataset_logits = torch.utils.data.TensorDataset(scores, labels.long()) 
    return dataset_logits
