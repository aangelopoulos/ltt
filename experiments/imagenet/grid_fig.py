import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import torch
import torchvision as tv
import argparse
import time
import numpy as np
from scipy.stats import binom
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from utils import *
import seaborn as sns
from core.concentration import *
import pdb

def grid_fig_plot(mosaic,alphas,classes):
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,3))


# r is the number of images to get
def get_images(r, images, classes_array, top_scores, labels, corrects, alpha, delta, num_calib, maxiter, transform):
    total=top_scores.shape[0]
    m=1000
    perm = torch.randperm(total)
    top_scores = top_scores[perm]
    labels = labels[perm]
    corrects = corrects[perm].float()
    calib_scores, val_scores = (top_scores[0:num_calib], top_scores[num_calib:])
    calib_labels, val_labels = (labels[0:num_calib], labels[num_calib:])
    calib_corrects, val_corrects = (corrects[0:num_calib], corrects[num_calib:])

    calib_scores, indexes = calib_scores.sort()
    calib_corrects = calib_corrects[indexes] 
    calib_accuracy = (calib_corrects.flip(dims=(0,)).cumsum(dim=0)/(torch.tensor(range(num_calib))+1)).flip(dims=(0,))
    calib_abstention_freq = (torch.tensor(range(num_calib))+1).float().flip(dims=(0,))/num_calib

    starting_index = ((1-calib_accuracy)/calib_abstention_freq < alpha).nonzero(as_tuple=True)[0][0]

    pfdr_pluses = torch.tensor( [ pfdr_ucb(num_calib, m, calib_accuracy[i], calib_abstention_freq[i], delta, maxiter) for i in range(starting_index, num_calib) ] )

    valid_set_index = max((pfdr_pluses > alpha).nonzero(as_tuple=True)[0][0]+starting_index-1, 0)  # -1 because it needs to be <= alpha
    
    lhat = calib_scores[valid_set_index]

    val_predictions = val_scores > lhat

    # Get the r lowest-score val images 
    val_scores = val_scores[val_predictions]
    val_images = images[perm][num_calib:][val_predictions]
    val_labels = val_labels[val_predictions]

    _, indexes = val_scores.sort()
    val_images = val_images[indexes]
    val_labels = val_labels[indexes]

    return_image_filenames = [val_images[i] for i in range(r)] 
    return_image_classes = [classes_array[val_labels[i]] for i in range(r)] 

    return_images = [transform(Image.open(image_filename)).permute(1,2,0) for image_filename in return_image_filenames]
    for i in range(len(return_images)):
        if return_images[i].shape[2] == 1:
            return_images[i] = torch.cat((return_images[i],return_images[i],return_images[i]), dim=2)
    return_image_row = torch.cat(return_images,dim=1)
    
    return return_image_row, return_image_classes

def experiment(alphas,delta,num_calib,maxiter,imagenet_val_dir):
    transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                    ])
    dataset = torchvision.datasets.ImageFolder(imagenet_val_dir, transform)
    image_filenames = np.array([x[0] for x in dataset.imgs])
    image_classes = np.array([x[1] for x in dataset.imgs])
    dataset_precomputed = get_logits_dataset('ResNet152', 'Imagenet', imagenet_val_dir)
    print('Dataset loaded')
    
    classes_array = get_imagenet_classes()
    T = platt_logits(dataset_precomputed)
    
    logits, labels = dataset_precomputed.tensors
    top_scores, top_classes = (logits/T.cpu()).softmax(dim=1).max(dim=1)
    corrects = top_classes==labels

    plot_rows_classes = [get_images(5, image_filenames, classes_array, top_scores, labels, corrects, alpha, delta, num_calib, maxiter, transform) for alpha in alphas]

    grid_plot = torch.cat([x[0] for x in plot_rows_classes], dim=0)
    list_classes = [ x[1] for x in plot_rows_classes ]
    
    print(list_classes)

    plt.imshow(grid_plot)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./outputs/grid_fig.pdf')

def platt_logits(calib_dataset, max_iters=10, lr=0.01, epsilon=0.01):
    calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=1024, shuffle=False, pin_memory=True) 
    nll_criterion = nn.CrossEntropyLoss().cuda()

    T = nn.Parameter(torch.Tensor([1.3]).cuda())

    optimizer = optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            optimizer.zero_grad()
            x = x.cuda()
            x.requires_grad = True
            out = x/T
            loss = nll_criterion(out, targets.long().cuda())
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break
    return T 

if __name__ == "__main__":
    sns.set(palette='pastel',font='serif')
    sns.set_style('white')
    fix_randomness(seed=0)

    imagenet_val_dir = '/scratch/group/ilsvrc/val'

    alphas = [0.025,0.05, 0.1]
    delta = 0.1
    maxiter = int(1e3)
    num_calib = 30000
    
    experiment(alphas,delta,num_calib,maxiter,imagenet_val_dir)
