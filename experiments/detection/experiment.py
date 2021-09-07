# import some common libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import os, json, cv2, random, sys, traceback
from utils import *

import multiprocessing as mp

import pickle as pkl
from tqdm import tqdm
import pdb

def get_loss_tables():
    print("Getting the loss tables!")
    with torch.no_grad():
	# Load cache
        with open('./.cache/boxes.pkl', 'rb') as f:
            boxes = pkl.load(f)

        with open('./.cache/roi_masks.pkl', 'rb') as f:
            roi_masks = pkl.load(f)

        with open('./.cache/softmax.pkl', 'rb') as f:
            softmax_outputs = pkl.load(f)

        with open('./.cache/gt_classes.pkl', 'rb') as f:
            gt_classes = pkl.load(f)

        with open('./.cache/gt_masks.pkl', 'rb') as f:
            gt_masks = pkl.load(f)
        
        lambda1s = torch.linspace(0,1,10) # Top score threshold
        lambda2s = torch.linspace(0,1,10) # Segmentation threshold
        lambda3s = torch.linspace(0.9,1,20) # APS threshold
        loss_tables = torch.zeros(3,lambda1s.shape[0],lambda2s.shape[0],lambda3s.shape[0])

        iou_correct = 0.5

        for i in tqdm(range(lambda1s.shape[0])):
            for j in range(lambda2s.shape[0]):
                for k in range(lambda3s.shape[0]):
                    confidence_threshold = lambda1s[i] 
                    segmentation_threshold = lambda2s[j] 
                    aps_threshold = lambda3s[k] 
                    neg_m_coverage, neg_miou, neg_recall = eval_detector(roi_masks, boxes, softmax_outputs, gt_classes, gt_masks, confidence_threshold, segmentation_threshold, aps_threshold, iou_correct )
                    loss_tables[0,i,j,k] = neg_m_coverage
                    loss_tables[1,i,j,k] = neg_miou
                    loss_tables[2,i,j,k] = neg_recall
                    print(f"l1: {lambda1s[i]:.3f}, l2: {lambda2s[j]:.3f}, l3: {lambda3s[k]:.3f}, nmc: {neg_m_coverage:.3f}, nmiou: {neg_miou:.3f}, nrec: {neg_recall:.3f}")
        torch.save(loss_tables, './.cache/loss_tables.pt')
        return loss_tables


# Three risks: coverage (APS), 1-mIOU@50, and 1-recall
def eval_detector(roi_masks, boxes, softmax_outputs, gt_classes, gt_masks, confidence_threshold, segmentation_threshold, aps_threshold, iou_correct):
    running_corrects = 0
    running_total = 0
    running_gt = 0
    running_sum_mean_covered_perimage = 0
    running_sum_miou_perimage = 0
    for i in range(len(roi_masks)):
        corrects, ious, unused, covered = eval_image(roi_masks[i],boxes[i],softmax_outputs[i],gt_classes[i],gt_masks[i],confidence_threshold,segmentation_threshold,aps_threshold,iou_correct)
        if corrects == None:
            continue
        running_corrects += ((corrects + (ious > iou_correct)).float() >= 2).float().sum()
        running_total += float(corrects.shape[0])
        running_gt += float(gt_masks[i].shape[0])
        running_sum_miou_perimage += float(ious.mean())
        running_sum_mean_covered_perimage += np.nan_to_num(float(covered[ious >= iou_correct].mean()))
        neg_m_coverage = 1 - running_sum_mean_covered_perimage / float(i+1) 
        neg_miou = 1 - running_sum_miou_perimage/float(i+1)
        neg_recall = running_corrects/running_gt

    return neg_m_coverage, neg_miou, neg_recall 

# Calculates per-image metrics 
def eval_image(roi_mask, box, softmax_output, gt_classes, gt_masks, confidence_threshold, segmentation_threshold, aps_threshold, iou_correct):
    pred_masks = roi_mask.to_bitmasks(box,gt_masks.shape[1],gt_masks.shape[2],segmentation_threshold).tensor

    if softmax_output.shape[0] == 0:
       return None, None, None, None 

    top_scores, indices = softmax_output.max(dim=1)[0].sort(descending=True)
    filter_idx = top_scores > confidence_threshold 

    softmax_output = softmax_output[filter_idx]
    box = box[filter_idx]
    roi_mask = roi_mask[filter_idx]
    # must reindex
    if softmax_output.shape[0] == 0:
       return None, None, None, None 

    top_scores, indices = softmax_output.max(dim=1)[0].sort(descending=True)

    est_classes = softmax_output.argmax(dim=1)

    # Setup for APS
    test_sorted, test_pi = softmax_output.sort(dim=1, descending=True)
    _, class_ranks = test_pi.sort(dim=1, descending=False)
    sizes = (test_sorted.cumsum(dim=1) <= aps_threshold).int().sum(dim=1)
    sizes = torch.max(sizes,torch.ones_like(sizes))
    #rank_of_true = (pi == labels_ind[:index_split,None]).int().argmax(dim=1) + 1

    # Starting with the most confident mask, correspond it with a ground truth mask based on IOU
    ious = torch.zeros_like(top_scores)
    corrects = torch.zeros_like(top_scores)
    covered = torch.zeros_like(top_scores)
    unused = torch.tensor(range(gt_masks.shape[0]))
    for index in indices:
        if unused.shape[0] == 0:
            break
        _int = (pred_masks[index] * gt_masks[unused]).sum(dim=1).sum(dim=1) 
        _uni = ((pred_masks[index].int() + gt_masks[unused].int()) >= 1).sum(dim=1).sum(dim=1)
        _iou = _int.float()/_uni.float()
        ious[index], max_iou_idx = (_iou.max().item(), _iou.argmax().item())
        curr_gt_class = gt_classes[unused][max_iou_idx]
        corrects[index] = curr_gt_class == est_classes[index]
        
        # Deal with APS
        covered[index] = class_ranks[index,curr_gt_class] <= (sizes[index] - 1)

        unused = unused[unused != unused[max_iou_idx]]

    return corrects, ious, unused, covered

if __name__ == "__main__":
    fix_randomness(seed=0)
    loss_tables = get_loss_tables()
    print(loss_tables)
    print("Success!")
