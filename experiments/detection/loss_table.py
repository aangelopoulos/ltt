# import some common libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import os, json, cv2, random, sys, traceback

import multiprocessing as mp

import pickle as pkl
from tqdm import tqdm
import pdb
from profilehooks import profile

def fix_randomness(seed=0):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

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

        n = len(roi_masks)        
        lambda1s = torch.linspace(0.5,0.8,50) # Top score threshold
        lambda2s = torch.linspace(0.3,0.7,5) # Segmentation threshold
        lambda3s = torch.logspace(-0.00436,0,25) # APS threshold
        loss_tables = torch.zeros(n,3,lambda1s.shape[0],lambda2s.shape[0],lambda3s.shape[0])

        iou_correct = 0.5

        for i in tqdm(range(lambda1s.shape[0])):
            for j in range(lambda2s.shape[0]):
                for k in range(lambda3s.shape[0]):
                    confidence_threshold = lambda1s[i] 
                    segmentation_threshold = lambda2s[j] 
                    aps_threshold = lambda3s[k] 
                    neg_m_coverages, neg_mious, neg_recalls = eval_detector(roi_masks, boxes, softmax_outputs, gt_classes, gt_masks, confidence_threshold, segmentation_threshold, aps_threshold, iou_correct )
                    loss_tables[:,0,i,j,k] = neg_m_coverages
                    loss_tables[:,1,i,j,k] = neg_mious
                    loss_tables[:,2,i,j,k] = neg_recalls
                    print(f"l1: {lambda1s[i]:.3f}, l2: {lambda2s[j]:.3f}, l3: {lambda3s[k]:.3f}, Rhat1: {loss_tables[:,0,i,j,k].mean()}, Rhat2: {loss_tables[:,1,i,j,k].mean()}, Rhat3: {loss_tables[:,2,i,j,k].mean()}")
        torch.save(loss_tables, './.cache/loss_tables.pt')
        return loss_tables


# Three risks: 1-mcoverage (APS), 1-mIOU@50, and 1-recall
def eval_detector(roi_masks, boxes, softmax_outputs, gt_classes, gt_masks, confidence_threshold, segmentation_threshold, aps_threshold, iou_correct):

    neg_m_coverages = torch.zeros((len(roi_masks),))
    neg_mious = torch.zeros((len(roi_masks),))
    neg_recalls = torch.zeros((len(roi_masks),))

    for i in range(len(roi_masks)):
        corrects, ious, unused, covered = eval_image(roi_masks[i],boxes[i],softmax_outputs[i],gt_classes[i],gt_masks[i],confidence_threshold,segmentation_threshold,aps_threshold,iou_correct)
        if corrects == None:
            continue
        neg_m_coverages[i] = 1 - covered.mean()
        neg_mious[i] = 1 - ious.mean() 
        neg_recalls[i] = 1 - corrects.sum()/gt_classes[i].shape[0]

    return neg_m_coverages, neg_mious, neg_recalls

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
    if aps_threshold == 1.0:
        sizes = torch.max(sizes,80*torch.ones_like(sizes))
    #rank_of_true = (pi == labels_ind[:index_split,None]).int().argmax(dim=1) + 1

    # Starting with the most confident mask, correspond it with a ground truth mask based on IOU
    pred_masks = pred_masks.cuda()
    gt_masks = gt_masks.cuda()
    repeated_pred_masks = pred_masks.repeat(gt_masks.shape[0],1,1,1).permute(1,0,2,3)
    repeated_gt_masks = gt_masks.repeat(pred_masks.shape[0],1,1,1)
    addition_pairwise = repeated_pred_masks + repeated_gt_masks
    intersections_pairwise = ( addition_pairwise == 2 ).float().sum(dim=2).sum(dim=2)
    unions_pairwise = ( addition_pairwise >= 1 ).float().sum(dim=2).sum(dim=2)
    ious_pairwise = intersections_pairwise/torch.max(unions_pairwise,torch.tensor([1.0,]).cuda())

    ious = torch.zeros_like(top_scores)
    corrects = torch.zeros_like(top_scores)
    covered = torch.zeros_like(top_scores)
    unused = torch.tensor(range(gt_masks.shape[0]))
    for index in indices:
        if unused.shape[0] == 0:
            break
        _iou = ious_pairwise[index][unused]
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
