# import some common libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import os, json, cv2, random, sys, traceback

import pickle as pkl
import pdb

# Calculates the max IOU with respect to any mask.
def eval_image(roi_mask, box, softmax_output, gt_classes, gt_masks, threshold):
    pred_masks = roi_mask.to_bitmasks(box,gt_masks.shape[1],gt_masks.shape[2],threshold).tensor

    top_scores, indices = softmax_output.max(dim=1)[0].sort(descending=True)
    est_classes = softmax_output.argmax(dim=1)

    # Starting with the most confident mask, correspond it with a ground truth mask based on IOU
    ious = torch.zeros_like(top_scores)
    corrects = torch.zeros_like(top_scores)
    unused = torch.tensor(range(gt_masks.shape[0]))
    for index in indices:
        if unused.shape[0] == 0:
            break
        _int = (pred_masks[index] * gt_masks[unused].cuda()).sum(dim=1).sum(dim=1) 
        _uni = ((pred_masks[index].int() + gt_masks[unused].cuda().int()) >= 1).sum(dim=1).sum(dim=1)
        _iou = _int.float()/_uni.float()
        ious[index], max_iou_idx = (_iou.max().item(), _iou.argmax().item())
        corrects[index] = gt_classes[unused][max_iou_idx] == est_classes[index] 
        unused = unused[unused != unused[max_iou_idx]]

    return corrects, ious 

if __name__ == "__main__":
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
        
    lambda1s = torch.linspace(0,1,100) # Top score threshold
    lambda2s = torch.linspace(0,1,100) # Segmentation threshold
    lambda3s = torch.linspace(0,1,1000) # APS threshold
    for i in range(len(roi_masks)):
        corrects, ious = eval_image(roi_masks[i],boxes[i],softmax_outputs[i],gt_classes[i],gt_masks[i],0.5)
