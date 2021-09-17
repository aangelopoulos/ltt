# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.layers import paste_masks_in_image
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data import MetadataCatalog
setup_logger()

# import some common libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import os, json, cv2, random, sys, traceback
import skimage.io as io

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from visualizer import Visualizer
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils
import pickle as pkl

from UQHeads import UQHeads

import pdb
from tqdm import tqdm
import seaborn as sns

def plot_borderless(img, name, cmap=None):
    dpi=400
    fig = plt.figure(figsize=(img.shape[1]/dpi,img.shape[0]/dpi))
    plt.imshow(img,interpolation='nearest',cmap=cmap)
    sns.despine(top=True,right=True,left=True,bottom=True)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_frame_on(False)
    ax.set_position([0, 0, 1, 1])
    plt.margins(0,0)
    plt.savefig(name, dpi=dpi, pad_inches=0, bbox_inches=0)
    plt.close(fig)

if __name__ == "__main__":
    with torch.no_grad():
        # Evaluations
        annType = ['segm','bbox','keypoints']
        annType = annType[0]      #specify type here
        dataType = 'val2017'
        prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
        dataDir='./datasets/coco/'
        annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
        cocoGt=COCO(annFile)

        # Get the label mapping from COCO to detectron2 standard
        label_map = MetadataCatalog['coco_2017_val'].thing_dataset_id_to_contiguous_id

        # Load the model
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NAME = "UQHeads"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 #set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)

        # visualize images, mask logits, mask and class 
        for img_id in [87038,]:
            img_metadata = cocoGt.loadImgs(img_id)[0]
            img = io.imread('%s/%s/%s'%(dataDir,dataType,img_metadata['file_name']))
            if len(img.shape) < 3:
                img = img[:,:,None]

            ann_ids = cocoGt.getAnnIds(imgIds=[img_id,])
            anns = cocoGt.loadAnns(ann_ids)
            try:
                outputs = predictor(img)
            except:
                extype, value, tb = sys.exc_info()
                traceback.print_exc()
                pdb.post_mortem(tb)
                print(f"Image {img_id} didn't work.")
                continue
            
            gt_masks_singleimage = []
            for ann in anns:
                try:
                    rleobj = maskUtils.frPyObjects([ann['segmentation'][0]],img_metadata['height'],img_metadata['width'])
                except:
                    rleobj = maskUtils.frPyObjects([ann['segmentation']],img_metadata['height'],img_metadata['width'])
                gt_masks_singleimage = gt_masks_singleimage + [maskUtils.decode(rleobj),]
            try:
                gt_masks_singleimage = torch.tensor(np.concatenate(gt_masks_singleimage,axis=2)).permute(2,0,1).cpu()
            except:
                continue

            # Ensure everything is on cpu
            outputs['instances'].roi_masks.tensor = outputs['instances'].roi_masks.tensor.cpu()
            outputs['instances'].pred_boxes.tensor = outputs['instances'].pred_boxes.tensor.cpu()
            pred_classes = outputs['instances'].pred_classes.cpu()
            # Get the outputs
            pred_roi_masks = outputs['instances'].roi_masks
            pred_masks = outputs['instances'].pred_masks.cpu()
            pred_boxes = outputs['instances'].pred_boxes
            pred_sets = outputs['instances'].pred_sets
            pred_softmax_outputs = outputs['instances'].softmax_outputs.cpu()
            gt_classes = [torch.tensor([label_map[ann['category_id']] for ann in anns]).cpu(),]
            gt_masks = gt_masks_singleimage
        
            figure_dir = './explanatory_figure/'
            os.makedirs(figure_dir, exist_ok=True)
            # Plot the original image
            plot_borderless(img, figure_dir + f"{img_id}_input.jpg", cmap=None)
            paste = retry_if_cuda_oom(paste_masks_in_image)
            pastemasks = paste(
                        pred_roi_masks.tensor,
                        pred_boxes,
                        (gt_masks.shape[1], gt_masks.shape[2]),
                        threshold=-1
            ).to(float)/255
            # Plot the mask logits 
            plot_borderless(pastemasks.sum(dim=0), figure_dir + f"{img_id}_mask_prethreshold.jpg", cmap='gray')
            # Plot the largest mask
            idx_largest = pastemasks.sum(dim=1).sum(dim=1).argmax()
            plot_borderless((pastemasks[idx_largest] > 0.5).float(), figure_dir + f"{img_id}_one_object.jpg", cmap='gray')
