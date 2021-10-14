# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
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
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils
import pickle as pkl

try:
    from .UQHeads import UQHeads
except:
    from UQHeads import UQHeads

import pdb
from tqdm import tqdm

def cache_data():
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
        cfg.MODEL.ROI_HEADS.APS_THRESH = 0.999 #set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)

        # get all images and annotations 
        pred_classes = []
        pred_masks = []
        pred_roi_masks = []
        pred_boxes = []
        pred_sets = []
        pred_softmax_outputs = []
        gt_classes = []
        gt_masks = []
        for img_id in tqdm(cocoGt.imgs):
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
            pred_classes = pred_classes + [outputs['instances'].pred_classes.cpu(),]
            pred_roi_masks = pred_roi_masks + [outputs['instances'].roi_masks,]
            pred_masks = pred_masks + [outputs['instances'].pred_masks.cpu(),]
            pred_boxes = pred_boxes + [outputs['instances'].pred_boxes,]
            pred_sets = pred_sets + [outputs['instances'].pred_sets,]
            pred_softmax_outputs = pred_softmax_outputs + [outputs['instances'].softmax_outputs.cpu(),]
            gt_classes = gt_classes + [torch.tensor([label_map[ann['category_id']] for ann in anns]).cpu(),]
            gt_masks = gt_masks + [gt_masks_singleimage,]
            #    threshold=0.5
            #    outputs["instances"].pred_masks = pred_roi_masks[-1].to_bitmasks(pred_boxes[-1],img.shape[0],img.shape[1],threshold).tensor
            #    break
        
	# Save cache
        os.makedirs('./.cache/', exist_ok=True)  
        with open('./.cache/boxes.pkl', 'wb') as f:
            pkl.dump(pred_boxes, f)

        with open('./.cache/roi_masks.pkl', 'wb') as f:
            pkl.dump(pred_roi_masks, f)

        with open('./.cache/softmax.pkl', 'wb') as f:
            pkl.dump(pred_softmax_outputs, f)

        with open('./.cache/gt_classes.pkl', 'wb') as f:
            pkl.dump(gt_classes, f)

        with open('./.cache/gt_masks.pkl', 'wb') as f:
            pkl.dump(gt_masks, f)

if __name__ == "__main__":
    cache_data()
