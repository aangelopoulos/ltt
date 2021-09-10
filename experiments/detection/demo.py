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
from visualizer import Visualizer
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils
import pickle as pkl

from UQHeads import UQHeads

import pdb
from tqdm import tqdm

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

        # Set lambda hat

        # get all images and annotations 
        #img_id = 470618 # 309615,261800,309391,509005,178749,356060,310338,231863,493227,519136
        img_id = cocoGt.getImgIds()[10]
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

        # Ensure everything is on cpu
        outputs['instances'] = outputs['instances'].to('cpu')
        tokeep = outputs["instances"].softmax_outputs.max(dim=1)[0] > 0.94444335 
        outputs['instances'].roi_masks.tensor = outputs['instances'].roi_masks.tensor[tokeep]
        outputs['instances'].pred_boxes.tensor = outputs['instances'].pred_boxes.tensor[tokeep]
        outputs['instances'].pred_sets = outputs['instances'].pred_sets[tokeep]
        outputs['instances'].pred_masks = outputs['instances'].roi_masks.to_bitmasks(outputs['instances'].pred_boxes,img.shape[0],img.shape[1],0.22444426).tensor
        outputs['instances'].softmax_outputs = outputs['instances'].softmax_outputs[tokeep]
        outputs['instances'].scores = outputs['instances'].scores[tokeep]
        outputs['instances'].class_ordering = outputs['instances'].class_ordering[tokeep]
        
        # We can use `Visualizer` to draw the predictions on the image.
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"])

        plt.figure()
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.savefig('output.jpg')
