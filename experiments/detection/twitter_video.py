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
import os, json, cv2, random, sys, traceback, re
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
    recompute = True 
    with torch.no_grad():
        # Evaluations
        if recompute:
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

            # visualize videos! 

        for video_name in ["train_station", "giraffes", "street_view", "girl_on_street" ]:
            figure_dir = f'./videos_twitter/{video_name}/'
            if recompute:
                vidcap = cv2.VideoCapture(f'videos_twitter/{video_name}.mp4')
                success,img = vidcap.read()
                count = 0
                while success:
                    if len(img.shape) < 3:
                        img = img[:,:,None]

                    try:
                        outputs = predictor(img)
                    except:
                        extype, value, tb = sys.exc_info()
                        traceback.print_exc()
                        pdb.post_mortem(tb)
                        print(f"Image {count} didn't work.")
                        continue
                    
                    # Ensure everything is on cpu
                    outputs['instances'].roi_masks.tensor = outputs['instances'].roi_masks.tensor.cpu()
                    outputs['instances'].pred_boxes.tensor = outputs['instances'].pred_boxes.tensor.cpu()
                    outputs['instances'].pred_masks = outputs['instances'].pred_masks.cpu()
                    outputs['instances'].softmax_outputs = outputs['instances'].softmax_outputs.cpu()
                    outputs['instances'].class_ordering = outputs['instances'].class_ordering.cpu()
                    outputs['instances'].scores = outputs['instances'].scores.cpu()
                    outputs['instances'].pred_classes = outputs['instances'].pred_classes.cpu()
                
                    os.makedirs(figure_dir, exist_ok=True)
                    # We can use `Visualizer` to draw the predictions on the image.
                    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                    out = v.draw_instance_predictions(outputs["instances"])

                    os.makedirs('./videos_twitter/', exist_ok=True)
                    dpi = 400
                    outImg = out.get_image()[:, :, ::-1]
                    outImg = cv2.resize(outImg,dsize=(img.shape[1],img.shape[0]),interpolation=cv2.INTER_CUBIC)
                    # Plot the mask logits 
                    plot_borderless(outImg, figure_dir + f"{count}.jpg")
                    success,img = vidcap.read()
                    count += 1
                    print(f'Read frame {count}', success)

            images = [img for img in os.listdir(figure_dir) if img.endswith(".jpg")]
            images.sort(key=lambda f: int(re.sub('\D', '', f)))
            frame = cv2.imread(os.path.join(figure_dir, images[0]))
            height, width, layers = frame.shape

            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            video = cv2.VideoWriter(figure_dir + f'proc_{video_name}.mp4', fourcc, 25, (width,height))

            for image in images:
                video.write(cv2.imread(os.path.join(figure_dir, image)))

            cv2.destroyAllWindows()
            video.release()
            print("Finished with video!")
