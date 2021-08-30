# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import matplotlib.pyplot as plt
import os, json, cv2, random
import skimage.io as io

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from visualizer import Visualizer
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from UQHeads import UQHeads

import pdb

if __name__ == "__main__":
    # Evaluations
    annType = ['segm','bbox','keypoints']
    annType = annType[0]      #specify type here
    dataType = 'val2017'
    prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
    dataDir='./datasets/coco/'
    annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
    cocoGt=COCO(annFile)
    # display COCO categories and supercategories
    cats = cocoGt.loadCats(cocoGt.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))

    # get all images containing given categories, select one at random
    for img_id in cocoGt.imgs:
        pdb.set_trace()
        img_metadata = cocoGt.loadImgs(img_id)[0]
        img = io.imread('%s/%s/%s'%(dataDir,dataType,img_metadata['file_name']))
        ann_ids = cocoGt.getAnnIds(imgIds=[img_id,])
        anns = cocoGt.loadAnns(ann_ids)


    catIds = cocoGt.getCatIds(catNms=['person','dog','skateboard']);
    imgIds = cocoGt.getImgIds(catIds=catIds );
    imgIds = cocoGt.getImgIds(imgIds = [324158])
    img = cocoGt.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    
    im = io.imread('%s/%s/%s'%(dataDir,dataType,img['file_name']))

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NAME = "UQHeads"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.figure()
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.savefig('output.jpg')

    # TODO:  See https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    # running evaluation
    #cocoEval = COCOeval(cocoGt,cocoDt,annType)
    #cocoEval.params.imgIds  = imgIds
    #cocoEval.evaluate()
    #cocoEval.accumulate()
    #cocoEval.summarize()


