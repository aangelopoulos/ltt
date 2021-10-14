import numpy as np
from typing import Tuple, Union, Dict, List, Optional
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, batched_nms, get_norm
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads, FastRCNNOutputLayers
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.structures import Boxes, ImageList, Instances

import pdb

def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    aps_thresh: float,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.
    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.
    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image, aps_thresh
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    aps_thresh: float,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).
    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.
    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero(as_tuple=False)
    idx_tokeep = filter_mask.int().sum(dim=1) >= 1
    
    # Subset by the scores that meet the criteria
    scores = scores[idx_tokeep]
    if scores.shape[0] == 0:
        result = Instances(image_shape)
        result.pred_boxes = Boxes(boxes[0:0,0,:])
        result.scores = scores
        result.pred_classes = scores[:,0].long()
        result.pred_sets = scores.bool()
        result.class_ordering = scores.long()
        result.softmax_outputs = scores
        return result, filter_inds[:, 0]

    top_scores = scores.max(dim=1)[0]#scores[filter_mask]

    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[idx_tokeep,scores.argmax(dim=1)]#boxes[filter_mask]
    # remove the scores that don't satisfy the threshold

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, top_scores, scores.argmax(dim=1), nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, top_scores, scores, filter_inds = boxes[keep], top_scores[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = top_scores
    result.pred_classes = filter_inds[:, 1]

    # Construct prediction sets
    scores = scores/scores.sum(dim=1).unsqueeze(dim=1)
    sortd, pi = scores.sort(dim=1, descending=True)
    cumsum = sortd.cumsum(dim=1)
    try:
        sizes = (cumsum > aps_thresh).int().argmax(dim=1) + 1
        sizes[sizes == 0] = 1 # Size at least 1
    except:
        sizes = torch.tensor([]) 
    # The prediction set will be a float to be compatible with the inbuilt detectron2 code.
    try:
        result.pred_sets = torch.cat([ to_set_mask(pi[i][0:sizes[i]], scores.shape[1]) for i in range(sizes.shape[0]) ], dim=0)
    except:
        result.pred_sets = torch.tensor([])
    result.class_ordering = pi 
    result.softmax_outputs = scores

    return result, filter_inds[:, 0]

def to_set_mask(pset, num_classes):
    pset_return = torch.zeros((num_classes,))
    pset_return[pset] = 1
    return pset_return.unsqueeze(dim=0) > 0 

class UQFastRCNNOutputLayers(FastRCNNOutputLayers):
    """
        A Standard ROIHeads which contains an addition of DensePose head.
    """
    def __init__(
        self,
        cfg,
        input_shape: ShapeSpec,
        aps_thresh: float,
    ):
        super().__init__(cfg, input_shape)
        self.aps_thresh = aps_thresh

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            self.aps_thresh,
        )

@ROI_HEADS_REGISTRY.register()
class UQHeads(StandardROIHeads):
    """
        A Standard ROIHeads which contains an addition of DensePose head.
    """
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        aps_thresh = cfg['MODEL']['ROI_HEADS']['APS_THRESH']
        self._init_uq_head(cfg, input_shape, aps_thresh)

    def _init_uq_head(self, cfg, input_shape, aps_thresh):
        self.aps_thresh = aps_thresh
        self.box_predictor = UQFastRCNNOutputLayers(cfg, self.box_head.output_shape, self.aps_thresh)

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ):
        instances, losses = super().forward(images, features, proposals, targets)
        return instances, losses

    # This is the forward method from StandardROIHeads
    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)

        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances
