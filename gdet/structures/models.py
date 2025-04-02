import torch
from collections import namedtuple
import numpy as np
from typing import TypedDict

class S2ANetHeadLossResult(TypedDict):
    loss_fam_cls:  "list[torch.Tensor]"
    loss_fam_bbox: "list[torch.Tensor]"
    loss_odm_cls:  "list[torch.Tensor]"
    loss_odm_bbox: "list[torch.Tensor]"

class S2ANetHeadForwardResult(TypedDict):
    fam_cls_scores: "list[torch.Tensor]" 
    fam_bbox_preds: "list[torch.Tensor]" 
    refine_anchors: "list[torch.Tensor]" 
    odm_cls_scores: "list[torch.Tensor]" 
    odm_bbox_preds: "list[torch.Tensor]"

class ConS2ANetHeadForwardResult(S2ANetHeadForwardResult):
    odm_contrast_feats: "list[torch.Tensor]"
    fam_sample_results: "list[torch.Tensor]"
    odm_sample_results: "list[torch.Tensor]"
    num_level_anchors: "list[torch.Tensor]"    
    
OdmLossResult = namedtuple('OdmLossResult', ['loss_cls', 'loss_bbox', "loss_contrast"])    
class TrainImageInfo(TypedDict):
    # Unprocess or Processing ImageInfo
    img_info: "AnnImageInfo"
    ann_info: "AnnItemDict"
    
    # pre pipeline
    proposals: "list"
    img_prefix: "str"
    seg_prefix: "str"
    proposal_file: "str"
    bbox_fields: "list"
    mask_fields: "list"
    seg_fields: "list"
    addtion: "bool"

    # LoadImageFromFile
    filename: "str"
    ori_filename: "str"
    ori_shape: "tuple"
    img_fields: "list[str]"
    img: "np.ndarray" # After DefaultFormatBundle, type is DataContainer

    ## LoadAnnotations
    gt_labels: "np.ndarray"  # After DefaultFormatBundle, type is DataContainer
    gt_bboxes: "np.ndarray"  # After DefaultFormatBundle, type is DataContainer
    gt_bboxes_ignore: "np.ndarray"
    
    # Resize
    scale: "tuple"
    scale_idx: "int"

    # RotatedResize
    scale_factor: "float"
    img_shape: "tuple"
    pad_shape: "tuple"
    keep_ratio: "bool"

    # PesudoRotatedRandomFlip
    flip: "bool"
    flip_direction: "str"

    # RandomRotate
    rotate: "bool"
    rotate_angle: "float" # rad

    # Normalize
    img_norm_cfg: "dict[str, float]"
    pad_shape: "tuple[int]"
    pad_fixed_size: "tuple"
    pad_size_divisor: "int"

    # Collect
    img_metas: "TrainImageMeta" # After Collect, type is DataContainer

class TrainImageMeta(TypedDict):
    filename: "str"
    ori_filename: "str"
    ori_shape: "tuple"
    img_shape: "tuple"
    pad_shape: "tuple"
    scale_factor: "float"
    flip: "bool"
    flip_direction: "str"
    img_norm_cfg: "dict[str, float]"
    
class ProcessedImageInfo(TypedDict):
    img_metas: "DataContainer| TrainImageMeta"
    img: "DataContainer| torch.Tensor"
    gt_bboxes: "DataContainer| torch.Tensor"
    gt_labels: "DataContainer| torch.Tensor"

BBoxHeadTarget = namedtuple("BBoxHeadTarget", 
    [
        "labels_list", "label_weights_list", "bbox_targets_list", "bbox_weights_list",
        "num_total_pos", "num_total_neg", "sample_results"
    ]
)    