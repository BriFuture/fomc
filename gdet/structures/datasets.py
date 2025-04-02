from typing import TypedDict
import numpy as np
import torch
class ImageInfo(TypedDict):
    filename: "str"
    height: "int"
    width: "int"
    id: "int"
    # img_prefix: "str"

class AnnInfo(TypedDict):
    bboxes: "np.ndarray"
    labels: "np.ndarray"
    masks: "np.ndarray"
    ids: "np.ndarray"


class DataBatchItems(TypedDict):
    """items before transformed
    """
    img_info : "ImageInfo"
    ann_info : "AnnInfo"
    text: "torch.Tensor"
    img_prefix: "str"
    bbox_fields: "list[str]"
    mask_fields: "list[str]"
    seg_fields:  "list[str]"

class DataItemsAfterLoad(DataBatchItems):
    ## LoadImageFromFile
    filename: "str"
    img: "np.ndarray"
    img_shape: "tuple[int]"
    ori_shape: "tuple[int]"
    img_fileds: "list[str]"
    ## LoadAnnotations
    gt_boxes: "np.ndarray"
    ids: "np.ndarray"
    gt_labels: "np.ndarray"
    gt_masks: "np.ndarray"
    gt_semantic_seg: "np.ndarray"
    ## Normalize
    img_norm_cfg: "dict[str, list[float]]"

class DataItemsAfterResize(DataItemsAfterLoad):
    scale: "tuple[int]"
    scale_idx: int
    scale_factor: "np.ndarray"
    pad_shape: "tuple[int]"
    keep_ration: "bool"

class DataItemsAfterPad(DataItemsAfterResize):
    pad_fixed_size: "tuple"
    size_divisor: "int"

from bfcommon.data_container import DataContainer
class DataItemsAfterBundle(DataItemsAfterPad):
    img: "DataContainer"
    gt_bboxes: "DataContainer"
    gt_labels: "DataContainer"
    ids: "DataContainer"
    gt_masks: "DataContainer"
    gt_semantic_seg: "DataContainer"

class ImageMeta(TypedDict):
    filename: "str"
    ori_shape: "tuple[int]"
    img_shape: "tuple[int]"
    pad_shape: "tuple[int]"
    scale_factor: "np.ndarray"
    img_norm_cfg: "dict[str, list[float]]"
    ids: "DataContainer"

class DataItemsAfterCollect(DataItemsAfterBundle):
    img_metas: "DataContainer"

class DataTransedItems(TypedDict):
    img_metas: "DataContainer"
    img: "DataContainer"
    gt_bboxes: "DataContainer"
    gt_labels: "DataContainer"
    text: "torch.Tensor"
    