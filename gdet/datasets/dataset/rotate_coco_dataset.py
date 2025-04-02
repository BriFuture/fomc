import os.path as osp
import numpy as np
import pickle
import tempfile
import torch
import copy
from collections import Counter

from pycocotools.coco import COCO
import logging
from gdet.structures.datasets import ImageInfo, AnnInfo, DataBatchItems, DataTransedItems
from gdet.registries import DATASETS
from gdet.core.bbox import obb2poly_np, poly2obb_np
from .custom_dataset import CustomDataset
from .coco_dataset import CocoDataset

logger = logging.getLogger("gdet.datasets")

@DATASETS.register_module()    
class RotateCocoDataset(CocoDataset):
    def __init__(self, cfg, transforms=None):
        super().__init__(cfg, transforms)
        self.version = cfg.version
        
    def load_annotations(self, ann_file: "str"):
        coco = COCO(ann_file)
        logger.info(f"Loading coco ann file: {ann_file}")
        self.m_coco: "COCO"  = coco
        ### bsf.c original all dataset
        # self.m_cat_ids = coco.getCatIds()
        # cat2label = {cat_id['name']: i for i, cat_id in coco.cats.items()}
        catId2label = {}
        catId2name = {}
        for idx, cat_id in coco.cats.items():
            if cat_id['name'] in self.dst_classes:
                catId2label[idx] = self.dst_classes.index(cat_id['name']) + 1 
                catId2name[idx] = cat_id['name']
        self.m_catId2label = catId2label
        ### bsf.c load only needed classes
        self.m_catId2name = catId2name
        self.m_cat_ids = list(self.m_catId2label.keys())

        self.m_img_ids = coco.getImgIds()
        ### bsf.c read img infos
        data_infos = []
        for ni, i in enumerate(self.m_img_ids):
            info = coco.imgs[i]
            info['filename'] = info['file_name']
            img_info = {
                "filename":info['file_name'],
                "height": info['height'],
                "width": info['width'],
                "id": info['id'],
            }
            data_infos.append(img_info)
        return data_infos    